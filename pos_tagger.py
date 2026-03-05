#@Author: Akash Manna
#@Date: 04/03/2026
import wandb
import nltk
from nltk.corpus import brown
import random
import torch
from torch.utils.data import Dataset, DataLoader
nltk.download('brown')
nltk.download('universal_tagset')

sentences = brown.tagged_sents(tagset='universal')

random.seed(42)
sentences = list(sentences)
random.shuffle(sentences)



class POSDataset(Dataset):
    def __init__(self, sentences, word_vocab, tag_vocab, context_size=2):
        self.C = context_size

        self.samples = []
        for sent in sentences:
            words= [word.lower() for word, _ in sent]
            tags= [tag for _, tag in sent]

            for i, (word, tag) in enumerate(zip(words, tags)):
                window_ids= []
                for offset in range(-self.C, self.C+ 1):
                    j= i+ offset
                    if j<0 or j>= len(words):
                        window_ids.append(0)
                    else:
                        window_ids.append(word_vocab.get(words[j], 0)) #0: UKN/PAD

                self.samples.append((window_ids, tag_vocab[tag]))   # ← inside word loop
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        window_ids, tag_id = self.samples[idx]
        # Return integer indices — EmbeddingLayer in POSTagger does the lookup
        return torch.tensor(window_ids, dtype=torch.long), torch.tensor(tag_id, dtype=torch.long)


#Step-1: Setup the dataset

def build_embedding_matrix(word_vocab, pt_path, embed_dim=350):
    """
    Aligns a saved .pt dict (keys: 'embeddings', 'vocab') to the POS tagger's
    word_vocab. Row i in the output = embedding for the word at index i.
    """
    checkpoint = torch.load(pt_path, weights_only=False)
    svd_tensor = checkpoint['embeddings']       # Tensor (V_svd, embed_dim)
    svd_vocab  = checkpoint['vocab']            # list of V_svd words

    svd_word2idx = {word: i for i, word in enumerate(svd_vocab)}

    vocab_size = len(word_vocab)
    matrix = torch.zeros(vocab_size, embed_dim) # index 0 (PAD) stays zero

    found, oov = 0, 0
    for word, idx in word_vocab.items():
        if word== '<PAD>':
            pass
        if word in svd_word2idx:
            matrix[idx] = svd_tensor[svd_word2idx[word]]
            found += 1       
        else:
            matrix[idx] = torch.empty(embed_dim).uniform_(-0.1, 0.1)
            oov += 1

    print(f"Embedding alignment — Found: {found} | OOV: {oov} | Total: {vocab_size}")
    return matrix.numpy()   # POSDataset converts to tensor internally


def save_pretrained_as_pt(pt_path, source='glove', dim=100):
    """
    Downloads a pre-trained embedding via gensim and saves it as a .pt dict
    with keys 'embeddings' (Tensor) and 'vocab' (list), matching SVD/CBOW format.

    source : 'glove'    → GloVe Wikipedia+Gigaword (glove-wiki-gigaword-{dim})
             'fasttext' → fastText Common Crawl (fasttext-wiki-news-subwords-{dim} or
                          crawl-300d-2M-subword via gensim)
    dim    : 50 | 100 | 200 | 300  (GloVe); 300 (fastText)
    """
    import os
    import gensim.downloader as api

    if os.path.exists(pt_path):
        print(f"Pre-trained .pt already exists at {pt_path}, skipping download.")
        return

    if source == 'glove':
        model_name = f'glove-wiki-gigaword-{dim}'
    elif source == 'fasttext':
        model_name = f'fasttext-wiki-news-subwords-{dim}'
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'glove' or 'fasttext'.")

    print(f"Downloading {model_name} via gensim (one-time download)...")
    wv = api.load(model_name)          # gensim KeyedVectors

    vocab  = wv.index_to_key           # list of words in order
    tensor = torch.tensor(wv.vectors, dtype=torch.float32)  # (V, dim)

    os.makedirs(os.path.dirname(pt_path) or '.', exist_ok=True)
    torch.save({'embeddings': tensor, 'vocab': vocab}, pt_path)
    print(f"Saved {source} embeddings to {pt_path} | Shape: {tensor.shape}")

import torch.nn as nn

#Step-2: embeddings model

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_matrix, freeze=False):
        """
        embedding_matrix : numpy array (vocab_size, embed_dim)
        freeze           : if True, embeddings are not updated during training
        """
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            requires_grad=not freeze
        )

    def forward(self, x):
        # x: (batch, 2C+1) → (batch, 2C+1, embed_dim) → (batch, (2C+1)*embed_dim)
        return self.embedding(x).view(x.size(0), -1)


#Step-3: classifier

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tags, dropout=0.3):
        """
        input_dim  : (2*C+1) * embed_dim
        hidden_dim : size of each hidden layer
        num_tags   : number of POS tags (12 for universal tagset)
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tags)
            # No Softmax here — nn.CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        return self.network(x)


#Step-4: Whole model

class POSTagger(nn.Module):
    def __init__(self, embedding_matrix, context_size, hidden_dim, num_tags, freeze=False):
        super().__init__()
        embed_dim = embedding_matrix.shape[1]
        input_dim = (2 * context_size + 1) * embed_dim

        self.embedding = EmbeddingLayer(embedding_matrix, freeze=freeze)
        self.classifier = MLPClassifier(input_dim, hidden_dim, num_tags)

    def forward(self, window_ids):
        # window_ids: (batch, 2C+1)
        x = self.embedding(window_ids)   # (batch, input_dim)
        return self.classifier(x)        # (batch, num_tags)


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

def train(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu', use_wandb=False):
    model.to(device)
    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, epochs + 1):
        # ── Training ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)           # (batch, num_tags)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()
            total      += y_batch.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validation ──
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:>3}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if use_wandb:
            wandb.log({
                "epoch":      epoch,
                "train/loss": train_loss,
                "train/acc":  train_acc,
                "val/loss":   val_loss,
                "val/acc":    val_acc,
            })

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best weights before returning
    model.load_state_dict(best_state)
    print(f"\nRestored best model (val loss: {best_val_loss:.4f})")
    if use_wandb:
        wandb.run.summary["best_val_loss"] = best_val_loss
    return model


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

def evaluate(model, loader, criterion, device='cpu'):
    """Returns (loss, accuracy) — used for val during training."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()
            total      += y_batch.size(0)
    return total_loss / total, correct / total


def evaluate_test(model, loader, tag_vocab, device='cpu', use_wandb=False,
                  word_vocab=None, n_errors=5):
    """Full test evaluation: Accuracy + Macro-F1 + Confusion Matrix.

    word_vocab : optional word→id dict; when provided, error examples show
                 the decoded context window instead of raw integer ids.
    n_errors   : number of misclassified examples to print (and log to wandb).
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    # id→word lookup for error decoding
    id2word = {v: k for k, v in word_vocab.items()} if word_vocab else {}

    model.eval()
    all_preds, all_labels = [], []
    error_examples = []          # (window_str, true_tag_str, pred_tag_str)

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits  = model(x_batch)
            preds   = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.tolist())

            # Collect error examples while we still have x_batch handy
            if len(error_examples) < n_errors:
                x_cpu = x_batch.cpu()
                for window_ids, true_id, pred_id in zip(
                        x_cpu.tolist(), y_batch.tolist(), preds.tolist()):
                    if true_id != pred_id and len(error_examples) < n_errors:
                        error_examples.append((window_ids, true_id, pred_id))

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm       = confusion_matrix(all_labels, all_preds)

    idx2tag   = {v: k for k, v in tag_vocab.items()}
    tag_names = [idx2tag[i] for i in range(len(tag_vocab))]

    print(f"\n{'─'*40}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Macro-F1 Score: {macro_f1:.4f}")
    print(f"{'─'*40}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"Tags: {tag_names}")
    print(cm)
    print(classification_report(all_labels, all_preds, target_names=tag_names))

    # ── Error analysis ────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"  Sample Misclassifications (first {n_errors})")
    print(f"{'─'*40}")
    error_rows = []
    for window_ids, true_id, pred_id in error_examples:
        words_str = ' '.join(id2word.get(i, f'<{i}>') for i in window_ids)
        true_str  = idx2tag[true_id]
        pred_str  = idx2tag[pred_id]
        print(f"  Context : [{words_str}]")
        print(f"  True tag: {true_str}  →  Predicted: {pred_str}\n")
        error_rows.append([words_str, true_str, pred_str])

    if use_wandb and error_rows:
        err_table = wandb.Table(
            columns=["context_window", "true_tag", "predicted_tag"],
            data=error_rows
        )
        wandb.log({"test/error_examples": err_table})
    # ─────────────────────────────────────────────────────────────────────────

    if use_wandb:
        # ── scalar summary ──
        wandb.run.summary["test/accuracy"] = acc
        wandb.run.summary["test/macro_f1"] = macro_f1

        # ── per-tag metrics table ──
        per_tag_f1 = f1_score(all_labels, all_preds, average=None)
        metrics_table = wandb.Table(
            columns=["tag", "f1"],
            data=[[tag_names[i], float(per_tag_f1[i])] for i in range(len(tag_names))]
        )
        wandb.log({"test/per_tag_f1": metrics_table})

        # ── confusion matrix ──
        # Pass integer indices + class_names; do NOT convert to strings first.
        wandb.log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_preds,
                class_names=tag_names,
            )
        })

    return acc, macro_f1, cm

if __name__ == "__main__":
    CONTEXT_SIZE = 4   # ← change this to 1, 2, 3, etc.
    BATCH_SIZE   = 64

    # Split first so vocab is built only from training data
    n = len(sentences)
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    train_sents = sentences[:train_end]        # 80%
    val_sents   = sentences[train_end:val_end] # 10%
    test_sents  = sentences[val_end:]          # 10%

    # Build word and tag vocabularies from training data only
    all_words = [word.lower() for sent in train_sents for word, tag in sent]
    all_tags  = [tag for sent in train_sents for word, tag in sent]

    word_vocab = {w: i+1 for i, w in enumerate(set(all_words))}  # 0 reserved for <PAD>
    tag_vocab  = {t: i   for i, t in enumerate(set(all_tags))}
    word_vocab['<PAD>'] = 0

    # ── Choose embedding variant here ──────────────────────────────────────
    EMBEDDING_VARIANT = 'glove'   # options: 'svd' | 'cbow' | 'glove' | 'fasttext'

    EMBEDDING_CONFIGS = {
        'svd':      ('embeddings/svd.pt',       350, None,        None),
        'cbow':     ('embeddings/cbow.pt',      100, None,        None),
        'glove':    ('embeddings/glove_embeddings.pt',     100, 'glove',     100),
        'fasttext': ('embeddings/fasttext_embeddings.pt',  300, 'fasttext',  300),
    }
    EMB_PT, EMB_DIM, PRETRAIN_SRC, PRETRAIN_DIM = EMBEDDING_CONFIGS[EMBEDDING_VARIANT]
    print(f"Using {EMBEDDING_VARIANT.upper()} embeddings | dim={EMB_DIM}")

    # Download and save pre-trained embeddings if not already cached
    if PRETRAIN_SRC is not None:
        save_pretrained_as_pt(EMB_PT, source=PRETRAIN_SRC, dim=PRETRAIN_DIM)

    embedding_matrix = build_embedding_matrix(word_vocab, EMB_PT, embed_dim=EMB_DIM)
    # ───────────────────────────────────────────────────────────────────────

    train_dataset = POSDataset(train_sents, word_vocab, tag_vocab, context_size=CONTEXT_SIZE)
    val_dataset   = POSDataset(val_sents,   word_vocab, tag_vocab, context_size=CONTEXT_SIZE)
    test_dataset  = POSDataset(test_sents,  word_vocab, tag_vocab, context_size=CONTEXT_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # Verify input dimension
    input_dim = (2 * CONTEXT_SIZE + 1) * embedding_matrix.shape[1]
    print(f"Input dim: {input_dim}")   # e.g., C=2, d=100 → 500

    # Instantiate model
    HIDDEN_DIM = 512
    NUM_TAGS   = len(tag_vocab)
    FREEZE_EMB = True   # Set False to fine-tune embeddings during training

    model = POSTagger(
        embedding_matrix=embedding_matrix,
        context_size=CONTEXT_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_tags=NUM_TAGS,
        freeze=FREEZE_EMB
    )
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 10
    print(f"Training on {DEVICE}")

    # ── W&B run ──────────────────────────────────────────────────────────────
    USE_WANDB = True
    wandb.init(
        project="pos-tagger-inlp",
        name=f"{EMBEDDING_VARIANT}_C{CONTEXT_SIZE}_H{HIDDEN_DIM}",
        config={
            "embedding_variant": EMBEDDING_VARIANT,
            "embed_dim":         EMB_DIM,
            "context_size":      CONTEXT_SIZE,
            "hidden_dim":        HIDDEN_DIM,
            "freeze_emb":        FREEZE_EMB,
            "epochs":            EPOCHS,
            "batch_size":        BATCH_SIZE,
            "optimizer":         "Adam",
            "lr":                1e-3,
            "num_tags":          NUM_TAGS,
            "train_size":        len(train_dataset),
            "val_size":          len(val_dataset),
            "test_size":         len(test_dataset),
        }
    )
    # ─────────────────────────────────────────────────────────────────────────

    model = train(model, train_loader, val_loader, criterion, optimizer,
                  epochs=EPOCHS, device=DEVICE, use_wandb=USE_WANDB)

    # Final evaluation on test set
    evaluate_test(model, test_loader, tag_vocab, device=DEVICE,
                  use_wandb=USE_WANDB, word_vocab=word_vocab, n_errors=5)

    wandb.finish()
