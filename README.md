# 2024801008_A2 — Word Embeddings & POS Tagging

Assignment 2 for Introduction to NLP (INLP), IIIT Hyderabad.  
Implements SVD-based and Word2Vec (CBOW) word embeddings, evaluates them via analogy tests and gender-bias analysis, and uses them in a window-based POS tagger.

---

## File Structure

```
Assignment_2/
├── svd_embeddings.py       # Builds co-occurrence matrix from Brown corpus and applies TruncatedSVD
├── word2vec.py             # Trains a CBOW Word2Vec model on Brown corpus
├── analogy_test.py         # Evaluates embeddings on word analogies; computes gender-bias scores
├── pos_tagger.py           # Window-based MLP POS tagger using pretrained embeddings
├── embeddings/
│   ├── svd.pt              # SVD embeddings (vocab + tensors)
│   ├── cbow.pt             # CBOW Word2Vec embeddings (vocab + tensors)
│   └── glove_embeddings.pt # GloVe embeddings (vocab + tensors)
├── confusion_matrix.png    # Confusion matrix from POS tagger evaluation
├── report.md               # Auto-updated results report
└── report.pdf              # Final report
```

---

## How to Run

### 1. Generate SVD Embeddings
```bash
python svd_embeddings.py
```
Builds a sentence-level co-occurrence matrix from the Brown corpus and saves 350-dimensional SVD embeddings to `embeddings/svd.pt`.

### 2. Train CBOW Word2Vec Embeddings
```bash
python word2vec.py
```
Trains a CBOW model on Brown corpus and saves embeddings to `embeddings/cbow.pt`. Also generates `train_data.txt`.

### 3. Run Analogy Tests & Gender Bias Analysis
```bash
python analogy_test.py
```
Evaluates SVD, CBOW, and GloVe embeddings on word analogy tasks and gender-bias cosine similarity. Updates `report.md` with results.

### 4. Train & Evaluate POS Tagger
```bash
python pos_tagger.py
```
Trains a window-based MLP POS tagger using pretrained embeddings on the Brown corpus (universal tagset). Logs metrics to W&B and saves `confusion_matrix.png`.

---

## Dependencies

```bash
pip install torch numpy scikit-learn scipy nltk tqdm wandb
```
