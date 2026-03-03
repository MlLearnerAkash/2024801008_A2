#@Author: Akash Manna
#@Date: 02/03/2026

import sys
import os
import time
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import nltk
nltk.download('stopwords')
from nltk.corpus import brown

#Step-1: Data Preparation
def prepare_brown_corpus(output_path= "train_data.txt"):
    nltk.download('brown')
    sentences = brown.sents() 

    with open(output_path, "w", encoding= "utf8") as f:
        for sent in sentences:
            cleaned = [word.lower() for word in sent if word.isalpha()]
            if len(cleaned) <= 1:
                continue

            f.write(' '.join(cleaned) + '\n')

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, input_file_name, min_count):
        self.negative_words = []
        self.discards = []
        self.negpos = 0
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.input_file_name = input_file_name

        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()
    
    def read_words(self, min_count):
        word_frequency = dict()
        
        for line in open(self.input_file_name, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1
        
        w_id = 1
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = w_id
            self.id2word[w_id] = w
            self.word_frequency[w_id] = c
            w_id += 1

        self.word2id["<SPACE>"] = 0
        self.id2word[0] = "<SPACE>"
        self.word_frequency[0] = 1
    
    def initTableDiscards(self):
        # f = relative frequency of each word
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(0.0001 / f) + (0.0001 / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = np.sum(pow_frequency)
        ratio = pow_frequency / words_pow
        # Fill a huge table (1e8 slots) with word_ids proportional to ratio
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for w_id, c in enumerate(count):
            self.negative_words += [w_id] * int(c)
        self.negative_words = np.array(self.negative_words)
        np.random.shuffle(self.negative_words)

    def getNegatives(self, target, size):
        response = self.negative_words[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negative_words)
        if len(response) != size:
            return np.concatenate((response, self.negative_words[0:self.negpos]))
        return [word_id if word_id != target else response[k-1] 
                for k, word_id in enumerate(response)]

#Step-2: dataset construction
class Word2vecDataLoader(Dataset):
    def __init__(self, data, window_size):
        self.data= data
        self.window_size= window_size
        self.input_file= open(data.input_file_name, encoding= "utf8")

    def __len__(self):
        return self.data.sentences_count
    
    def __getitem__(self, idx):
        while True:
            line= self.input_file.readline()

            if not line:
                self. input_file.seek(0, 0)
                line= self.input_file.readline()
            if len(line)>1:
                words= line.split()

                if len(words)> 1:
                    word_ids= [
                        self.data.word2id[w]
                        for w in words if w in self.data.word2id and
                        np.random.rand()<  self.data.discards[self.data.word2id[w]]
                    ]

                    boundary = self.window_size // 2
                    cbow_data = []

                    for i, v in enumerate(word_ids):
                        context = []
                        for u in word_ids[max(i - boundary, 0) : i + boundary + 1]:
                            if u != v:
                                context.append(u)

                        # pad to fixed length 2*boundary
                        if len(context) < 2 * boundary:
                            context += [0] * (2 * boundary - len(context))

                        negatives = self.data.getNegatives(v, 5)
                        cbow_data.append((context, v, negatives))

                    return cbow_data

    @staticmethod
    def collate(batches):
        all_u    = [u     for batch in batches for u, _, _   in batch if len(batch) > 0]
        all_v    = [v     for batch in batches for _, v, _   in batch if len(batch) > 0]
        all_neg  = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return (
            torch.LongTensor(all_u),    # shape: [N, 2*boundary]
            torch.LongTensor(all_v),    # shape: [N]
            torch.LongTensor(all_neg),  # shape: [N, 5]
        )
    


#Step-3: CBOWModel
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, emb_dimension):
        super(CBOWModel, self).__init__()
        self.emb_dimension = emb_dimension

        # Two embedding tables: input (context) and output (target)
        self.u_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)

        # Initialise: small uniform for input, zeros for output
        initrange = 1.0 / emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        # pos_u: [N, 2*boundary]  → context word IDs
        # pos_v: [N]              → center word IDs
        # neg_v: [N, 5]           → negative word IDs

        # 1. Average context embeddings
        emb_u = self.u_embeddings(pos_u)          # [N, 2k, D]
        emb_u = torch.mean(emb_u, dim=1)          # [N, D]

        # 2. Positive target embedding
        emb_v = self.v_embeddings(pos_v)           # [N, D]

        # 3. Negative embeddings
        emb_neg_v = self.v_embeddings(neg_v)       # [N, 5, D]

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)   # [N]
        score = torch.clamp(score, max=10, min=-10)           # numerical stability
        score = -F.logsigmoid(score)                          # positive loss

        # Negative score: batch matrix multiply → [N, 5, D] × [N, D, 1] → [N, 5]
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()  # [N, 5]
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)          # negative loss

        return torch.mean(score + neg_score)
    
    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for w_id, w in id2word.items():
                e = ' '.join(map(str, embedding[w_id]))
                f.write('%s %s\n' % (w, e))
    

#Ste-4: Training Loop and saving the embedding model


class CBOWTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=128,
                 window_size=9, iterations=5, initial_lr=0.001, min_count=3):

        # 1. Build vocab + tables
        self.data = DataReader(input_file, min_count)

        # 2. Build dataset + dataloader
        dataset = Word2vecDataLoader(self.data, window_size)
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, num_workers=0,       # set >0 on Linux for speed
            collate_fn=dataset.collate
        )

        # 3. Build model
        self.output_file = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.model = CBOWModel(self.emb_size, self.emb_dimension)

        # 4. Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def train(self):
        for iteration in range(self.iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.iterations} ---")

            # Fresh optimizer each epoch
            optimizer = optim.SparseAdam(self.model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(self.dataloader)
            )

            running_loss = 0.0

            for i, batch in enumerate(tqdm(self.dataloader)):
                # Skip trivially small batches
                if len(batch[0]) <= 1:
                    continue

                pos_u = batch[0].to(self.device)   # context IDs
                pos_v = batch[1].to(self.device)   # center IDs
                neg_v = batch[2].to(self.device)   # negative IDs

                optimizer.zero_grad()
                loss = self.model(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Exponential moving average of loss for display
                running_loss = running_loss * 0.9 + loss.item() * 0.1
                if i > 0 and i % 500 == 0:
                    print(f"  Step {i}, Loss: {running_loss:.4f}")

            # Save after every epoch
            self.model.save_embedding(self.data.id2word, self.output_file)
            print(f"Embeddings saved to {self.output_file}")
                


#Step-5: Evaluation




if __name__ == "__main__":
    # prepare_brown_corpus("train_data.txt")
    # data = DataReader("train_data.txt", min_count=3)
    # print("Vocab size:", len(data.word2id))
    # print("Total tokens:", data.token_count)
    # print("Sample negatives:", data.getNegatives(5, size=5))

    # dataset = Word2vecDataLoader(data, window_size=9)
    # sample = dataset[0]           # one sentence → list of triples
    # ctx, center, negs = sample[0] # first triple
    # print("Context IDs:", ctx)    # expect list of 8 ints (2*4)
    # print("Center ID:", center)   # one int
    # print("Negatives:", negs)     # 5 ints

    trainer = CBOWTrainer(
        input_file="train_data.txt",
        output_file="word_embeddings.txt",
        emb_dimension=100,    # start small for testing
        batch_size=128,
        window_size=9,
        iterations=3,         # start with 3 epochs to validate
        initial_lr=0.001,
        min_count=3
    )
    trainer.train()

