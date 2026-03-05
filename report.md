## Analogy Test
We evaluate three embedding variants using the analogy formula `vec(B) − vec(A) + vec(C)`, selecting the top-5 nearest neighbours (excluding query words) by cosine similarity.
### SVD
**Paris:France :: Delhi:?    (capital city)**
_(one or more query words not in vocabulary)_

**King:Man :: Queen:?        (gender)**
| Rank | Word | Cosine Similarity |
|------|------|------------------|
| 1 | woman | 0.7993 |
| 2 | like | 0.7855 |
| 3 | really | 0.7818 |
| 4 | gone | 0.7818 |
| 5 | little | 0.7812 |

**Swim:Swimming :: Run:?     (verb tense)**
| Rank | Word | Cosine Similarity |
|------|------|------------------|
| 1 | put | 0.7834 |
| 2 | time | 0.7800 |
| 3 | said | 0.7779 |
| 4 | gone | 0.7756 |
| 5 | coming | 0.7734 |

### Word2Vec (CBOW)
**Paris:France :: Delhi:?    (capital city)**
| Rank | Word | Cosine Similarity |
|------|------|------------------|
| 1 | properties | 0.9781 |
| 2 | unity | 0.9771 |
| 3 | sides | 0.9768 |
| 4 | connected | 0.9767 |
| 5 | inner | 0.9764 |

**King:Man :: Queen:?        (gender)**
| Rank | Word | Cosine Similarity |
|------|------|------------------|
| 1 | whether | 0.9978 |
| 2 | every | 0.9978 |
| 3 | considered | 0.9978 |
| 4 | during | 0.9978 |
| 5 | these | 0.9978 |

**Swim:Swimming :: Run:?     (verb tense)**
| Rank | Word | Cosine Similarity |
|------|------|------------------|
| 1 | after | 0.9990 |
| 2 | good | 0.9990 |
| 3 | days | 0.9989 |
| 4 | they | 0.9989 |
| 5 | has | 0.9989 |

### GloVe
**Paris:France :: Delhi:?    (capital city)**
| Rank | Word | Cosine Similarity |
|------|------|------------------|
| 1 | india | 0.8602 |
| 2 | pakistan | 0.7835 |
| 3 | lanka | 0.6693 |
| 4 | bangladesh | 0.6641 |
| 5 | sri | 0.6440 |

**King:Man :: Queen:?        (gender)**
| Rank | Word | Cosine Similarity |
|------|------|------------------|
| 1 | woman | 0.8040 |
| 2 | girl | 0.7349 |
| 3 | she | 0.6818 |
| 4 | her | 0.6592 |
| 5 | mother | 0.6542 |

**Swim:Swimming :: Run:?     (verb tense)**
| Rank | Word | Cosine Similarity |
|------|------|------------------|
| 1 | three | 0.7016 |
| 2 | running | 0.7008 |
| 3 | since | 0.6961 |
| 4 | four | 0.6906 |
| 5 | leading | 0.6884 |

## Gender Bias in Pre-trained Embeddings
Cosine similarity between occupation words and gendered words (`man` / `woman`) computed on GloVe pre-trained embeddings.

| Occupation | cos(word, man) | cos(word, woman) | Closer to |
|------------|----------------|------------------|-----------|
| doctor | 0.6092 | 0.6333 | **woman**(difference is small) |
| nurse | 0.4562 | 0.6139 | **woman** |
| homemaker | 0.2356 | 0.4258 | **woman** |



## Embedding Comparison

Model, Accuracy, Macro-F1
SVD, 0.90, 0.83
CBOW, 0.73, 0.59
GLOV, 0.96, 0.91

## False Predictions

Sample misclassifications from the GloVe POS tagger (first 5 errors on the test set).

| # | Context Window | True Tag | Predicted Tag |
|---|---------------|----------|---------------|
| 1 | eileen got to dancing , | PRT | ADP |
| 2 | little tiny dancing step to | VERB | NOUN |
| 3 | to a hummed tune that | VERB | ADV |
| 4 | could hardly notice , and | VERB | NOUN |
| 5 | to pick up strange men | ADP | PRT |