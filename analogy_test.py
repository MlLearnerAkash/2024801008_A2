#@Author: Akash Manna, 2024801008
#@Date: 05/03/2026

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


# ── Embedding loader ──────────────────────────────────────────────────────────

def load_embeddings(pt_path):
    """Load a .pt file saved as {'embeddings': Tensor(V,D), 'vocab': [str,…]}
    and return a dict  word -> numpy vector."""
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    embeddings_tensor = data['embeddings'].numpy()          # (V, D)
    vocab = data['vocab']                                   # list of V words
    return {word: embeddings_tensor[i] for i, word in enumerate(vocab)}


# ── Core analogy function ─────────────────────────────────────────────────────

def analogy(a, b, c, embeddings, top_k=5):
    """Return the top-k predicted words for the analogy A:B :: C:?
    using  target = vec(B) - vec(A) + vec(C)."""
    if any(w not in embeddings for w in [a, b, c]):
        missing = [w for w in [a, b, c] if w not in embeddings]
        raise KeyError(f"Words not found in vocabulary: {missing}")

    target_vec = embeddings[b] - embeddings[a] + embeddings[c]    # (D,)

    # Build matrix once for fast batch cosine similarity
    words  = [w for w in embeddings if w not in {a, b, c}]
    matrix = np.stack([embeddings[w] for w in words])              # (V', D)

    sims = cosine_similarity(target_vec.reshape(1, -1), matrix)[0] # (V',)
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(words[i], float(sims[i])) for i in top_indices]


# ── Runner ────────────────────────────────────────────────────────────────────

ANALOGIES = [
    ("paris",    "france",   "delhi",    "Paris:France :: Delhi:?    (capital city)"),
    ("king",     "man",      "queen",    "King:Man :: Queen:?        (gender)"),
    ("swim",     "swimming", "run",      "Swim:Swimming :: Run:?     (verb tense)"),
]

EMBEDDING_FILES = {
    "SVD":     "embeddings/svd.pt",
    "Word2Vec (CBOW)": "embeddings/cbow.pt",
    "GloVe":   "embeddings/glove_embeddings.pt",
}


def run_all(top_k=5):
    results = {}   # model -> list of (analogy_label, top_k_list)

    for model_name, pt_path in EMBEDDING_FILES.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")
        try:
            embeddings = load_embeddings(pt_path)
        except Exception as e:
            print(f"  Could not load {pt_path}: {e}")
            continue

        model_results = []
        for a, b, c, label in ANALOGIES:
            print(f"\n  {label}")
            try:
                preds = analogy(a, b, c, embeddings, top_k=top_k)
                for rank, (word, score) in enumerate(preds, 1):
                    print(f"    {rank}. {word:<20s}  (cos={score:.4f})")
                model_results.append((label, preds))
            except KeyError as e:
                print(f"    [SKIPPED] {e}")
                model_results.append((label, []))

        results[model_name] = model_results

    return results


# ── report.md writer ──────────────────────────────────────────────────────────

def update_report(results, report_path="report.md", top_k=5):
    lines = []
    lines.append("## Analogy Test\n")
    lines.append(
        "We evaluate three embedding variants using the analogy formula "
        "`vec(B) − vec(A) + vec(C)`, selecting the top-5 nearest neighbours "
        "(excluding query words) by cosine similarity.\n"
    )

    for model_name, model_results in results.items():
        lines.append(f"### {model_name}\n")
        for label, preds in model_results:
            lines.append(f"**{label}**\n")
            if not preds:
                lines.append("_(one or more query words not in vocabulary)_\n\n")
                continue
            lines.append(f"| Rank | Word | Cosine Similarity |\n")
            lines.append(f"|------|------|------------------|\n")
            for rank, (word, score) in enumerate(preds, 1):
                lines.append(f"| {rank} | {word} | {score:.4f} |\n")
            lines.append("\n")

    # Read existing report content (skip old Analogy Test section)
    try:
        with open(report_path, "r") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""

    # Replace / append the Analogy Test section
    marker = "## Analogy Test"
    if marker in existing:
        before = existing[: existing.index(marker)]
    else:
        before = existing.rstrip() + "\n\n"

    with open(report_path, "w") as f:
        f.write(before + "".join(lines))

    print(f"\nreport.md updated at: {report_path}")


# ── Gender-bias cosine similarity (Pre-trained GloVe only) ───────────────────

BIAS_PAIRS = [
    ("doctor",    "Pair A: doctor    vs. {man/woman}"),
    ("nurse",     "Pair B: nurse     vs. {man/woman}"),
    ("homemaker", "Pair C: homemaker vs. {man/woman}"),
]


def gender_bias_similarity(pt_path="embeddings/glove_embeddings.pt"):
    """For each occupation word compute cos(word, 'man') and cos(word, 'woman')
    using the pre-trained GloVe embeddings and return a list of result dicts."""
    embeddings = load_embeddings(pt_path)

    print(f"\n{'='*60}")
    print("  Gender Bias – Cosine Similarity (GloVe Pre-trained)")
    print(f"{'='*60}")
    print(f"  {'Word':<12s}  {'cos(word,man)':>14s}  {'cos(word,woman)':>16s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*16}")

    records = []
    for word, label in BIAS_PAIRS:
        if word not in embeddings:
            print(f"  {word:<12s}  [OOV]")
            records.append({"label": label, "word": word, "man": None, "woman": None})
            continue

        cos_man = float(cosine_similarity(
            embeddings[word].reshape(1, -1),
            embeddings["man"].reshape(1, -1)
        ))
        cos_woman = float(cosine_similarity(
            embeddings[word].reshape(1, -1),
            embeddings["woman"].reshape(1, -1)
        ))
        print(f"  {word:<12s}  {cos_man:>14.4f}  {cos_woman:>16.4f}")
        records.append({"label": label, "word": word, "man": cos_man, "woman": cos_woman})

    return records


def update_report_bias(records, report_path="report.md"):
    """Append / replace the Gender Bias section in report.md."""
    lines = []
    lines.append("## Gender Bias in Pre-trained Embeddings\n")
    lines.append(
        "Cosine similarity between occupation words and gendered words "
        "(`man` / `woman`) computed on GloVe pre-trained embeddings.\n\n"
    )
    lines.append("| Occupation | cos(word, man) | cos(word, woman) | Closer to |\n")
    lines.append("|------------|----------------|------------------|-----------|\n")
    for r in records:
        if r["man"] is None:
            lines.append(f"| {r['word']} | OOV | OOV | – |\n")
            continue
        closer = "man" if r["man"] > r["woman"] else "woman"
        lines.append(
            f"| {r['word']} | {r['man']:.4f} | {r['woman']:.4f} | **{closer}** |\n"
        )
    lines.append("\n")

    try:
        with open(report_path, "r") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""

    marker = "## Gender Bias in Pre-trained Embeddings"
    if marker in existing:
        before = existing[: existing.index(marker)]
    else:
        before = existing.rstrip() + "\n\n"

    with open(report_path, "w") as f:
        f.write(before + "".join(lines))

    print(f"\nreport.md gender-bias section updated.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all(top_k=5)
    update_report(results)

    bias_records = gender_bias_similarity()
    update_report_bias(bias_records)


