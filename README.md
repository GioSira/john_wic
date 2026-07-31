# John-Wic
**John-WiC** is a high-entropy benchmark dataset for the **Word in Context (WiC)** task. It is designed to probe lexical semantic models in challenging disambiguation settings, by focusing on target words that appear in highly ambiguous or semantically diverse contexts — i.e., instances where the entropy over possible word senses is high.

---

## Overview

The **Word in Context (WiC)** task ([Pilehvar & Camacho-Collados, 2019](https://aclanthology.org/N19-1128/)) frames lexical ambiguity as a binary classification problem: given a target word appearing in two distinct sentences, the system must decide whether the word carries the **same sense** in both contexts.

John-WiC extends this framework by concentrating on **high-entropy** instances — pairs where the target word is particularly polysemous or context-dependent — to provide a more demanding evaluation benchmark than existing WiC datasets.

---

## Motivation: Why High Entropy?

Standard WiC datasets may include many instances where context already strongly disambiguates the target word, making the task tractable via shallow lexical cues. John-WiC focuses specifically on **high-entropy** instances — words whose sense distribution is broad and uncertain — to:

- stress-test the ability of language models to perform fine-grained sense disambiguation;
- expose limitations of models that rely on surface-level contextual patterns;
- provide a complementary, harder evaluation split alongside existing WiC benchmarks.

---

## Dataset Format

The dataset is distributed as a single **JSONL** file (`john_wic.jsonl`), where each line is a self-contained JSON object representing one instance. Each instance contains:

| Field | Type | Description |
|---|---|---|
| `lemma` | `string` | The target word (lemma form) whose sense is to be compared |
| `sentence1` | `string` | First sentence in which the target lemma appears |
| `sentence2` | `string` | Second sentence in which the target lemma appears |
| `sentence1_sense` | `string` | WordNet sense identifier for the target word in `sentence1` (format: `wn:XXXXXXXXXPOS`) |
| `sentence2_sense` | `string` | WordNet sense identifier for the target word in `sentence2` (format: `wn:XXXXXXXXXPOS`) |
| `sentence1_gloss` | `string` | Human-readable WordNet gloss (definition) of the sense in `sentence1` |
| `sentence2_gloss` | `string` | Human-readable WordNet gloss (definition) of the sense in `sentence2` |
| `label` | `string` | `"1"` if the lemma carries the **same sense** in both sentences, `"0"` otherwise |

The WordNet sense IDs follow the format `wn:<offset><pos>`, where `<pos>` is one of `n` (noun), `v` (verb), `a` (adjective), or `r` (adverb).

**Example — same sense (label `"1"`):**

```json
{
  "lemma": "show",
  "sentence1_sense": "wn:02148788v",
  "sentence2_sense": "wn:02148788v",
  "sentence1_gloss": "give an exhibition of to an interested audience",
  "sentence2_gloss": "give an exhibition of to an interested audience",
  "sentence1": "The artist will show her latest paintings at the gallery tonight.",
  "sentence2": "The gallery will show the new paintings to the visitors tonight.",
  "label": "1"
}
```

**Example — different sense (label `"0"`):**

```json
{
  "lemma": "show",
  "sentence1_sense": "wn:01086549v",
  "sentence2_sense": "wn:06879521n",
  "sentence1_gloss": "finish third or better in a horse or dog race",
  "sentence2_gloss": "something intended to communicate a particular impression",
  "sentence1": "The underdog managed to show in the final race of the day.",
  "sentence2": "He maintained a show of confidence despite his inner turmoil.",
  "label": "0"
}
```

### Statistics

| Metric | Value |
|---|---|
| Total instances | 20,154 |
| Unique lemmas | 2,588 |
| Unique WordNet senses | 9,782 |
| Same-sense pairs (label = 1) | 10,098 (50.1%) |
| Different-sense pairs (label = 0) | 10,056 (49.9%) |
| POS coverage | Nouns (48.3%), Verbs (34.8%), Adjectives (14.6%), Adverbs (2.3%) |
| Avg. sentence length | ~11 words (sentence1), ~14 words (sentence2) |

---

## Licenses

**Dataset:**  This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material for any purpose, including commercially, provided appropriate credit is given and a link to the license is included.

**Code:** All scripts and code in this repository are released under the [MIT License](LICENSE).

---

## Citation

> *Citation information will be provided upon publication. This repository is currently anonymized for double-blind peer review.*

---

## Contact

> *Contact information will be provided upon publication.*

