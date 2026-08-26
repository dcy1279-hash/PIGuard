# FIT5230 Milestone 1 — Targeted token obfuscation against PIGuard

This folder extends the official PIGuard/InjecGuard evaluation without changing the model checkpoint, labels, prediction rule, or evaluation-set size. The Dark-team objective is to test whether a prompt that remains an instruction can evade the input-stage prompt-injection detector after a small surface-form change.

## Reproduce in Colab

Open the public notebook:

https://colab.research.google.com/github/dcy1279-hash/PIGuard/blob/main/FIT5230_M1_InjecGuard_Public.ipynb

Run the cells from top to bottom. The notebook clones this public repository, creates an isolated Python 3.10 environment, downloads the official checkpoint from the link published by the PIGuard authors, validates all 75 BIPIA text transformations, and runs the full WildGuard, BIPIA text, BIPIA code, and NotInject evaluation. It does not require access to a private Google Drive.

## M1 transformation

For each of the 75 public BIPIA text prompts, the evaluator deterministically selects exactly one command-bearing token. Primary verbs such as `write`, `provide`, `show`, and `summarize` are preferred; a small fixed fallback list gives complete coverage.

Two variants are compared with the unchanged prompt:

- **ASCII:** leetspeak-like substitutions, for example `write` to `wr173`.
- **Unicode:** visually similar Cyrillic characters, for example `write` to `wrіtе`.

The surrounding prompt, word order, and requested action are not changed. The Unicode variant is intended to remain human-readable, but downstream task-following or human annotation would provide stronger evidence of attack-label preservation.

## Completed M1 result

The completed CPU run produced:

| Variant | Detection rate | Guard miss rate | Successful flips among 22 baseline-detected prompts | Adverse flips |
|---|---:|---:|---:|---:|
| Original | 29.33% | 70.67% | — | — |
| ASCII | 30.67% | 69.33% | 4/22 (18.18%) | 5/75 |
| Unicode | 20.00% | 80.00% | 8/22 (36.36%) | 1/75 |

ASCII did not improve aggregate evasion because its five adverse flips outweighed four successful flips. Unicode reduced the detector's BIPIA-text accuracy from 29.33% to 20.00%, increased the guard miss rate to 80.00%, and reduced mean injection probability by approximately 0.0803.

The official baseline summary remains separate from the attack variants:

- BIPIA code accuracy: 96.00%
- BIPIA original text/code baseline: 62.67%
- NotInject over-defense accuracy: 91.74%
- WildGuard benign accuracy: 80.54%
- Official overall accuracy: 78.31%

## Output and interpretation

`eval_m1.py` reports aggregate metrics and writes `/content/m1_targeted_results.csv`. The CSV contains anonymous sample identifiers, category information, predictions, probabilities, and paired flips; it excludes prompt text.

The most important attack metric is the paired successful flip: an original prompt classified as injection that becomes benign after modification. Prompts already missed by the baseline are not counted as new attack successes.

## Route to later milestones

- **M2:** validate functionality/semantics, analyse results by category, and test stronger controlled Unicode or structural transformations.
- **M3:** evaluate transfer to another prompt guard or a peer defence using the same paired samples.
- **M4:** reverse roles and test normalization or confusable-character canonicalization as a defence against the observed weakness.

## Attribution

Baseline paper and repository: PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free, ACL 2025, by the original PIGuard authors. This fork retains the upstream repository history and licence.
