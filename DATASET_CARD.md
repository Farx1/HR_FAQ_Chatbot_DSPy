# Dataset card: TechCorp Solutions HR corpus

## Summary

Fictional HR documents and question/answer pairs for a company that does not exist, **TechCorp Solutions** (described in `company_data/techcorp_solutions/company_info.json` as a San Francisco software company with 850 employees). Any resemblance to a real company's policies is generic: the documents follow common US HR practice.

Phone numbers (`555-` prefix), e-mail addresses and URLs (`techcorp-solutions.com`) are placeholders and must not be contacted.

## Contents

| Path | Content | Size |
|------|---------|------|
| `company_data/techcorp_solutions/{policies,benefits,payroll,onboarding}/*.md` | 14 policy documents (leave, remote work, dress code, harassment, health, 401(k), wellness, education, salary, bonus, expenses, new-hire checklist, probation, training) | ~79 KB |
| `company_data/techcorp_solutions/company_info.json` | Company metadata used by the API | 1 file |
| `data/train_alpaca.json` | 91 Alpaca-format pairs (`instruction`, `input`, `output`) used for LoRA fine-tuning; 57 answers name TechCorp | 91 rows |
| `data/val_alpaca.json` | Empty list | 0 rows |
| `data/ood_test.json` | 10 off-topic questions with a French refusal as target | 10 rows |

## How it was made

- The Markdown documents were written for this project to give the RAG index realistic content; how they were authored is not recorded in the repository.
- `data/prepare_data.py` downloads and converts `syncora/hr-policies-qa-dataset` from the Hugging Face Hub, but the committed `data/train_alpaca.json` contains TechCorp-specific answers, so it was not produced by that script as it stands. Its exact generation process is not recorded in the repository.
- The RAG index (`backend/chroma_db/`) is rebuilt from the Markdown files: one chunk per `## ` section, embedded with `sentence-transformers/all-MiniLM-L6-v2`.

## Intended use

Testing retrieval, OOD filtering and fine-tuning code on a small, safe corpus with no personal data.

## Known issues

- No held-out split: `val_alpaca.json` is empty, and the historical benchmark sampled its test questions from `train_alpaca.json`.
- Training answers and documents can disagree on details; the documents are the reference for the API.
- Single author, single style; questions are short and well-formed, unlike real employee messages.

## License

MIT, like the rest of the repository. The documents are fictional and contain no personal data.
