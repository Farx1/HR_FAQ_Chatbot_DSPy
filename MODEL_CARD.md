# Model card: HR FAQ LoRA adapter for DialoGPT-small

## Summary

A LoRA adapter for [`microsoft/DialoGPT-small`](https://huggingface.co/microsoft/DialoGPT-small) (GPT-2 architecture, 124M parameters), fine-tuned to answer HR questions about the fictional company TechCorp Solutions. Built as a learning / portfolio project.

**The adapter weights are not published.** `models/hr_faq_dialogpt_lora_adapters/` holds only `adapter_config.json` and an empty Hugging Face README template; `*.safetensors` and `*.bin` are git-ignored. To use the adapter, retrain it (see below).

## Details

| Field | Value | Source |
|-------|-------|--------|
| Base model | `microsoft/DialoGPT-small` | `adapter_config.json` |
| Method | LoRA (Hu et al., 2021), see [docs/EVALUATION.md](docs/EVALUATION.md#lora) | |
| Rank $r$ / $\alpha$ / dropout | 8 / 16 / 0.1 | `adapter_config.json` |
| Target modules | `c_attn`, `c_proj` (attention and projection layers of every block) | `adapter_config.json` |
| Training data | `data/train_alpaca.json`, 91 question/answer pairs | `training/train_cpu.py` |
| Epochs / batch size / learning rate | 1 / 1 / 5e-4 | `training/train_cpu.py` |
| Max sequence length | 128 tokens | `training/train_cpu.py` |
| Seed | 42 | `training/train_cpu.py` |
| Hardware | CPU | `training/train_cpu.py` |

Trainable parameters per block: `c_attn` (768 x 2304) gets $8 \times (768 + 2304) = 24{,}576$, `c_proj` (768 x 768) gets $8 \times 1536 = 12{,}288$, about 1.4 % and 2.1 % of the frozen weights they adapt.

Other configs in `models/` (`hr_faq_dialogpt_large_lora`, Mistral-7B scripts in `training/`) are experiments without published weights or results.

## Intended use

- Demo of a LoRA + RAG + DSPy pipeline on a small CPU model.
- Not for answering real employees: the knowledge comes from fictional documents, and the model has no notion of which company or country it is talking about.

## How to reproduce

```bash
pip install -r requirements.txt
python training/train_cpu.py   # writes models/hr_faq_dialogpt_lora/ (weights stay local)
```

One epoch over 91 short examples is a very small amount of training; expect answers that copy the style of the training set more than its facts.

## Evaluation

No evaluation of this adapter can be reproduced from a fresh clone, because the weights are not committed. The numbers in `reports/professional_benchmark_results.json` came from a local copy of the adapter, with sampling (`do_sample=True`), a test set drawn from the training data, and a template fallback counted as model output; see the README Limitations.

## Limitations and risks

- **Train/test overlap**: the historical benchmark tests on questions from the training file.
- **Hallucination**: a 124M-parameter dialogue model fine-tuned for one epoch produces fluent but often wrong policy details (amounts, deadlines). In the API, answers come from retrieved documents (`generate_rag_response`) or fixed fallback text, not from this model, whenever RAG is available.
- **English only**, US-style HR vocabulary (401(k), FMLA, COBRA).
- **DialoGPT** was trained on Reddit conversations; its biases and toxic-output risks carry over (see the DialoGPT model card).

## License

Code: MIT (this repository). The base model `microsoft/DialoGPT-small` is under the MIT license on its Hugging Face page.
