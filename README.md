# HR FAQ Chatbot — DSPy-Optimized

An intelligent HR assistant chatbot built with fine-tuned LLMs and DSPy optimization to answer employee questions about HR policies, benefits, payroll, and company procedures with high accuracy and out-of-domain detection.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=for-the-badge)
![DSPy](https://img.shields.io/badge/DSPy-Optimized-00D4AA?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![CI](https://github.com/Farx1/HR_FAQ_Chatbot_DSPy/workflows/CI/badge.svg?style=for-the-badge)

</div>

---

## 🚀 Overview

This project implements a production-ready HR FAQ chatbot that combines:

- **Fine-tuned language models** (LoRA/QLoRA) for domain-specific knowledge
- **DSPy wrapper** around the fine-tuned model (adapter + module with keyword OOD filter; see [Limitations](#limitations))
- **RAG (Retrieval-Augmented Generation)** for context-aware responses from company documents
- **Out-of-domain detection** with 90% rejection rate for non-HR questions
- **Modern web interface** with SSE word-by-word display of the answer

Built for **HR departments** and **companies** looking to automate employee support, with a focus on **reliability**, **accuracy**, and **statistical validation** of improvements.

---

## 🧰 Tech Stack

- **Languages**: Python 3.8+, TypeScript
- **Frameworks**: FastAPI (backend), Next.js 16 (frontend), React 19
- **AI / ML**: PyTorch, Transformers, PEFT (LoRA), DSPy, Sentence Transformers
- **Data / Storage**: ChromaDB (vector database), JSON (datasets), SQLite
- **Tools**: Uvicorn, Mantine UI, Framer Motion, Rouge/BLEU evaluation

---

## ✨ Features

- **Fine-tuned LLM** – LoRA/QLoRA parameter-efficient fine-tuning on Mistral-7B or DialoGPT for domain adaptation
- **DSPy Integration** – DSPy adapter/module around the fine-tuned model; a BootstrapFewShot script exists (`dspy_module/optimize_dspy.py`) but is not used by the benchmark
- **OOD Detection** – 90% rejection rate for off-topic questions with keyword-based classification
- **RAG Support** – Retrieval-augmented generation from company HR documents (policies, benefits, payroll)
- **Benchmarks** – Paired t-tests, confidence intervals and error analysis (small test set, see [Limitations](#limitations))
- **Modern Web UI** – Next.js frontend with SSE word-by-word display, responsive design, and premium UX
- **Production Ready** – Complete pipeline from training to deployment with health checks and error handling

---

## 🧠 How it works (high-level)

- **Frontend**: Built with Next.js and React, handles the chat interface, displays the answer word by word via Server-Sent Events (SSE), and shows metrics/confidence scores.

- **Backend / API**: FastAPI server that:
  - Processes questions through OOD detection (early rejection of non-HR queries)
  - Retrieves relevant context from company documents using ChromaDB vector search
  - Generates answers using fine-tuned models with DSPy-optimized prompts
  - Streams responses back to frontend for real-time UX

- **Data**: 
  - Company HR documents stored in `company_data/` (policies, benefits, payroll)
  - Vector embeddings in ChromaDB for semantic search
  - Training data in Alpaca format for fine-tuning

- **AI / ML part**:
  - Uses fine-tuned DialoGPT-small or Mistral-7B with LoRA adapters for parameter efficiency
  - DSPy framework optimizes prompts automatically using ChainOfThought reasoning
  - RAG engine retrieves relevant document snippets based on semantic similarity
  - Keyword-based OOD detection filters out non-HR questions before processing

---

## 📦 Project Structure

- `backend/` – FastAPI server with RAG engine (`server.py`, `rag_engine.py`)
- `webapp/` – Next.js frontend application (`src/app/page.tsx` for main UI)
- `dspy_module/` – DSPy integration (`hr_faq_dspy.py` for adapter and modules)
- `training/` – Fine-tuning scripts (`train.py` for GPU, `train_cpu.py` for CPU)
- `evaluation/` – Evaluation and benchmark scripts
- `demo/` – Interactive command-line demos
- `data/` – Training datasets (Alpaca format) and test sets
- `models/` – Fine-tuned LoRA adapters and checkpoints
- `company_data/` – HR documents for RAG (policies, benefits, payroll)
- `reports/` – Benchmark results and analysis (`reports/latest_run/` for structured outputs)
- `MODEL_CARD.md`, `DATASET_CARD.md` – what the LoRA adapter and the TechCorp data are (and are not)
- `docs/EVALUATION.md` – metric definitions (Recall@k, MRR, nDCG), confidence intervals, paired bootstrap, LoRA math

---

## 🧪 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Farx1/HR_FAQ_Chatbot_DSPy.git
cd HR_FAQ_Chatbot_DSPy
```

### 2. Install dependencies

**Python dependencies:**
```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

**Frontend dependencies:**
```bash
cd webapp
npm install
cd ..
```

### 3. Environment variables (optional)

**Backend:** Copy `backend/.env.example` to `backend/.env` if you need to customize configuration (optional, defaults work for most cases).

**Frontend:** Create `webapp/.env.local` if you need to customize the backend URL:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

The default is `http://localhost:8000`, so this is only needed if your backend runs on a different port or host.

### 3.5. Verify fresh clone (optional but recommended)

To verify that everything is set up correctly, run the verification script:

**Linux/Mac:**
```bash
chmod +x scripts/verify_fresh_clone.sh
./scripts/verify_fresh_clone.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\verify_fresh_clone.ps1
```

This script will:
- Create a Python virtual environment
- Install all dependencies
- Run linting (ruff)
- Run all tests (pytest)
- Test backend imports
- Test health endpoint (FastAPI TestClient)
- Build the frontend
- Verify everything works offline (no external API calls)

If all checks pass, your environment is ready for development!

### 4. Run the app

**Option 1: One-command startup (recommended)**

```bash
# If you have Make installed (Linux/Mac)
make dev

# Or use the scripts directly:
# Linux/Mac
./scripts/dev.sh
# or
./start.sh

# Windows
.\scripts\dev.ps1
# or
.\start.ps1
```

This will:
- Create a Python virtual environment if missing
- Install all dependencies (Python + Node.js)
- Start the backend on port 8000
- Start the frontend on port 3000
- Wait for backend health check
- Print URLs and success message

**Option 2: Manual startup**

Terminal 1 (Backend):
```bash
cd backend
uvicorn server:app --reload --port 8000
```

Terminal 2 (Frontend):
```bash
cd webapp
npm run dev
```

Then open: **http://localhost:3000**

### 5. Test the chatbot

- Ask HR questions like: "How many vacation days do I get per year?"
- Try OOD questions like: "What is the capital of France?" (should be rejected)
- Check the quality panel for confidence scores (fixed heuristic values per code path, not model probabilities) and OOD detection status

---

## 📊 Results

### Performance Comparison

Numbers from `reports/professional_benchmark_results.json` (run of 2025-12-15, DialoGPT-small, 20 HR + 40 OOD questions). Read them with the [Limitations](#limitations) below.

| Metric | Baseline | DSPy module | Difference | Paired t-test |
|--------|:--------:|:-----------:|:----------:|:-------------:|
| **ROUGE-L** | 0.014 | 0.126 | +0.111 | p = 1.4e-6, d = 1.54 |
| **Exact Match** | 0.0 | 0.0 | 0 | not significant |
| **BLEU** | 0.0 | 0.0 | 0 | not significant |
| **OOD Rejection** | 0% | 90% | +90 pts | p = 4.2e-21, d = 2.96 |
| **Latency (mean)** | 356 ms | 258 ms | -27.5% | not tested |

Only ROUGE-L and OOD rejection differ significantly. Exact Match and BLEU are 0 for both systems.

### Held-out evaluation

`python evaluation/honest_eval.py --generation` (CPU, about 12 min). Full output is in `reports/honest_eval.json`. The test set `data/heldout_test.json` has two parts. The 42 HR questions were drafted with an AI assistant from the documents and are not yet reviewed by a person; none of them is in `train_alpaca.json`. The 40 OOD questions come from `benchmark_professional.py`. Intervals are 95 %: Wilson for proportions, percentile bootstrap (10,000 resamples, seed 0) for means.

| Measure | n | Value | 95 % CI |
|---------|:-:|:-----:|:-------:|
| RAG Recall@1, gold section (Chroma, all-MiniLM-L6-v2) | 42 | 0.74 | 0.59-0.85 |
| RAG Recall@3, gold section | 42 | 0.93 | 0.81-0.98 |
| RAG Recall@1, gold document | 42 | 0.88 | 0.75-0.95 |
| RAG Recall@3, gold document | 42 | 0.98 | 0.88-1.00 |
| RAG MRR@5, gold section | 42 | 0.83 | 0.73-0.91 |
| API keyword filter: OOD rejected | 40 | 0.90 | 0.77-0.96 |
| API keyword filter: HR wrongly rejected | 42 | 0.40 | 0.27-0.56 |
| DSPy keyword filter: OOD rejected | 40 | 0.45 | 0.31-0.60 |
| DSPy keyword filter: HR wrongly rejected | 42 | 0.21 | 0.12-0.36 |
| ROUGE-L, baseline (DialoGPT-small, greedy, no LoRA) | 42 | 0.014 | 0.005-0.025 |
| ROUGE-L, DSPy module, template and refusals scored 0 | 42 | 0.000 | 0.000-0.000 |
| ROUGE-L, DSPy module, template scored as an answer | 42 | 0.105 | 0.087-0.123 |
| DSPy module answers that are the template | 42 | 0.79 | 0.64-0.88 |

- No DSPy-module answer was a model answer: 33 were the fallback template and 9 were refusals. Its paired difference with the baseline is -0.014 (95 % CI -0.024 to -0.005).
- The two filters do not differ significantly on the 82 questions (exact McNemar test, p = 0.17).
- With PR #3 (whole-word matching), the DSPy filter's HR false rejections fall to 1/42 (0.02, CI 0.004-0.12). OOD rejection does not change.
- The 90 % OOD rejection in the table above does not measure the filter. `benchmark_professional.py` counts any answer that contains "hr" or "contact" as a rejection, and the fallback template contains both.
- Generation runs without the LoRA adapter, whose weights are not in the repo. The DSPy module does not query the RAG index.

### Benchmark Output

Run the professional benchmark:
```bash
# Activate virtual environment first (if using venv)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run benchmark
python benchmark_professional.py
```

Results are saved to `reports/latest_run/`:
- `metrics.json` – Summary statistics
- `predictions.jsonl` – All individual predictions
- `config.yaml` – Benchmark configuration
- `report.md` – Human-readable report
- `professional_benchmark_results.json` – Full results JSON

### Reproducing Results

**Baseline vs DSPy Optimized:**
- **Baseline**: Fine-tuned model without DSPy optimization (direct inference)
- **DSPy module**: `HRFAQModule.forward` = keyword OOD filter, then direct `generate` with a fixed prompt, then a template fallback for short answers. No optimized DSPy program is loaded.

**To rerun the benchmark:**
1. Train the LoRA adapter first: the weights (`*.safetensors`) are not committed, only configs and tokenizer files
2. Run `python benchmark_professional.py`
3. Compare baseline vs DSPy results in `reports/latest_run/metrics.json`
4. The improvement is calculated as: `(DSPy_ROUGE_L - Baseline_ROUGE_L) / Baseline_ROUGE_L * 100`

**Note**: The benchmark sets seed 42, but generation uses `do_sample=True` and the weights are not in the repo, so exact numbers are not reproducible from a fresh clone.

### Limitations

- **Template fallback counted as model output**: 16/20 DSPy answers are the hard-coded template from `dspy_module/hr_faq_dspy.py` ("According to company HR policies, <question> is handled through established procedures..."). It echoes the question, which lifts ROUGE-L. The ROUGE-L gain mostly measures this template, not DSPy.
- **No DSPy optimization in the benchmark**: `optimize_dspy.py` (BootstrapFewShot) is never called by `benchmark_professional.py`; no compiled program is loaded.
- **Train/test leakage**: the 20 HR test questions are sampled from `data/train_alpaca.json`, the fine-tuning data. `data/val_alpaca.json` is empty.
- **Small sample**: 20 HR and 40 OOD questions.
- **OOD rejection is a keyword filter**, applied only on the DSPy side. 2 of the 20 HR questions were wrongly rejected (dress code, training) by substring matches.
- **LoRA weights not committed**: `models/` holds configs and tokenizers only.
- **Streaming is simulated**: `/ask/stream` computes the full answer (RAG formatting or keyword fallback, no model call), then replays it word by word with `asyncio.sleep`.
- **Confidence scores are constants** chosen per code path in `backend/server.py`.

---

## 📚 What I learned

- **DSPy Framework** – Learned how to wrap a local Hugging Face model as a DSPy LM and module, and how easily a benchmark can overstate a gain (see Limitations)
- **RAG Implementation** – Built a retrieval-augmented generation system using ChromaDB and sentence transformers for context-aware responses from company documents
- **Parameter-Efficient Fine-tuning** – Applied LoRA/QLoRA techniques to fine-tune large language models efficiently, reducing memory requirements while maintaining performance
- **Statistical Validation** – Implemented proper statistical testing (paired t-tests, confidence intervals, effect sizes) to validate model improvements scientifically
- **Full-Stack AI Application** – Integrated fine-tuned models, RAG, and DSPy into a web application with SSE display and error handling
- **Evaluation Metrics** – Gained deep understanding of ROUGE-L, BLEU, and OOD detection metrics for chatbot evaluation

---

## 🔮 Possible improvements

- **Multi-language support** – Extend to support questions in multiple languages (French, Spanish, etc.)
- **User authentication** – Add login system to personalize responses based on employee role/department
- **Feedback loop** – Implement user feedback collection to continuously improve the model
- **Advanced RAG** – Experiment with re-ranking, multi-hop retrieval, and hybrid search (keyword + semantic)
- **Additional DSPy optimizers** – Try MIPRO, GEPA, or other DSPy optimizers for further improvements
- **Deployment** – Containerize with Docker and deploy to cloud (AWS, GCP, Azure) with auto-scaling
- **Monitoring & Logging** – Add comprehensive logging, metrics collection (Prometheus), and alerting
- **Testing** – Expand unit tests, integration tests, and end-to-end tests for reliability
- **Documentation** – Add API documentation (OpenAPI/Swagger) and user guides

---

## 👤 About me (Jules)

I'm **Jules**, a M2 **Data & AI Engineering** student at **ESILV (Paris, France)**,  
with a focus on **LLMs, agentic AI, privacy-preserving ML, and quantum computing**.

- 🎓 Currently: M2 Data & IA, Quantum track  
- 🌐 Portfolio: https://julesbarth-myportfolio.fr  
- 💼 LinkedIn: https://www.linkedin.com/in/jules-barth  
- 📧 Email: julesbarth13@gmail.com  

Feel free to reach out if this project resonates with what you're building. 🚀

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **[Mistral AI](https://mistral.ai/)** — Base model
- **[DSPy](https://github.com/stanfordnlp/dspy)** — Optimization framework
- **[Hugging Face](https://huggingface.co/)** — Transformers & PEFT
- **[Vercel](https://vercel.com/)** — Next.js framework
