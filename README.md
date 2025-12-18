# HR FAQ Chatbot — DSPy-Optimized

An intelligent HR assistant chatbot built with fine-tuned LLMs and DSPy optimization to answer employee questions about HR policies, benefits, payroll, and company procedures with high accuracy and out-of-domain detection.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=for-the-badge)
![DSPy](https://img.shields.io/badge/DSPy-Optimized-00D4AA?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## 🚀 Overview

This project implements a production-ready HR FAQ chatbot that combines:

- **Fine-tuned language models** (LoRA/QLoRA) for domain-specific knowledge
- **DSPy framework** for automatic prompt optimization, achieving +784% ROUGE-L improvement
- **RAG (Retrieval-Augmented Generation)** for context-aware responses from company documents
- **Out-of-domain detection** with 90% rejection rate for non-HR questions
- **Modern web interface** with real-time streaming responses

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
- **DSPy Integration** – Automatic prompt optimization achieving +784% ROUGE-L improvement over baseline
- **OOD Detection** – 90% rejection rate for off-topic questions with keyword-based classification
- **RAG Support** – Retrieval-augmented generation from company HR documents (policies, benefits, payroll)
- **Professional Benchmarks** – Statistical significance testing (p<0.001) with comprehensive error analysis
- **Modern Web UI** – Next.js frontend with real-time streaming, responsive design, and premium UX
- **Production Ready** – Complete pipeline from training to deployment with health checks and error handling

---

## 🧠 How it works (high-level)

- **Frontend**: Built with Next.js and React, handles the chat interface, real-time streaming via Server-Sent Events (SSE), and displays metrics/confidence scores.

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

Create `webapp/.env.local` if you need to customize the backend URL:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

The default is `http://localhost:8000`, so this is only needed if your backend runs on a different port or host.

### 4. Run the app

**Option 1: One-command startup (recommended)**

```bash
# Linux/Mac
./start.sh

# Windows
.\start.ps1
```

This script will:
- Create a Python virtual environment if missing
- Install all dependencies
- Start the backend on port 8000
- Start the frontend on port 3000
- Wait for health checks and print URLs

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
- Check the quality panel for confidence scores and OOD detection status

---

## 📊 Results

### Performance Comparison

| Metric | Baseline | DSPy Optimized | Improvement |
|--------|:--------:|:--------------:|:-----------:|
| **ROUGE-L** | 0.014 | **0.126** | **+784.6%** |
| **OOD Rejection** | 0% | **90%** | **+90.0%** |
| **Latency** | 343ms | **272ms** | **-20.7%** |

**Statistical Significance**: All improvements are statistically significant (p<0.001) with large effect sizes (Cohen's d > 1.0).

### Benchmark Output

Run the professional benchmark:
```bash
python benchmark_professional.py
```

Results are saved to `reports/latest_run/`:
- `metrics.json` – Summary statistics
- `predictions.jsonl` – All individual predictions
- `config.yaml` – Benchmark configuration
- `report.md` – Human-readable report
- `professional_benchmark_results.json` – Full results JSON

---

## 📚 What I learned

- **DSPy Framework** – Learned how to integrate DSPy for automatic prompt optimization, achieving significant improvements in response quality without manual prompt engineering
- **RAG Implementation** – Built a retrieval-augmented generation system using ChromaDB and sentence transformers for context-aware responses from company documents
- **Parameter-Efficient Fine-tuning** – Applied LoRA/QLoRA techniques to fine-tune large language models efficiently, reducing memory requirements while maintaining performance
- **Statistical Validation** – Implemented proper statistical testing (paired t-tests, confidence intervals, effect sizes) to validate model improvements scientifically
- **Full-Stack AI Application** – Integrated fine-tuned models, RAG, and DSPy into a production-ready web application with real-time streaming and error handling
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
- 💼 Looking for: **6-month end-of-studies internship (Data / ML / LLM)** starting **February 2026**  
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
