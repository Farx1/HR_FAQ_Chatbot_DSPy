# HR FAQ Chatbot - Fine-tuned with DSPy Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional HR FAQ chatbot fine-tuned on Mistral AI models with DSPy optimization for automatic prompt engineering. This project demonstrates state-of-the-art parameter-efficient fine-tuning (LoRA/QLoRA) combined with DSPy framework for improved performance.

## 🎯 Features

- **Fine-tuned Language Model**: LoRA/QLoRA fine-tuning on Mistral-7B-Instruct or DialoGPT for parameter efficiency
- **DSPy Integration**: Automatic prompt optimization using DSPy framework
- **Professional Benchmarking**: Comprehensive evaluation with statistical significance testing
- **Out-of-Domain Detection**: Intelligent rejection of non-HR questions
- **Production Ready**: Complete pipeline from data preparation to deployment

## 📊 Results

### Professional Benchmark Results

| Metric | Baseline | DSPy Optimized | Improvement |
|--------|----------|---------------|-------------|
| **ROUGE-L** | 0.014 ± 0.029 | **0.126 ± 0.074** | **+0.111** (p<0.001) |
| **OOD Rejection** | 0.0% | **90.0%** | **+90.0%** (p<0.001) |
| **Latency** | 0.343s | 0.272s | -20.7% |

**Statistical Significance**: All improvements are statistically significant (p<0.001) with large effect sizes (Cohen's d > 1.0).

See `reports/professional_benchmark_report.md` for detailed analysis.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd finetunemodel

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from dspy_module.hr_faq_dspy import HRFAQAdapter, HRFAQModule
import dspy

# Load model
adapter = HRFAQAdapter()
dspy.configure(lm=adapter)
module = HRFAQModule(adapter=adapter)

# Ask a question
result = module(question="How many vacation days do I get per year?")
print(result.answer)
```

### Interactive Demo

```bash
python demo/interactive_demo_cpu.py
```

## 📁 Project Structure

```
├── data/                    # Datasets and preprocessing
│   ├── train_alpaca.json   # Training data (Alpaca format)
│   ├── val_alpaca.json     # Validation data
│   └── ood_test.json       # Out-of-domain test set
├── models/                  # Fine-tuned models and LoRA adapters
├── training/                # Training scripts
│   ├── train.py            # Main training script
│   └── train_cpu.py         # CPU-optimized training
├── evaluation/              # Evaluation scripts
│   └── eval_cpu.py          # Evaluation metrics
├── dspy_module/             # DSPy integration
│   ├── hr_faq_dspy.py       # DSPy adapter and modules
│   ├── benchmark_dspy.py     # Benchmark comparison
│   └── optimize_dspy.py     # Prompt optimization
├── demo/                     # Interactive demos
├── reports/                  # Evaluation reports and results
│   └── professional_benchmark_report.md
└── benchmark_professional.py  # Professional benchmark script
```

## 🔬 Methodology

### Fine-tuning

- **Model**: Mistral-7B-Instruct-v0.3 (or DialoGPT-small for CPU)
- **Method**: LoRA/QLoRA (Parameter-Efficient Fine-Tuning)
- **Dataset**: HR policies Q&A dataset in Alpaca format
- **Hyperparameters**: See `config.py`

### DSPy Optimization

- **Framework**: DSPy for automatic prompt engineering
- **Modules**: ChainOfThought reasoning, custom adapters
- **Optimization**: Keyword-based OOD detection, improved prompts

### Evaluation

- **Metrics**: ROUGE-L, BLEU, Exact Match, OOD Rejection Rate
- **Statistical Analysis**: Paired t-tests, confidence intervals, effect sizes
- **Test Set**: 20 HR questions, 40 OOD questions across 8 categories

## 📈 Benchmarking

### Run Professional Benchmark

```bash
python benchmark_professional.py
```

This will:
- Create a comprehensive test set
- Evaluate baseline and DSPy models
- Perform statistical significance testing
- Generate detailed reports

### Results Location

- **JSON Results**: `reports/professional_benchmark_results.json`
- **Markdown Report**: `reports/professional_benchmark_report.md`

## 🛠️ Development

### Training

```bash
# Prepare data
python data/prepare_data.py

# Train model
python training/train.py
```

### Evaluation

```bash
# Standard evaluation
python evaluation/eval_cpu.py

# DSPy benchmark
python dspy_module/benchmark_dspy.py

# Professional benchmark
python benchmark_professional.py
```

## 📚 Documentation

- **DSPy Integration**: See `DSPY_INTEGRATION.md`
- **Professional Benchmark**: See `reports/professional_benchmark_report.md`
- **DSPy Module**: See `dspy_module/README.md`

## 🔧 Configuration

Key configuration parameters are in `config.py`:

```python
MODEL_NAME = "microsoft/DialoGPT-small"  # or "mistralai/Mistral-7B-Instruct-v0.3"
LORA_R = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
```

## 📊 Performance

### Key Improvements with DSPy

1. **ROUGE-L**: +784.6% improvement (0.014 → 0.126)
2. **OOD Detection**: +90% rejection rate (0% → 90%)
3. **Error Reduction**: 
   - Too short responses: 15 → 1
   - Off-topic responses: 20 → 7
   - Incomplete responses: 7 → 0

### Statistical Validation

- All improvements are statistically significant (p<0.001)
- Large effect sizes (Cohen's d > 1.0)
- 95% confidence intervals reported

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the base model
- **DSPy** team for the optimization framework
- **Hugging Face** for transformers and PEFT libraries

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project uses DialoGPT-small for CPU demonstration. For production, use Mistral-7B-Instruct with GPU for significantly better performance.
