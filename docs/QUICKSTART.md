# Quick Start Guide

Get started with the HR FAQ Chatbot in minutes!

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd finetunemodel

# Install dependencies
pip install -r requirements.txt
```

## Quick Demo

### 1. Interactive Chatbot

```bash
python demo/interactive_demo_cpu.py
```

Then ask questions like:
- "How many vacation days do I get per year?"
- "What is the sick leave policy?"
- "How do I request a salary review?"

### 2. Run Benchmark

```bash
python benchmark_professional.py
```

This will evaluate the model and generate a comprehensive report.

## Using the Model Programmatically

### Basic Usage

```python
from dspy_module.hr_faq_dspy import HRFAQAdapter, HRFAQModule
import dspy

# Initialize
adapter = HRFAQAdapter()
dspy.configure(lm=adapter)
module = HRFAQModule(adapter=adapter)

# Ask a question
result = module(question="How many vacation days do I get per year?")
print(result.answer)
```

### With Error Handling

```python
from dspy_module.hr_faq_dspy import HRFAQAdapter, HRFAQModule
import dspy

try:
    adapter = HRFAQAdapter()
    dspy.configure(lm=adapter)
    module = HRFAQModule(adapter=adapter)
    
    result = module(question="Your question here")
    
    if result and hasattr(result, 'answer'):
        print(f"Answer: {result.answer}")
    else:
        print("No answer generated")
        
except Exception as e:
    print(f"Error: {e}")
```

## Training Your Own Model

### 1. Prepare Data

```bash
python data/prepare_data.py
```

### 2. Train

```bash
# For CPU (DialoGPT-small)
python training/train_cpu.py

# For GPU (Mistral-7B)
python training/train.py
```

### 3. Evaluate

```bash
python evaluation/eval_cpu.py
```

## Next Steps

- Read the [README.md](../README.md) for full documentation
- Check [BENCHMARK.md](BENCHMARK.md) for evaluation details
- See [DSPY_INTEGRATION.md](../DSPY_INTEGRATION.md) for DSPy optimization

## Troubleshooting

### Model Not Found

If you see "Model not found" errors:
1. Make sure you've trained the model first
2. Check that `models/hr_faq_dialogpt_lora` exists
3. Run training: `python training/train_cpu.py`

### Import Errors

If you see import errors:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version: `python --version` (should be 3.8+)
3. Verify you're in the project root directory

### Memory Issues

If you run out of memory:
1. Use the CPU version: `train_cpu.py` and `eval_cpu.py`
2. Reduce batch size in `config.py`
3. Use a smaller model (DialoGPT-small)

## Getting Help

- Check the [README.md](../README.md)
- Review [BENCHMARK.md](BENCHMARK.md)
- Open an issue on GitHub

