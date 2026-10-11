# Professional Benchmark Report - HR FAQ Chatbot

## Executive Summary

This report presents a comprehensive evaluation of the HR FAQ chatbot comparing baseline fine-tuned model performance against DSPy-optimized implementation. The benchmark was conducted on 20 HR questions and 40 out-of-domain (OOD) questions.

**Date**: 2025-12-15 (benchmark run, from the results JSON metadata)  
**Model**: microsoft/DialoGPT-small  
**Benchmark Version**: professional_v1.0

## Methodology

### Test Set Composition

- **HR Questions**: 20 questions covering various HR topics
- **OOD Questions**: 40 questions across 8 categories (technology, general knowledge, cooking, mechanical, finance, language, weather, health)

### Evaluation Metrics

1. **ROUGE-L**: Longest common subsequence-based F-measure
2. **Exact Match (EM)**: Exact string matching
3. **BLEU**: N-gram precision-based metric
4. **OOD Rejection Rate**: Percentage of out-of-domain questions correctly rejected
5. **Latency**: Average response time per question

### Statistical Analysis

- Mean, standard deviation, and 95% confidence intervals
- Paired t-test for significance testing (α = 0.05)
- Effect size (Cohen's d)
- Error categorization and analysis

## Results

### HR Questions Performance

#### ROUGE-L Score

| Model | Mean ± SD | 95% CI | Improvement |
|-------|-----------|--------|-------------|
| Baseline | 0.014 ± 0.029 | [0.001, 0.028] | - |
| DSPy | 0.126 ± 0.074 | [0.091, 0.160] | +0.111 (+784.6%) |


**Statistical Significance**: p = 0.0000 (significant)  
**Effect Size (Cohen's d)**: 1.543

#### Exact Match Score

| Model | Mean ± SD | Improvement |
|-------|-----------|-------------|
| Baseline | 0.000 ± 0.000 | - |
| DSPy | 0.000 ± 0.000 | +0.000 (+0.0%) |

#### BLEU Score

| Model | Mean ± SD | Improvement |
|-------|-----------|-------------|
| Baseline | 0.000 ± 0.000 | - |
| DSPy | 0.000 ± 0.000 | +0.000 (+0.0%) |

### Out-of-Domain Detection

| Model | Rejection Rate | 95% CI | Improvement |
|-------|---------------|--------|-------------|
| Baseline | 0.0% ± 0.0% | [0.0%, 0.0%] | - |
| DSPy | 90.0% ± 30.4% | [80.3%, 99.7%] | +90.0% (+90000.0%) |


**Statistical Significance**: p = 0.0000 (significant)  
**Effect Size (Cohen's d)**: 2.962

### Performance Metrics

#### Latency

| Model | Mean ± SD (seconds) |
|-------|---------------------|
| Baseline | 0.356 ± 0.161 |
| DSPy | 0.258 ± 0.183 |
| Difference | -0.098 (-27.5%) |

### Error Analysis

#### Baseline Model Errors

- Too short responses: 15
- Too long responses: 0
- Off-topic responses: 20
- Incomplete responses: 7

#### DSPy Model Errors

- Too short responses: 1
- Too long responses: 0
- Off-topic responses: 7
- Incomplete responses: 0

## Key Findings

1. **ROUGE-L**: +0.111 over baseline (p = 1.4e-6). 16/20 DSPy answers are the hard-coded template fallback of `HRFAQModule.forward`, which echoes the question; the gain mostly measures this template.
2. **OOD Detection**: 90.0% rejection (p = 4.2e-21), from a keyword filter that the baseline does not have.
3. **Exact Match and BLEU**: 0 for both systems, no difference.
4. **Latency**: -27.5% mean, not tested for significance.

## Limitations

- No DSPy optimizer (BootstrapFewShot, MIPRO) was run; the DSPy side is the module with a fixed prompt.
- The 20 HR test questions come from `data/train_alpaca.json`, the fine-tuning data (train/test leakage).
- Generation uses `do_sample=True`; the LoRA weights are not committed. Exact numbers are not reproducible from a fresh clone.
- Small sample: 20 HR and 40 OOD questions.

## Conclusions

The keyword OOD filter rejects most off-topic questions. The ROUGE-L difference does not show a gain from DSPy optimization.

## Recommendations

1. **Before any deployment claim**: rerun with a held-out test set, without the template fallback, and with an optimized DSPy program.
2. **Further Optimization**: Consider additional DSPy optimizers (MIPRO, GEPA) for further improvements.
3. **Dataset Expansion**: Increase test set size for more robust statistical validation.
4. **Domain-Specific Metrics**: Implement HR-specific evaluation metrics (relevance, completeness, accuracy).

---

**Report Generated**: 2025-12-17  
**Benchmark Version**: professional_v1.0
