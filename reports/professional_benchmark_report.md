# Professional Benchmark Report - HR FAQ Chatbot

## Executive Summary

This report presents a comprehensive evaluation of the HR FAQ chatbot comparing baseline fine-tuned model performance against DSPy-optimized implementation. The benchmark was conducted on 20 HR questions and 40 out-of-domain (OOD) questions.

**Date**: 2025-12-15 14:27:06  
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

1. **ROUGE-L Improvement**: DSPy shows an improvement of +0.111 (+784.6%) over baseline.
2. **OOD Detection**: DSPy achieves 90.0% rejection rate, an improvement of +90.0 percentage points.
3. **Statistical Significance**: Significant improvements.

## Conclusions

DSPy optimization demonstrates significant improvements in:
- Semantic similarity (ROUGE-L)
- Out-of-domain question detection
- Response quality and structure

The improvements are statistically significant.

## Recommendations

1. **Production Deployment**: DSPy-optimized model shows superior performance and is recommended for production use.
2. **Further Optimization**: Consider additional DSPy optimizers (MIPRO, GEPA) for further improvements.
3. **Dataset Expansion**: Increase test set size for more robust statistical validation.
4. **Domain-Specific Metrics**: Implement HR-specific evaluation metrics (relevance, completeness, accuracy).

---

**Report Generated**: 2025-12-15 14:27:06  
**Benchmark Version**: professional_v1.0
