# Professional Benchmark Documentation

## Overview

The professional benchmark provides comprehensive evaluation of the HR FAQ chatbot with statistical significance testing, error analysis, and detailed performance metrics.

## Running the Benchmark

```bash
python benchmark_professional.py
```

## Test Set

- **HR Questions**: 20 questions covering various HR topics (vacation, leave, training, etc.)
- **OOD Questions**: 40 questions across 8 categories:
  - Technology (5 questions)
  - General Knowledge (5 questions)
  - Cooking (5 questions)
  - Mechanical (5 questions)
  - Finance (5 questions)
  - Language (5 questions)
  - Weather (5 questions)
  - Health (5 questions)

## Metrics

### Primary Metrics

1. **ROUGE-L**: Longest common subsequence-based F-measure
   - Measures semantic similarity between generated and expected answers
   - Range: 0.0 to 1.0 (higher is better)

2. **Exact Match (EM)**: Exact string matching
   - Binary metric: 1.0 if exact match, 0.0 otherwise
   - Range: 0.0 to 1.0 (higher is better)

3. **BLEU**: N-gram precision-based metric
   - Measures n-gram overlap between generated and expected answers
   - Range: 0.0 to 1.0 (higher is better)

4. **OOD Rejection Rate**: Percentage of out-of-domain questions correctly rejected
   - Measures ability to detect and reject non-HR questions
   - Range: 0.0 to 1.0 (higher is better)

5. **Latency**: Average response time per question
   - Measured in seconds
   - Lower is better

### Statistical Analysis

- **Mean and Standard Deviation**: Central tendency and variability
- **95% Confidence Intervals**: Uncertainty quantification
- **Paired t-test**: Statistical significance testing (α = 0.05)
- **Effect Size (Cohen's d)**: Magnitude of improvement

## Results Interpretation

### Statistical Significance

- **p < 0.05**: Statistically significant improvement
- **p ≥ 0.05**: Not statistically significant (may need larger sample)

### Effect Size

- **|d| < 0.2**: Negligible effect
- **|d| < 0.5**: Small effect
- **|d| < 0.8**: Medium effect
- **|d| ≥ 0.8**: Large effect

### Error Categories

- **Too Short**: Responses with < 5 words
- **Too Long**: Responses with > 200 words
- **Off-topic**: Responses with ROUGE-L < 0.1
- **Incomplete**: Responses not ending with punctuation

## Output Files

- `reports/professional_benchmark_results.json`: Complete results in JSON format
- `reports/professional_benchmark_report.md`: Human-readable report

## Example Results

```
HR Questions - ROUGE-L:
  Baseline: 0.014 ± 0.029 [0.001, 0.028]
  DSPy:     0.126 ± 0.074 [0.091, 0.160]
  Improvement: +0.111
  Significance: p=0.0000 (*)

OOD Questions - Rejection Rate:
  Baseline: 0.0% ± 0.0% (0.0%)
  DSPy:     90.0% ± 30.4% (90.0%)
  Improvement: +0.900 (+90.0%)
  Significance: p=0.0000 (*)
```

## Best Practices

1. **Run multiple times**: For more robust results, run the benchmark multiple times
2. **Check confidence intervals**: Wider intervals indicate more uncertainty
3. **Review error analysis**: Understand where the model fails
4. **Compare effect sizes**: Large effect sizes indicate meaningful improvements

## Troubleshooting

### Small Test Set

If you see warnings about small test sets:
- The benchmark will still run but results may not be statistically significant
- Consider expanding the test set for more robust evaluation

### Memory Issues

If you encounter memory issues:
- Use the CPU version: `training/train_cpu.py`
- Reduce batch size in `config.py`
- Use smaller model: DialoGPT-small instead of Mistral-7B

## References

- ROUGE: Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries.
- BLEU: Papineni, K., et al. (2002). BLEU: a method for automatic evaluation of machine translation.
- Statistical Testing: Student's t-test for paired samples

