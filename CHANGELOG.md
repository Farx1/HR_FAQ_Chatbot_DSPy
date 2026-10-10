# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-12-17

### Added
- Professional benchmark with statistical significance testing
- DSPy integration for automatic prompt optimization
- Comprehensive evaluation metrics (ROUGE-L, BLEU, Exact Match, OOD Rejection)
- Error analysis and categorization
- Professional benchmark report generation
- GitHub-ready documentation and structure

### Changed
- Improved OOD detection with keyword-based classification
- Enhanced prompts for better response quality
- Optimized model parameters for better performance

### Results
- ROUGE-L: 0.014 → 0.126 (+0.111), mostly from the hard-coded template fallback (16/20 answers); see README Limitations
- OOD Rejection: 0% → 90% (keyword filter on the DSPy side only)
- Error counts: too-short responses 15 → 1 (-93%), off-topic responses 20 → 7 (-65%)

## [0.1.0] - Initial Release

### Added
- Basic fine-tuning pipeline
- LoRA/QLoRA support
- Evaluation scripts
- Interactive demos

