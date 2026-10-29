# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Security
- `POST /rebuild-index` requires `ADMIN_TOKEN` (header `X-Admin-Token`); disabled when unset
- CORS limited to `CORS_ORIGINS` (default `http://localhost:3000`) instead of `*` with credentials

### Added
- `SECURITY.md`, `CITATION.cff`, pre-commit config (ruff), Dependabot
- CI: actions pinned by commit SHA, read-only `GITHUB_TOKEN`

## 2025-12-18 (from git history, no tag)

### Fixed
- CI: pytest exit code 5, TypeScript types, Next.js path alias, Next.js build cache (`80c5831`, `34816d5`, `c26053b`, `036b99f`, `4d0f356`)
- `.gitignore` excluded `tests/`; test files added (`d664959`, `4c94718`)

### Changed
- README rewritten (`b1bd786`)

## [1.0.0] - 2025-12-17

Git: `55f8d04` (2025-12-15), `0272298` web UI, `5a45b51` cleanup, `ba49dda` early OOD check in `/ask`. No git tag exists. The results below were written at the time and are not reproduced; see the README benchmark section.

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

