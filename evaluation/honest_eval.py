"""Held-out evaluation: retrieval (Recall@k, MRR), OOD filters, optional generation.

Usage:
    python evaluation/honest_eval.py               # retrieval + OOD (CPU, ~1 min)
    python evaluation/honest_eval.py --generation  # + DialoGPT-small greedy generation

Test set: data/heldout_test.json (not taken from data/train_alpaca.json).
Deterministic: fixed test set, greedy decoding, fixed bootstrap seed.
Intervals: Wilson for proportions, percentile bootstrap for means,
paired bootstrap for system differences (Dror et al., ACL 2018).
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_SET = ROOT / "data" / "heldout_test.json"
SEED = 0
N_BOOT = 10_000
KS = (1, 3, 5)
FALLBACK_PREFIX = "According to company HR policies,"  # template in HRFAQModule.forward
REJECT_PREFIX = "I apologize, but this question appears to be outside"


def wilson(k, n, z=1.959964):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def proportion(flags):
    k, n = int(sum(flags)), len(flags)
    return {"value": k / n, "k": k, "n": n, "ci95_wilson": wilson(k, n)}


def bootstrap_mean(x, rng):
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "value": float(x.mean()),
        "n": len(x),
        "ci95_bootstrap": [float(lo), float(hi)],
    }


def paired_bootstrap(a, b, rng):
    """Mean of b - a over the same items, 95% CI, one-sided p for H0: b <= a."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    delta = float(d.mean())
    means = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Berg-Kirkpatrick et al. 2012: shift the bootstrap distribution to H0.
    p = float(np.mean(means - delta >= delta)) if delta > 0 else 1.0
    return {
        "delta": delta,
        "n": len(d),
        "ci95_bootstrap": [float(lo), float(hi)],
        "p_one_sided": p,
    }


def mcnemar_exact(a_ok, b_ok):
    """Exact McNemar test on paired binary outcomes (two-sided)."""
    only_a = sum(x and not y for x, y in zip(a_ok, b_ok))
    only_b = sum(y and not x for x, y in zip(a_ok, b_ok))
    n = only_a + only_b
    tail = (
        sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
        if n
        else 1.0
    )
    return {
        "only_first_correct": only_a,
        "only_second_correct": only_b,
        "p_two_sided": min(1.0, 2 * tail),
    }


def eval_retrieval(hr, rng):
    from backend.rag_engine import RAGEngine

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = RAGEngine(
            company_data_path=str(ROOT / "company_data" / "techcorp_solutions"),
            persist_directory=tmp,
        )
        if not engine.initialize():
            sys.exit(
                "RAG dependencies missing: pip install chromadb sentence-transformers"
            )
        n_chunks = engine.collection.count()
        rows = []
        for x in hr:
            hits = engine.search(x["question"], top_k=max(KS))
            got = [(Path(h["source"]).name, h["section_title"]) for h in hits]
            gold = (x["doc"], x["section"])
            rank = next((i + 1 for i, g in enumerate(got) if g == gold), None)
            doc_rank = next((i + 1 for i, g in enumerate(got) if g[0] == gold[0]), None)
            rows.append(
                {
                    "question": x["question"],
                    "rank": rank,
                    "doc_rank": doc_rank,
                    "top": got,
                }
            )

    out = {"embedding_model": engine.embedding_model_name, "n_chunks": n_chunks}
    for level, key in (("chunk", "rank"), ("doc", "doc_rank")):
        for k in KS:
            out[f"recall@{k}_{level}"] = proportion(
                [r[key] is not None and r[key] <= k for r in rows]
            )
    out[f"mrr@{max(KS)}_chunk"] = bootstrap_mean(
        [1 / r["rank"] if r["rank"] else 0.0 for r in rows], rng
    )
    out["per_question"] = rows
    return out


def eval_ood(hr, ood):
    from backend.server import is_hr_related as api_filter

    filters = {"api": api_filter}
    try:
        from dspy_module.hr_faq_dspy import HRFAQModule

        filters["dspy"] = HRFAQModule()._is_hr_related
    except ImportError as e:
        print(f"Skipping DSPy filter: {e}")

    hr_q = [x["question"] for x in hr]
    out, correct = {}, {}
    for name, accept in filters.items():
        ood_rejected = [not accept(q) for q in ood]
        hr_rejected = [not accept(q) for q in hr_q]
        correct[name] = ood_rejected + [not r for r in hr_rejected]
        out[name] = {
            "ood_rejection_rate": proportion(ood_rejected),
            "hr_false_rejection_rate": proportion(hr_rejected),
            "ood_accepted": [q for q, r in zip(ood, ood_rejected) if not r],
            "hr_rejected": [q for q, r in zip(hr_q, hr_rejected) if r],
        }
    if len(filters) == 2:
        out["mcnemar_api_vs_dspy"] = mcnemar_exact(correct["api"], correct["dspy"])
    return out


def eval_generation(hr, rng, model_name="microsoft/DialoGPT-small"):
    from types import SimpleNamespace

    import torch
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dspy_module.benchmark_dspy import generate_baseline_response
    from dspy_module.hr_faq_dspy import HRFAQModule

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    def greedy(**kw):  # override the sampling settings hard-coded by the callers
        kw.update(do_sample=False, temperature=None, top_p=None)
        return model.generate(**kw)

    lm = SimpleNamespace(generate=greedy)
    module = HRFAQModule(adapter=SimpleNamespace(tokenizer=tok, pytorch_model=lm))
    scorer = rouge_scorer.RougeScorer(["rougeL"])

    def rouge(pred, ref):
        return scorer.score(ref, pred)["rougeL"].fmeasure if pred else 0.0

    rows, t0 = [], time.time()
    for x in hr:
        base = generate_baseline_response(lm, tok, x["question"])
        pipe = module(question=x["question"]).answer
        fallback, rejected = (
            pipe.startswith(FALLBACK_PREFIX),
            pipe.startswith(REJECT_PREFIX),
        )
        rows.append(
            {
                "question": x["question"],
                "baseline": base,
                "pipeline": pipe,
                "fallback": fallback,
                "rejected": rejected,
                "rougeL_baseline": rouge(base, x["reference"]),
                "rougeL_pipeline_raw": rouge(pipe, x["reference"]),
                # A template or a refusal is not an answer: score it as a failure.
                "rougeL_pipeline": 0.0
                if fallback or rejected
                else rouge(pipe, x["reference"]),
            }
        )
    col = lambda k: [r[k] for r in rows]  # noqa: E731
    return {
        "model": model_name,
        "lora_adapter": "none (weights not in the repo)",
        "decoding": "greedy (do_sample=False), max_new_tokens 128 baseline / 150 pipeline",
        "seconds": round(time.time() - t0, 1),
        "rougeL_baseline": bootstrap_mean(col("rougeL_baseline"), rng),
        "rougeL_pipeline": bootstrap_mean(col("rougeL_pipeline"), rng),
        "rougeL_pipeline_counting_template": bootstrap_mean(
            col("rougeL_pipeline_raw"), rng
        ),
        "pipeline_vs_baseline": paired_bootstrap(
            col("rougeL_baseline"), col("rougeL_pipeline"), rng
        ),
        "fallback_rate": proportion(col("fallback")),
        "hr_rejected_rate": proportion(col("rejected")),
        "per_question": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--generation", action="store_true", help="also run DialoGPT-small generation"
    )
    ap.add_argument("--out", default=str(ROOT / "reports" / "honest_eval.json"))
    args = ap.parse_args()

    data = json.loads(TEST_SET.read_text(encoding="utf-8"))
    train = json.loads(
        (ROOT / "data" / "train_alpaca.json").read_text(encoding="utf-8")
    )
    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    leaked = {norm(t["instruction"]) for t in train} & {
        norm(x["question"]) for x in data["hr"]
    }
    assert not leaked, f"test questions found in train set: {leaked}"

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    results = {
        "test_set": str(TEST_SET.relative_to(ROOT)).replace("\\", "/"),
        "test_set_sha256": hashlib.sha256(TEST_SET.read_bytes()).hexdigest(),
        "git_commit": commit.stdout.strip(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_hr": len(data["hr"]),
        "n_ood": len(data["ood"]),
        "retrieval": eval_retrieval(data["hr"], np.random.default_rng(SEED)),
        "ood": eval_ood(data["hr"], data["ood"]),
    }
    if args.generation:
        results["generation"] = eval_generation(data["hr"], np.random.default_rng(SEED))

    Path(args.out).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for k, v in results["retrieval"].items():
        if isinstance(v, dict) and "value" in v:
            print(f"retrieval {k}: {v['value']:.3f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
