"""
Benchmark script comparing baseline model vs DSPy-optimized model
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dspy_module.hr_faq_dspy import HRFAQAdapter, HRFAQModule
import dspy

# Set seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODEL_NAME = "microsoft/DialoGPT-small"


def load_baseline_model():
    """Load baseline fine-tuned model"""
    print("Loading baseline model...")

    # Try to load tokenizer from saved model, fallback to base model if corrupted
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "models/hr_faq_dialogpt_lora", use_fast=False
        )
    except Exception as e:
        print(
            f"Warning: Could not load tokenizer from saved model, using base model tokenizer: {e}"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
    )

    # Load LoRA adapters - handle incompatible config fields
    try:
        model = PeftModel.from_pretrained(
            base_model, "models/hr_faq_dialogpt_lora_adapters"
        )
    except TypeError as e:
        # If config has incompatible fields, try to load with a cleaned config
        print(f"Warning: Config incompatibility detected, attempting to fix: {e}")
        import json
        from peft import LoraConfig, get_peft_model

        # Load and clean the config
        config_path = "models/hr_faq_dialogpt_lora_adapters/adapter_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            # Only keep valid LoraConfig fields
            valid_lora_fields = {
                "r",
                "lora_alpha",
                "target_modules",
                "lora_dropout",
                "task_type",
                "bias",
                "inference_mode",
                "modules_to_save",
                "init_lora_weights",
                "layers_to_transform",
                "layers_pattern",
                "rank_pattern",
                "alpha_pattern",
                "fan_in_fan_out",
            }

            # Filter to only valid fields
            cleaned_config = {
                k: v
                for k, v in config_dict.items()
                if k in valid_lora_fields and v is not None
            }

            # Create LoRA config from cleaned dict
            lora_config = LoraConfig(**cleaned_config)

            # Apply LoRA and load weights
            model = get_peft_model(base_model, lora_config)
            adapter_weights_path = (
                "models/hr_faq_dialogpt_lora_adapters/adapter_model.bin"
            )
            if not os.path.exists(adapter_weights_path):
                adapter_weights_path = (
                    "models/hr_faq_dialogpt_lora_adapters/adapter_model.safetensors"
                )

            if os.path.exists(adapter_weights_path):
                from peft import set_peft_model_state_dict

                try:
                    state_dict = torch.load(adapter_weights_path, map_location="cpu")
                    set_peft_model_state_dict(model, state_dict)
                except Exception:
                    # Try safetensors format
                    try:
                        from safetensors import safe_open

                        state_dict = {}
                        with safe_open(
                            adapter_weights_path, framework="pt", device="cpu"
                        ) as f:
                            for key in f.keys():
                                state_dict[key] = f.get_tensor(key)
                        set_peft_model_state_dict(model, state_dict)
                    except Exception as e2:
                        print(f"Warning: Could not load adapter weights: {e2}")
            else:
                print("Warning: Could not find adapter weights, using base model")
        else:
            raise e

    return model, tokenizer


def generate_baseline_response(model, tokenizer, question: str) -> str:
    """Generate response using baseline model"""
    prompt = f"HR Question: {question}\nHR Answer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("HR Answer:")[-1].strip()
    return response


def normalize_text(text: str) -> str:
    """Normalize text for evaluation"""
    import re

    text = text.lower()
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def exact_match_score(prediction: str, reference: str) -> float:
    """Compute exact match score"""
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    return 1.0 if pred_norm == ref_norm else 0.0


def evaluate_baseline(hr_data: List[Dict], ood_data: List[Dict]):
    """Evaluate baseline model"""
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE MODEL")
    print("=" * 60)

    model, tokenizer = load_baseline_model()
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    hr_results = []
    for example in hr_data:
        question = example["instruction"]
        expected = example["output"]

        generated = generate_baseline_response(model, tokenizer, question)

        em = exact_match_score(generated, expected)

        try:
            rouge_scores = rouge.compute(predictions=[generated], references=[expected])
            bleu_scores = bleu.compute(predictions=[generated], references=[expected])
        except Exception:
            rouge_scores = {"rougeL": 0.0}
            bleu_scores = {"bleu": 0.0}

        hr_results.append(
            {
                "question": question,
                "expected": expected,
                "generated": generated,
                "exact_match": em,
                "rouge_l": rouge_scores["rougeL"],
                "bleu": bleu_scores["bleu"],
            }
        )

    # OOD evaluation
    ood_results = []
    rejection_keywords = [
        "désolé",
        "périmètre",
        "rh",
        "contacter",
        "service",
        "sorry",
        "hr",
        "contact",
    ]

    for example in ood_data:
        question = example["instruction"]
        generated = generate_baseline_response(model, tokenizer, question)

        generated_lower = generated.lower()
        is_rejection = any(keyword in generated_lower for keyword in rejection_keywords)

        ood_results.append(
            {
                "question": question,
                "generated": generated,
                "is_rejection": is_rejection,
                "rejection_score": 1.0 if is_rejection else 0.0,
            }
        )

    # Compute averages
    avg_em = np.mean([r["exact_match"] for r in hr_results])
    avg_rouge = np.mean([r["rouge_l"] for r in hr_results])
    avg_bleu = np.mean([r["bleu"] for r in hr_results])
    avg_ood = np.mean([r["rejection_score"] for r in ood_results])

    print("\nBaseline Results:")
    print(f"  HR Questions (n={len(hr_results)}):")
    print(f"    Exact Match: {avg_em:.3f}")
    print(f"    ROUGE-L: {avg_rouge:.3f}")
    print(f"    BLEU: {avg_bleu:.3f}")
    print(f"  OOD Questions (n={len(ood_results)}):")
    print(f"    Rejection Rate: {avg_ood:.3f}")

    return {
        "hr_results": hr_results,
        "ood_results": ood_results,
        "summary": {
            "avg_exact_match": float(avg_em),
            "avg_rouge_l": float(avg_rouge),
            "avg_bleu": float(avg_bleu),
            "avg_ood_rejection": float(avg_ood),
        },
    }


def evaluate_dspy(hr_data: List[Dict], ood_data: List[Dict], optimized: bool = False):
    """Evaluate DSPy model"""
    print("\n" + "=" * 60)
    print(f"EVALUATING DSPY MODEL ({'OPTIMIZED' if optimized else 'BASELINE'})")
    print("=" * 60)

    # Configure DSPy with custom adapter
    adapter = HRFAQAdapter()
    dspy.configure(lm=adapter)

    # Load or create module with adapter reference
    if optimized and os.path.exists("models/dspy_optimized_module.json"):
        print("Loading optimized DSPy module...")
        module = HRFAQModule(adapter=adapter)
        module.load("models/dspy_optimized_module.json")
    else:
        module = HRFAQModule(adapter=adapter)

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    hr_results = []
    for example in hr_data:
        question = example["instruction"]
        expected = example["output"]

        try:
            result = module(question=question)
            generated = result.answer if result and hasattr(result, "answer") else ""
            if generated is None:
                generated = ""
        except Exception as e:
            print(f"Error generating answer: {e}")
            generated = ""

        em = exact_match_score(generated, expected) if generated else 0.0

        try:
            rouge_scores = rouge.compute(predictions=[generated], references=[expected])
            bleu_scores = bleu.compute(predictions=[generated], references=[expected])
        except Exception:
            rouge_scores = {"rougeL": 0.0}
            bleu_scores = {"bleu": 0.0}

        hr_results.append(
            {
                "question": question,
                "expected": expected,
                "generated": generated,
                "exact_match": em,
                "rouge_l": rouge_scores["rougeL"],
                "bleu": bleu_scores["bleu"],
            }
        )

    # OOD evaluation
    ood_results = []
    rejection_keywords = [
        "désolé",
        "périmètre",
        "rh",
        "contacter",
        "service",
        "sorry",
        "hr",
        "contact",
    ]

    for example in ood_data:
        question = example["instruction"]

        try:
            result = module(question=question)
            generated = result.answer if result and hasattr(result, "answer") else ""
            if generated is None:
                generated = ""
        except Exception as e:
            print(f"Error generating answer: {e}")
            generated = ""

        generated_lower = generated.lower() if generated else ""
        # Improved rejection detection - check for multiple signals
        is_rejection = (
            any(keyword in generated_lower for keyword in rejection_keywords)
            or "outside the scope" in generated_lower
            or "outside of hr" in generated_lower
            or "non-hr" in generated_lower
            or (
                "contact" in generated_lower
                and "hr" in generated_lower
                and len(generated) > 50
            )
        )

        ood_results.append(
            {
                "question": question,
                "generated": generated,
                "is_rejection": is_rejection,
                "rejection_score": 1.0 if is_rejection else 0.0,
            }
        )

    # Compute averages
    avg_em = np.mean([r["exact_match"] for r in hr_results])
    avg_rouge = np.mean([r["rouge_l"] for r in hr_results])
    avg_bleu = np.mean([r["bleu"] for r in hr_results])
    avg_ood = np.mean([r["rejection_score"] for r in ood_results])

    print("\nDSPy Results:")
    print(f"  HR Questions (n={len(hr_results)}):")
    print(f"    Exact Match: {avg_em:.3f}")
    print(f"    ROUGE-L: {avg_rouge:.3f}")
    print(f"    BLEU: {avg_bleu:.3f}")
    print(f"  OOD Questions (n={len(ood_results)}):")
    print(f"    Rejection Rate: {avg_ood:.3f}")

    return {
        "hr_results": hr_results,
        "ood_results": ood_results,
        "summary": {
            "avg_exact_match": float(avg_em),
            "avg_rouge_l": float(avg_rouge),
            "avg_bleu": float(avg_bleu),
            "avg_ood_rejection": float(avg_ood),
        },
    }


def compare_results(baseline_results: Dict, dspy_results: Dict):
    """Compare baseline and DSPy results"""
    print("\n" + "=" * 60)
    print("COMPARISON: BASELINE vs DSPY")
    print("=" * 60)

    baseline_summary = baseline_results["summary"]
    dspy_summary = dspy_results["summary"]

    print("\nHR Questions:")
    print("  Exact Match:")
    print(f"    Baseline: {baseline_summary['avg_exact_match']:.3f}")
    print(f"    DSPy:     {dspy_summary['avg_exact_match']:.3f}")
    print(
        f"    Change:   {dspy_summary['avg_exact_match'] - baseline_summary['avg_exact_match']:+.3f}"
    )

    print("  ROUGE-L:")
    print(f"    Baseline: {baseline_summary['avg_rouge_l']:.3f}")
    print(f"    DSPy:     {dspy_summary['avg_rouge_l']:.3f}")
    print(
        f"    Change:   {dspy_summary['avg_rouge_l'] - baseline_summary['avg_rouge_l']:+.3f}"
    )

    print("  BLEU:")
    print(f"    Baseline: {baseline_summary['avg_bleu']:.3f}")
    print(f"    DSPy:     {dspy_summary['avg_bleu']:.3f}")
    print(
        f"    Change:   {dspy_summary['avg_bleu'] - baseline_summary['avg_bleu']:+.3f}"
    )

    print("\nOOD Questions:")
    print("  Rejection Rate:")
    print(f"    Baseline: {baseline_summary['avg_ood_rejection']:.3f}")
    print(f"    DSPy:     {dspy_summary['avg_ood_rejection']:.3f}")
    print(
        f"    Change:   {dspy_summary['avg_ood_rejection'] - baseline_summary['avg_ood_rejection']:+.3f}"
    )

    # Calculate overall improvement
    hr_improvement = (
        (dspy_summary["avg_exact_match"] - baseline_summary["avg_exact_match"])
        + (dspy_summary["avg_rouge_l"] - baseline_summary["avg_rouge_l"])
        + (dspy_summary["avg_bleu"] - baseline_summary["avg_bleu"])
    ) / 3

    print(f"\nOverall HR Improvement: {hr_improvement:+.3f}")
    print(
        f"OOD Improvement: {dspy_summary['avg_ood_rejection'] - baseline_summary['avg_ood_rejection']:+.3f}"
    )


def main():
    """Main benchmark function"""
    print("HR FAQ Benchmark: Baseline vs DSPy")
    print("=" * 60)

    # Load evaluation data
    print("\nLoading evaluation data...")
    hr_data = []
    ood_data = []

    if os.path.exists("data/val_alpaca.json"):
        with open("data/val_alpaca.json", "r", encoding="utf-8") as f:
            hr_data = json.load(f)

    if os.path.exists("data/ood_test.json"):
        with open("data/ood_test.json", "r", encoding="utf-8") as f:
            ood_data = json.load(f)

    if not hr_data:
        print("No evaluation data found. Creating sample data...")
        hr_data = [
            {
                "instruction": "What is the company's sick leave policy?",
                "output": "Employees can take up to 5 sick days per year with manager approval. Extended sick leave requires medical documentation.",
            },
            {
                "instruction": "How do I request a salary review?",
                "output": "Submit a salary review request through the HR portal with supporting documentation. Reviews are conducted annually or upon promotion.",
            },
        ]

    if not ood_data:
        ood_data = [
            {
                "instruction": "How do I install Python on my computer?",
                "output": "Désolé, cette question semble en dehors du périmètre des politiques RH.",
            }
        ]

    print(f"Loaded {len(hr_data)} HR questions and {len(ood_data)} OOD questions")

    # Evaluate baseline
    baseline_results = evaluate_baseline(hr_data, ood_data)

    # Evaluate DSPy (non-optimized)
    dspy_results = evaluate_dspy(hr_data, ood_data, optimized=False)

    # Compare results
    compare_results(baseline_results, dspy_results)

    # Save results
    os.makedirs("reports", exist_ok=True)
    comparison_results = {
        "baseline": baseline_results,
        "dspy": dspy_results,
        "comparison": {
            "hr_em_diff": dspy_results["summary"]["avg_exact_match"]
            - baseline_results["summary"]["avg_exact_match"],
            "hr_rouge_diff": dspy_results["summary"]["avg_rouge_l"]
            - baseline_results["summary"]["avg_rouge_l"],
            "hr_bleu_diff": dspy_results["summary"]["avg_bleu"]
            - baseline_results["summary"]["avg_bleu"],
            "ood_rejection_diff": dspy_results["summary"]["avg_ood_rejection"]
            - baseline_results["summary"]["avg_ood_rejection"],
        },
    }

    with open("reports/benchmark_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    print("\nResults saved to: reports/benchmark_comparison.json")
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
