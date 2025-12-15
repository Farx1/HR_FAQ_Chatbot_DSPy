"""
Script principal pour exécuter le benchmark complet
Compare baseline vs DSPy et génère un rapport
"""

import os
import sys
import json
from pathlib import Path

def check_requirements():
    """Vérifier que tous les prérequis sont satisfaits"""
    print("Vérification des prérequis...")
    
    checks = {
        "Modèle entraîné": os.path.exists("models/hr_faq_dialogpt_lora"),
        "Données d'évaluation": os.path.exists("data/val_alpaca.json") or os.path.exists("data/ood_test.json"),
        "DSPy installé": False
    }
    
    try:
        import dspy
        checks["DSPy installé"] = True
    except ImportError:
        pass
    
    all_ok = all(checks.values())
    
    for check, status in checks.items():
        status_str = "[OK]" if status else "[FAIL]"
        print(f"  {status_str} {check}")
    
    if not all_ok:
        print("\n[WARNING] Certains prerequis manquent.")
        if not checks["Modèle entraîné"]:
            print("   -> Executez: python training/train_cpu.py")
        if not checks["DSPy installé"]:
            print("   -> Executez: pip install dspy-ai")
        return False
    
    print("\n[OK] Tous les prerequis sont satisfaits!\n")
    return True


def run_baseline_evaluation():
    """Exécuter l'évaluation baseline"""
    print("="*60)
    print("ÉTAPE 1: Évaluation Baseline")
    print("="*60)
    
    if os.path.exists("reports/evaluation_results.json"):
        print("Résultats baseline déjà disponibles.")
        with open("reports/evaluation_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
        return results
    
    print("Exécution de l'évaluation baseline...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "evaluation/eval_cpu.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("[OK] Evaluation baseline terminee")
            if os.path.exists("reports/evaluation_results.json"):
                with open("reports/evaluation_results.json", "r", encoding="utf-8") as f:
                    return json.load(f)
        else:
            print(f"[ERROR] Erreur: {result.stderr}")
            return None
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'evaluation: {e}")
        return None


def run_dspy_benchmark():
    """Exécuter le benchmark DSPy"""
    print("\n" + "="*60)
    print("ÉTAPE 2: Benchmark DSPy")
    print("="*60)
    
    print("Exécution du benchmark DSPy...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "dspy_module/benchmark_dspy.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("[OK] Benchmark DSPy termine")
            print(result.stdout)
            
            if os.path.exists("reports/benchmark_comparison.json"):
                with open("reports/benchmark_comparison.json", "r", encoding="utf-8") as f:
                    return json.load(f)
        else:
            print(f"[ERROR] Erreur: {result.stderr}")
            print(result.stdout)
            return None
    except Exception as e:
        print(f"[ERROR] Erreur lors du benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_summary_report(baseline_results, comparison_results):
    """Générer un rapport de synthèse"""
    print("\n" + "="*60)
    print("RAPPORT DE SYNTHÈSE")
    print("="*60)
    
    if not baseline_results or not comparison_results:
        print("Données insuffisantes pour générer le rapport.")
        return
    
    baseline_summary = baseline_results.get("summary", {})
    comparison = comparison_results.get("comparison", {})
    
    report = f"""
# Rapport de Benchmark - Baseline vs DSPy

## Résultats Baseline

### Questions RH
- Exact Match: {baseline_summary.get('avg_exact_match', 0):.3f}
- ROUGE-L: {baseline_summary.get('avg_rouge_l', 0):.3f}
- BLEU: {baseline_summary.get('avg_bleu', 0):.3f}

### Questions Hors Domaine
- Taux de Refus: {baseline_summary.get('avg_ood_rejection', 0):.3f}

## Résultats DSPy

### Questions RH
- Exact Match: {baseline_summary.get('avg_exact_match', 0) + comparison.get('hr_em_diff', 0):.3f}
- ROUGE-L: {baseline_summary.get('avg_rouge_l', 0) + comparison.get('hr_rouge_diff', 0):.3f}
- BLEU: {baseline_summary.get('avg_bleu', 0) + comparison.get('hr_bleu_diff', 0):.3f}

### Questions Hors Domaine
- Taux de Refus: {baseline_summary.get('avg_ood_rejection', 0) + comparison.get('ood_rejection_diff', 0):.3f}

## Améliorations DSPy

- Exact Match: {comparison.get('hr_em_diff', 0):+.3f}
- ROUGE-L: {comparison.get('hr_rouge_diff', 0):+.3f}
- BLEU: {comparison.get('hr_bleu_diff', 0):+.3f}
- OOD Rejection: {comparison.get('ood_rejection_diff', 0):+.3f}

## Conclusion

DSPy {'améliore' if any(v > 0 for v in comparison.values()) else 'ne modifie pas significativement'} les performances du modèle.
Les améliorations sont {'significatives' if max(comparison.values()) > 0.1 else 'modestes'}.
"""
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(report)
    print(f"\nRapport sauvegardé dans: reports/benchmark_summary.md")


def main():
    """Fonction principale"""
    print("="*60)
    print("BENCHMARK COMPLET - Baseline vs DSPy")
    print("="*60)
    print()
    
    # Vérifier les prérequis
    if not check_requirements():
        print("\n❌ Prérequis non satisfaits. Veuillez les installer avant de continuer.")
        return
    
    # Évaluation baseline
    baseline_results = run_baseline_evaluation()
    
    # Benchmark DSPy
    comparison_results = run_dspy_benchmark()
    
    # Générer le rapport
    if baseline_results and comparison_results:
        generate_summary_report(baseline_results, comparison_results)
        print("\n" + "="*60)
        print("[OK] BENCHMARK TERMINE AVEC SUCCES")
        print("="*60)
        print("\nFichiers generes:")
        print("  - reports/evaluation_results.json (baseline)")
        print("  - reports/benchmark_comparison.json (comparaison)")
        print("  - reports/benchmark_summary.md (rapport)")
    else:
        print("\n[WARNING] Le benchmark n'a pas pu etre complete entierement.")
        print("Verifiez les erreurs ci-dessus.")


if __name__ == "__main__":
    main()

