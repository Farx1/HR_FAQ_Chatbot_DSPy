# DSPy Integration for HR FAQ Chatbot

Ce module intègre DSPy pour optimiser les prompts et améliorer les performances du chatbot FAQ RH.

## Pourquoi DSPy ?

DSPy permet de :
1. **Optimiser automatiquement les prompts** : Au lieu de tweaker manuellement les prompts, DSPy trouve automatiquement les meilleurs prompts et exemples few-shot
2. **Structurer les entrées/sorties** : Les signatures DSPy définissent clairement les formats d'entrée et de sortie
3. **Améliorer le raisonnement** : ChainOfThought permet au modèle de raisonner avant de répondre
4. **Évaluer et comparer** : Facilite la comparaison des performances avant/après optimisation

## Structure

- `hr_faq_dspy.py` : Module principal avec adaptateur DSPy, signatures et modules
- `benchmark_dspy.py` : Script de benchmark comparant baseline vs DSPy
- `optimize_dspy.py` : Script d'optimisation DSPy (utilise BootstrapFewShot ou MIPRO)

## Utilisation

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Benchmark Baseline vs DSPy

```bash
python dspy_module/benchmark_dspy.py
```

Ce script :
- Évalue le modèle baseline (sans DSPy)
- Évalue le modèle avec DSPy (sans optimisation)
- Compare les résultats (Exact Match, ROUGE-L, BLEU, OOD rejection)

### 3. Optimisation DSPy (optionnel)

```bash
python dspy_module/optimize_dspy.py
```

Ce script optimise les prompts en utilisant BootstrapFewShot ou MIPRO.

### 4. Benchmark avec module optimisé

Modifiez `benchmark_dspy.py` pour utiliser `optimized=True` après l'optimisation.

## Résultats attendus

DSPy devrait améliorer :
- **Exact Match** : Meilleure correspondance avec les réponses attendues
- **ROUGE-L** : Meilleure similarité sémantique
- **BLEU** : Meilleure qualité de génération
- **OOD Rejection** : Meilleure détection des questions hors domaine

## Architecture

### HRFAQAdapter
Adaptateur personnalisé qui enveloppe le modèle fine-tuné pour qu'il fonctionne avec DSPy.

### HRFAQModule
Module DSPy utilisant ChainOfThought pour améliorer le raisonnement.

### HRFAQWithRejection
Module avancé avec détection automatique des questions hors domaine.

## Notes

- L'adaptateur utilise le modèle fine-tuné existant (DialoGPT-small pour CPU)
- DSPy optimise les prompts, pas les poids du modèle
- L'optimisation nécessite des données d'entraînement suffisantes

