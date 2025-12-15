
# Rapport d'Évaluation - Chatbot FAQ RH (Version CPU)

## Résumé Exécutif

Le modèle DialoGPT-small fine-tuné avec LoRA a été évalué sur 5 questions RH et 10 questions hors domaine.

**Note:** Cette évaluation utilise un modèle réduit (DialoGPT-small) pour démonstration sur CPU. Les performances réelles avec Mistral-7B seraient significativement meilleures.

### Métriques Principales

**Questions RH:**
- Exact Match: 0.000 (objectif démo: ≥ 0.3)
- ROUGE-L: 0.014
- BLEU: 0.000

**Questions Hors Domaine:**
- Taux de Refus: 0.000 (objectif démo: ≥ 0.5)

### Critères d'Acceptation (Démo)

✗ CRITÈRES NON ATTEINTS

## Exemples de Réponses


### Exemple HR 1
**Question:** What is the company's sick leave policy?
**Réponse Attendue:** Employees can take up to 5 sick days per year with manager approval. Extended sick leave requires medical documentation.
**Réponse Générée:** It doesn't matter, it matters how you work.
**Score EM:** 0.000

### Exemple HR 2
**Question:** How do I request a salary review?
**Réponse Attendue:** Submit a salary review request through the HR portal with supporting documentation. Reviews are conducted annually or upon promotion.
**Réponse Générée:** This is your job.
**Score EM:** 0.000

### Exemple HR 3
**Question:** What training opportunities are available?
**Réponse Attendue:** The company offers various training programs including technical skills, leadership development, and professional certifications. Check the learning portal for available courses.
**Réponse Générée:** Working on a new project at the moment.
**Score EM:** 0.000

## Tests Hors Domaine

### Exemple OOD 1
**Question:** How do I install Python on my computer?
**Réponse:** It's a python script
**Refus Correct:** ✗

### Exemple OOD 2
**Question:** What is the capital of France?
**Réponse:** No, I don't think so.
**Refus Correct:** ✗

### Exemple OOD 3
**Question:** How do I bake a chocolate cake?
**Réponse:** Yes, you can.
**Refus Correct:** ✗


## Recommandations pour Production

1. **Modèle de Production:** Utiliser Mistral-7B-Instruct-v0.3 avec GPU pour de meilleures performances.

2. **Dataset:** Augmenter le dataset d'entraînement avec plus d'exemples RH variés.

3. **Entraînement:** Augmenter le nombre d'époques et ajuster les hyperparamètres.

4. **Évaluation:** Implémenter des tests plus robustes avec des métriques spécifiques au domaine RH.

## Conclusion

Cette démonstration montre le pipeline complet de fine-tuning d'un modèle pour les FAQ RH. 
Le prototype nécessite des améliorations 
mais nécessite un modèle plus puissant et plus de données pour un déploiement en production.
