# Explainer_XAI_Agent

Ce package fournit un agent IA combinant LIME et LangChain pour expliquer les prédictions de modèles de Machine Learning en langage naturel.

## Installation

```bash
pip install git+https://github.com/ton-github/lime_explainer_agent.git
```

## Utilisation

```python
from explainer_agent import LimeExplainerAgent
agent = LimeExplainerAgent()
agent.explain_classification(...)
```

```python
from explainer_agent import ShapExplainerAgent
agent = ShapExplainerAgent()
agent.explain_prediction(...)
