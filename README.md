# Explainer_XAI_Agent

Ce package fournit un agent LIME et un agent SHAP pour expliquer les prédictions de modèles de Machine Learning en langage naturel.

## Attention: installastion du serveur Ollama obligatoire
Le package fonctionne grâce à Ollama. Il faut donc initialiser son serveur Ollama afin de pourvoir utilser le package.
Voici un code pour lancer un serveur local Ollama sur Google Colab.

```bash
!sudo apt update

!sudo apt install -y pciutils

!curl -fsSL https://ollama.com/install.sh | sh
```

```python
import threading
import subprocess
import time

def run_ollama_serve():
    subprocess.Popen(["ollama", "serve"])

thread = threading.Thread(target=run_ollama_serve)

thread.start()
time.sleep(5)  # Allows service to initialize
```

```bash
!ollama pull llama3.2
```

## Installation

```bash
!pip install git+https://github.com/GautFR/Explainer_XAI_Agent.git
```

## Utilisation

### Fonction pour LIME local
```python
from explainer import LimeExplainerAgent
agent = LimeExplainerAgent(
            llm_provider=...,
            model_name=...,
            temperature=...,
            api_key=...
        )
agent.explain_classification(...)
#ou
agent.explain_regression(...)
```

### Fonction pour SHAP local
```python
from explainer import ShapExplainerAgent
agent = ShapExplainerAgent(
            llm_provider=...,
            model_name=...,
            temperature=...,
            api_key=...
        )
agent.explain_classification(...)
#ou
agent.explain_regression(...)
```

### Fonction pour SHAP Global
```python
from explainer import ShapExplainerAgentGlobal
agent = ShapExplainerAgentGlobal(
            llm_provider=...,
            model_name=...,
            temperature=...,
            api_key=...
        )
agent.explain_classification(...)
#ou
agent.explain_regression(...)
```

## Exemple Classification LIME

```python
from explainer import LimeExplainerAgent
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
```

```python
def demo_classification_package():
    """Démontre l'utilisation de l'agent en classification avec le jeu de données Iris"""
    from sklearn.datasets import load_iris # Corrected indentation
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Chargement des données
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    class_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Contexte d'explication
    context = "Jeu de données Iris avec des mesures de sépales/pétales de trois espèces d'iris."

    # Liste des LLM à tester
    llm_configs = [
        {"provider": "gemini", "model": "gemini-2.0-flash", "api_key": userdata.get("GOOGLE_API_KEY")},
        #{"provider": "ollama", "model": "llama3.2", "api_key": None}
    ]

    # Pour chaque IA
    for config in llm_configs:
        print(f"\n--- Explication avec {config['provider'].upper()} ({config['model']}) ---")
        explainer = LimeExplainerAgent(
            llm_provider=config["provider"],
            model_name=config["model"],
            temperature=0.2,
            api_key=config.get("api_key")
        )

        # Afficher les fournisseurs disponibles une seule fois (optionnel)
        if config == llm_configs[0]:
            providers_info = explainer.get_available_providers()
            for provider, info in providers_info.items():
                print(f"\n{provider.upper()} - {info['description']}")
                print(f"  Modèles recommandés: {', '.join(info['models_recommandés'])}")
                print(f"  Nécessite API key: {'Oui' if info['nécessite_api_key'] else 'Non'}")

        # Générer l'explication
        explainer.explain_classification(
            X_train, X_test, y_test, model, context,
            class_names_dict=class_names,
            instance_index=5,
            num_features=4
        )

if __name__ == "__main__":
    print("=== DÉMONSTRATION EN CLASSIFICATION ===")
    demo_classification_package()
```
