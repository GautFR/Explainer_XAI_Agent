# Explainer_XAI_Agent

Ce package fournit un agent LIME et un agent SHAP pour expliquer les prédictions de modèles de Machine Learning en langage naturel.

## Attention: installation du serveur Ollama obligatoire
Le package fonctionne par défaut avec Ollama. Il faut donc initialiser son serveur Ollama afin de pourvoir utilser le package.
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

LIME (Local Interpretable Model-agnostic Explanations) est une méthode d’explicabilité qui approxime localement un modèle complexe par un modèle simple et interprétable, afin d'expliquer les prédictions instance par instance.

La fonction présentée instancie un agent LimeExplainerAgent, qui utilise un LLM pour générer des explications locales des prédictions d’un modèle de classification ou de régression via les méthodes explain_classification(...) ou explain_regression(...).

```python
from explainer import LimeExplainerAgent
explainer = LimeExplainerAgent(
            llm_provider=...,
            model_name=...,
            temperature=...,
            api_key=...
        )
explainer.explain_classification(...)
#ou
explainer.explain_regression(...)
```

### Fonction pour SHAP local

SHAP (SHapley Additive exPlanations) est une méthode d’explicabilité fondée sur la théorie des jeux, qui attribue à chaque caractéristique une contribution à la prédiction d’un modèle. En mode local, SHAP explique une prédiction individuelle en calculant la contribution marginale de chaque variable pour une observation donnée.

La fonction suivante instancie un agent ShapExplainerAgent qui utilise SHAP en mode local pour fournir une explication détaillée instance par instance, via les méthodes explain_classification(...) ou explain_regression(...), en identifiant l'influence précise de chaque variable sur la prédiction du modèle.

```python
from explainer import ShapExplainerAgent
explainer = ShapExplainerAgent(
            llm_provider=...,
            model_name=...,
            temperature=...,
            api_key=...
        )
explainer.explain_classification(...)
#ou
explainer.explain_regression(...)
```

### Fonction pour SHAP Global

SHAP (SHapley Additive exPlanations) est une méthode d’explicabilité basée sur la théorie des jeux, qui mesure l'impact de chaque variable sur les prédictions d’un modèle. En mode global, SHAP agrège les explications locales sur l’ensemble des données pour fournir une vision d’ensemble de l’importance moyenne des variables.

La fonction suivante instancie un agent ShapExplainerAgentGlobal qui peut être utilisé pour produire des explications globales en analysant les contributions moyennes des variables sur un ensemble d’exemples, permettant ainsi de mieux comprendre le comportement global du modèle.

```python
from explainer import ShapExplainerAgentGlobal
explainer = ShapExplainerAgentGlobal(
            llm_provider=...,
            model_name=...,
            temperature=...,
            api_key=...
        )
explainer.explain_classification(...)
#ou
explainer.explain_regression(...)
```

## Exemple Classification LIME

```python
#installation des packages
%pip install langchain-ollama langchain-google-genai langchain-openai langchain-anthropic
%pip install lime shap
```

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
    # Import des bibliothèques nécessaires pour le machine learning
    from sklearn.datasets import load_iris # Corrected indentation
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Chargement des données Iris depuis sklearn
    iris = load_iris()
    # Conversion des données en DataFrame pandas pour faciliter la manipulation
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Conversion des labels en Series pandas
    y = pd.Series(iris.target)
    # Dictionnaire pour mapper les indices numériques aux noms des espèces
    class_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    # Division des données en ensembles d'entraînement et de test (70%/30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Création et entraînement du modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Description du contexte pour l'explication des prédictions
    context = "Jeu de données Iris avec des mesures de sépales/pétales de trois espèces d'iris."

    # Configuration des différents modèles de langage à tester
    llm_configs = [
        # Configuration pour Gemini (Google)
        {"provider": "gemini", "model": "gemini-2.0-flash", "api_key": userdata.get("GOOGLE_API_KEY")},
        # Configuration pour Ollama (commentée pour le moment)
        #{"provider": "ollama", "model": "llama3.2", "api_key": None}
    ]

    # Boucle pour tester chaque configuration de modèle de langage
    for config in llm_configs:
        print(f"\n--- Explication avec {config['provider'].upper()} ({config['model']}) ---")
        
        # Création de l'instance de l'explainer LIME avec la configuration actuelle
        explainer = LimeExplainerAgent(
            llm_provider=config["provider"],
            model_name=config["model"],
            temperature=0.2,  # Température basse pour des réponses plus déterministes
            api_key=config.get("api_key")
        )

        # Affichage des informations sur les fournisseurs disponibles (une seule fois)
        if config == llm_configs[0]:
            providers_info = explainer.get_available_providers()
            # Parcours et affichage des informations de chaque fournisseur
            for provider, info in providers_info.items():
                print(f"\n{provider.upper()} - {info['description']}")
                print(f"  Modèles recommandés: {', '.join(info['models_recommandés'])}")
                print(f"  Nécessite API key: {'Oui' if info['nécessite_api_key'] else 'Non'}")

        # Génération de l'explication pour une instance spécifique
        explainer.explain_classification(
            X_train, X_test, y_test, model, context,
            class_names_dict=class_names,  # Noms des classes pour l'affichage
            instance_index=5,  # Index de l'instance à expliquer
            num_features=4  # Nombre de features à inclure dans l'explication
        )

# Point d'entrée principal du script
if __name__ == "__main__":
    print("=== DÉMONSTRATION EN CLASSIFICATION ===")
    demo_classification_package()
```
