# Explainer_XAI_Agent

Ce package fournit un agent IA LIME et un agent SHAP pour expliquer les prédictions de modèles de Machine Learning en langage naturel.

## Installation

```bash
pip install git https://github.com/GautFR/Explainer_XAI_Agent.git
```

## Utilisation

```python
from explainer import LimeExplainerAgent
agent = LimeExplainerAgent()
agent.explain_classification(...)
```

```python
from explainer import ShapExplainerAgent
agent = ShapExplainerAgent()
agent.explain_prediction(...)
```

## Exemple Classification LIME

```python
from explainer import LimeExplainerAgent
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def demo_classification_package():
    """Démontre l'utilisation de l'agent en classification avec le jeu de données Iris"""
    from sklearn.datasets import load_iris # Corrected indentation
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Chargement des données
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    # Mapping des classes
    class_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Entraînement d'un modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #Utilisation du package perso
    context = "Il s'agit du jeu de données Iris qui contient des mesures de pétales et de sépales de trois espèces différentes d'iris."
    
    # Call the method using an instance of the class
    agent = LimeExplainerAgent() # Create an instance
    agent.explain_classification( # Call method on the instance
        X_train, X_test, y_test, model, context,
        class_names_dict=class_names, instance_index=5, num_features=4
    )

if __name__ == "__main__":
    print("=== DÉMONSTRATION EN CLASSIFICATION ===")
    demo_classification_package()
```
