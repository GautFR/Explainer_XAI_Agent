import numpy as np
import pandas as pd
from IPython.display import Markdown, display
import lime
import lime.lime_tabular
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

class LimeExplainerAgent:
    """
    Agent IA qui explique les prédictions de modèles ML (classification et régression)
    à l'aide de LIME et génère un résumé en langage naturel avec LangChain et Llama 3.2.
    """

#------------------------------------------------------------------------------------------------------------------#
    def __init__(self, model_name="llama3.2"):
        """
        Initialise l'agent avec le modèle LLM spécifié.

        Parameters:
        -----------
        model_name : str, default="llama3.2"
            Nom du modèle Ollama à utiliser pour les explications
        """
        self.llm = OllamaLLM(model=model_name)

        # Template de prompt pour l'explication des résultats LIME en classification
        self.prompt_template_classification = """Tu es un expert en data science qui explique les résultats d'algorithmes d'IA à des non-spécialistes.

Contexte du jeu de données : {context}
Instance analysée : {instance_details}
Prédiction du modèle : {prediction} (probabilité: {probability:.2f})
Classe réelle : {true_class}

Voici les variables qui ont le plus influencé cette prédiction selon LIME :
{lime_features}

En tant qu'expert, explique :
1. La méthode LIME en 2-3 phrases simples pour la classification
2. Les variables les plus importantes dans cette prédiction et leur impact (positif ou négatif)
3. Une interprétation claire de chaque facteur pour un public non technique
4. Une conclusion générale sur cette prédiction

Utilise un ton pédagogique et des analogies si nécessaire. Sois précis mais accessible.
"""

        # Template de prompt pour l'explication des résultats LIME en régression
        self.prompt_template_regression = """Tu es un expert en data science qui explique les résultats d'algorithmes d'IA à des non-spécialistes.

Contexte du jeu de données : {context}
Instance analysée : {instance_details}
Valeur prédite par le modèle : {prediction} {unit}
Valeur réelle : {true_value} {unit}

Voici les variables qui ont le plus influencé cette prédiction selon LIME :
{lime_features}

En tant qu'expert, explique :
1. La méthode LIME en 2-3 phrases simples pour la régression
2. Les variables les plus importantes pour cette prédiction et comment elles ont augmenté ou diminué la valeur prédite
3. Une interprétation claire de chaque facteur pour un public non technique
4. L'écart entre la prédiction et la valeur réelle

Utilise un ton pédagogique et des analogies si nécessaire. Sois précis mais accessible.
"""
        self.prompt_classification = ChatPromptTemplate.from_template(self.prompt_template_classification)
        self.prompt_regression = ChatPromptTemplate.from_template(self.prompt_template_regression)
        self.chain_classification = self.prompt_classification | self.llm
        self.chain_regression = self.prompt_regression | self.llm

#------------------------------------------------------------------------------------------------------------------#       
    def set_model(self, model_name: str): #Corrected indentation
        """
        Change dynamiquement le modèle LLM utilisé par l'agent.
        """
        try:
            self.llm = OllamaLLM(model=model_name)
            self.chain_classification = self.prompt_classification | self.llm
            self.chain_regression = self.prompt_regression | self.llm
        except Exception as e:
            raise ValueError(f"Impossible d'utiliser le modèle '{model_name}' : {e}")

#------------------------------------------------------------------------------------------------------------------#
    def explain_classification(self, X_train, X_test, y_test, model, context, #Corrected indentation
                               class_names_dict=None, instance_index=0,
                               num_features=6, show_lime_viz=True):
        """
        Explique une prédiction de classification à l'aide de LIME puis génère un résumé en langage naturel

        Parameters:
        -----------
        X_train : pandas.DataFrame
            Données d'entraînement pour initialiser LIME
        X_test : pandas.DataFrame
            Données de test contenant l'instance à expliquer
        y_test : pandas.Series
            Étiquettes de test
        model : object
            Modèle ML à expliquer (doit avoir une méthode predict_proba)
        context : str
            Description du contexte du jeu de données
        class_names_dict : dict, optional
            Mapping entre indices de classes et noms lisibles
        instance_index : int, default=0
            Indice de l'instance à expliquer
        num_features : int, default=6
            Nombre de caractéristiques à inclure dans l'explication
        show_lime_viz : bool, default=True
            Afficher la visualisation LIME standard

        Returns:
        --------
        tuple
            (explication LIME, nom classe prédite, liste des poids, probabilités, explication textuelle)
        """
        # Définition des noms de classes pour LIME
        class_names = list(class_names_dict.values()) if class_names_dict else [str(i) for i in range(len(np.unique(y_test)))]

        # Création de l'explainer LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            class_names=class_names,
            mode='classification',
            discretize_continuous=False
        )

        # Instance à expliquer
        instance = X_test.iloc[instance_index].values

        # Classe réelle
        true_label = y_test.iloc[instance_index]
        true_class_name = class_names_dict[true_label] if class_names_dict else str(true_label)

        # Prédictions
        predicted_proba = model.predict_proba(instance.reshape(1, -1))[0]
        predicted_class_index = np.argmax(predicted_proba)
        predicted_class_label = class_names_dict[predicted_class_index] if class_names_dict else str(predicted_class_index)

        # Explication LIME
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=num_features,
            labels=[predicted_class_index]
        )

        # Extraction des poids d'explication
        explanation_weights = exp.as_list(label=predicted_class_index)

        # Affichage des détails de base
        print(f"\n=== CLASSIFICATION - Instance #{instance_index} ===")
        print("Valeurs de l'instance :", X_test.iloc[[instance_index]].to_dict('records')[0])
        print(f"\nClasse prédite : {predicted_class_label} (probabilité: {predicted_proba[predicted_class_index]:.4f})")
        print(f"Classe réelle   : {true_class_name}")

        print("\nFacteurs d'influence selon LIME :")
        for feature, weight in explanation_weights:
            impact = "positif" if weight > 0 else "négatif"
            print(f"- {feature} : {weight:.4f} (impact {impact})")

        # Affichage de la visualisation LIME standard si demandé
        if show_lime_viz:
            try:
                exp.show_in_notebook(show_all=False, predict_proba=True)
            except Exception as e:
                print(f"Visualisation non affichable : {e}")

        # Préparation de l'entrée pour le LLM
        lime_features_text = "\n".join([f"- {feature} : {weight:.4f}" for feature, weight in explanation_weights])
        instance_details = ", ".join([f"{col}: {val}" for col, val in X_test.iloc[instance_index].to_dict().items()])

        # Génération de l'explication en langage naturel avec LLM
        llm_explanation = self.chain_classification.invoke({ # Corrected: Use self.chain_classification
            "context": context,
            "instance_details": instance_details,
            "prediction": predicted_class_label,
            "probability": predicted_proba[predicted_class_index],
            "true_class": true_class_name,
            "lime_features": lime_features_text
        })

        # Affichage de l'explication LLM en markdown
        print("\n=== Explication générée par l'IA ===")
        display(Markdown(llm_explanation))

        return exp, predicted_class_label, explanation_weights, predicted_proba, llm_explanation

#------------------------------------------------------------------------------------------------------------------#
    def explain_regression(self, X_train, X_test, y_test, model, context, #Corrected indentation
                           instance_index=0, num_features=6, unit="",
                           show_lime_viz=True):
        """
        Explique une prédiction de régression à l'aide de LIME puis génère un résumé en langage naturel

        Parameters:
        -----------
        X_train : pandas.DataFrame
            Données d'entraînement pour initialiser LIME
        X_test : pandas.DataFrame
            Données de test contenant l'instance à expliquer
        y_test : pandas.Series
            Valeurs cibles réelles
        model : object
            Modèle ML à expliquer (doit avoir une méthode predict)
        context : str
            Description du contexte du jeu de données
        instance_index : int, default=0
            Indice de l'instance à expliquer
        num_features : int, default=6
            Nombre de caractéristiques à inclure dans l'explication
        unit : str, default=""
            Unité de mesure de la variable prédite (ex: "€", "kg", etc.)
        show_lime_viz : bool, default=True
            Afficher la visualisation LIME standard

        Returns:
        --------
        tuple
            (explication LIME, valeur prédite, liste des poids, explication textuelle)
        """
        # Création de l'explainer LIME pour la régression
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            mode='regression',
            discretize_continuous=False
        )

        # Instance à expliquer
        instance = X_test.iloc[instance_index].values

        # Valeur réelle
        true_value = y_test.iloc[instance_index]

        # Prédiction
        predicted_value = model.predict(instance.reshape(1, -1))[0]

        # Explication LIME
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict,
            num_features=num_features
        )

        # Extraction des poids d'explication
        explanation_weights = exp.as_list()

        # Affichage des détails de base
        print(f"\n=== RÉGRESSION - Instance #{instance_index} ===")
        print("Valeurs de l'instance :", X_test.iloc[[instance_index]].to_dict('records')[0])
        print(f"\nValeur prédite : {predicted_value:.4f} {unit}")
        print(f"Valeur réelle  : {true_value:.4f} {unit}")
        print(f"Erreur absolue : {abs(predicted_value - true_value):.4f} {unit}")
        print(f"Erreur relative : {abs((predicted_value - true_value) / true_value) * 100:.2f}%")

        print("\nFacteurs d'influence selon LIME :")
        for feature, weight in explanation_weights:
            impact = "positif" if weight > 0 else "négatif"
            print(f"- {feature} : {weight:.4f} (impact {impact})")

        # Affichage de la visualisation LIME standard si demandé
        if show_lime_viz:
            try:
                exp.show_in_notebook(show_all=False)
            except Exception as e:
                print(f"Visualisation non affichable : {e}")

        # Préparation de l'entrée pour le LLM
        lime_features_text = "\n".join([f"- {feature} : {weight:.4f}" for feature, weight in explanation_weights])
        instance_details = ", ".join([f"{col}: {val}" for col, val in X_test.iloc[instance_index].to_dict().items()])

        # Génération de l'explication en langage naturel avec LLM
        llm_explanation = self.chain_regression.invoke({ # Corrected: Use self.chain_regression
            "context": context,
            "instance_details": instance_details,
            "prediction": f"{predicted_value:.4f}",
            "true_value": f"{true_value:.4f}",
            "unit": unit,
            "lime_features": lime_features_text
        })

        # Affichage de l'explication LLM en markdown
        print("\n=== Explication générée par l'IA ===")
        display(Markdown(llm_explanation))

        return exp, predicted_value, explanation_weights, llm_explanation