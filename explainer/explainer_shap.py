import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ShapExplainerAgent:
    """
    Agent IA qui explique les prédictions de modèles ML à l'aide de SHAP et génère
    un résumé en langage naturel avec LangChain et Llama 3.2 par défaut.
    Compatible avec les modèles de classification et de régression.
    """

    def __init__(self, model_name="llama3.2"):
        """
        Initialise l'agent avec le modèle LLM spécifié.

        Parameters:
        -----------
        model_name : str, default="llama3.2"
            Nom du modèle Ollama à utiliser pour les explications
        """
        self.llm = OllamaLLM(model=model_name)

        # Template de prompt pour l'explication des résultats SHAP (classification)
        self.prompt_template_classification = """Tu es un expert en data science qui explique les résultats d'algorithmes d'IA à des non-spécialistes.

Contexte du jeu de données : {context}
Instance analysée : {instance_details}
Prédiction du modèle : {prediction} (probabilité: {probability:.2f})
Classe réelle : {true_class}

Voici les variables qui ont le plus influencé cette prédiction selon SHAP (valeurs positives = contribution vers la classe prédite, négatives = contribution contre) :
{shap_features}

En tant qu'expert, explique :
1. La méthode SHAP en 2-3 phrases simples
2. Les variables les plus importantes dans cette prédiction et leur impact (positif ou négatif)
3. Une interprétation claire de chaque facteur influent pour un public non technique
4. Une conclusion générale sur cette prédiction (est-elle fiable ? pourquoi ?)

Utilise un ton pédagogique et des analogies si nécessaire. Sois précis mais accessible.
"""
        # Template de prompt pour l'explication des résultats SHAP (régression)
        self.prompt_template_regression = """Tu es un expert en data science qui explique les résultats d'algorithmes d'IA à des non-spécialistes.

Contexte du jeu de données : {context}
Instance analysée : {instance_details}
Valeur prédite par le modèle : {prediction:.2f} {unit}
Valeur réelle : {true_value:.2f} {unit}
Erreur absolue : {error:.2f}

Voici les variables qui ont le plus influencé cette prédiction selon SHAP (valeurs positives = augmentation de la valeur prédite, négatives = diminution) :
{shap_features}

En tant qu'expert, explique :
1. La méthode SHAP en 2-3 phrases simples
2. Les variables les plus importantes dans cette prédiction et leur impact (positif ou négatif)
3. Une interprétation claire de chaque facteur influent pour un public non technique
4. Une conclusion générale sur cette prédiction (est-elle fiable ? pourquoi ?)

Utilise un ton pédagogique et des analogies si nécessaire. Sois précis mais accessible.
"""
        self.prompt_classification = ChatPromptTemplate.from_template(self.prompt_template_classification)
        self.prompt_regression = ChatPromptTemplate.from_template(self.prompt_template_regression)
        self.chain_classification = self.prompt_classification | self.llm
        self.chain_regression = self.prompt_regression | self.llm

#------------------------------------------------------------------------------------------------------------------#       
    def set_model(self, model_name: str):
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
    def explain_classification(self, X_test, y_test, model, context,
                               class_names_dict=None, instance_index=0,
                               num_features=None, show_shap_viz=True):
        """
        Explique une prédiction de classification à l'aide de SHAP puis génère un résumé en langage naturel

        Parameters:
        -----------
        model : object
            Modèle de classification entraîné (comme RandomForest, XGBoost, etc.)
        X_test : pandas.DataFrame
            DataFrame contenant les données de test
        y_test : pandas.Series
            Série pandas contenant les classes réelles
        context : str
            Description du contexte du jeu de données
        class_names_dict : dict, optional
            Dictionnaire pour associer des indices de classe à des noms lisibles
        instance_index : int, default=0
            Index de l'observation à analyser
        num_features : int, optional
            Nombre de caractéristiques à inclure dans l'explication (par défaut, toutes sont incluses)
        show_shap_viz : bool, default=True
            Afficher la visualisation SHAP waterfall standard
      
        Returns:
        --------
        tuple
            (valeurs SHAP, classe prédite, probabilités, explication textuelle)
        """
        # Extraction de l'observation cible
        instance = X_test.iloc[instance_index].values.reshape(1, -1)
        feature_names = X_test.columns

        # Prédiction pour l'observation
        proba = model.predict_proba(instance)[0]
        pred_class_index = np.argmax(proba)
        pred_class_label = model.predict(instance)[0]
        true_label = y_test.iloc[instance_index]

        # Noms de classes lisibles
        nom_classe_predite = class_names_dict[pred_class_label] if class_names_dict else str(pred_class_label)
        nom_classe_reelle = class_names_dict[true_label] if class_names_dict else str(true_label)

        # Affichage des valeurs de l'observation
        print(f"Instance #{instance_index}")
        print("Valeurs de l'observation :")
        print(pd.DataFrame(X_test.iloc[instance_index]).T)

        # Affichage des probabilités prédites
        print("\nProbabilités prédites pour chaque classe :")
        for i, p in enumerate(proba):
            nom = class_names_dict[i] if class_names_dict else f"Classe {i}"
            print(f"{nom} : {p:.4f}")

        # Affichage des classes prédite et réelle
        print(f"\nClasse prédite : {nom_classe_predite} (probabilité: {proba[pred_class_index]:.4f})")
        print(f"Classe réelle  : {nom_classe_reelle}")

        # Création de l'explainer SHAP
        explainer_shap = shap.Explainer(model)
        shap_values = explainer_shap.shap_values(X_test)

        # Extraction des valeurs SHAP spécifiques à l'instance
        values = shap_values[instance_index] if len(shap_values[0].shape) == 2 else shap_values

        # Construction de l'objet SHAP Explanation
        if hasattr(values, 'shape') and len(values.shape) > 1:
            # Cas multi-classe
            explanation = shap.Explanation(
                values=values[:, pred_class_index],
                base_values=explainer_shap.expected_value[pred_class_index],
                data=X_test.iloc[instance_index].values,
                feature_names=feature_names
            )
            shap_vals = values[:, pred_class_index]
        else:
            # Cas binaire
            explanation = shap.Explanation(
                values=values,
                base_values=explainer_shap.expected_value,
                data=X_test.iloc[instance_index].values,
                feature_names=feature_names
            )
            shap_vals = values

        # Affichage du graphique waterfall si demandé
        if show_shap_viz:
            shap.plots.waterfall(explanation, max_display = (num_features + 1))

        # Création du DataFrame des valeurs SHAP
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_vals
        }).sort_values(by='shap_value', key=abs, ascending=False)

        # Limiter le nombre de features si num_features est spécifié
        if num_features is not None:
            shap_df = shap_df.head(num_features)

        print("\nValeurs SHAP par ordre d'importance :")
        for _, row in shap_df.iterrows():
            impact = "positif" if row['shap_value'] > 0 else "négatif"
            print(f"- {row['feature']} : {row['shap_value']:.4f} (impact {impact})")

        # Préparation des entrées pour le LLM
        shap_features_text = "\n".join([
            f"- {row['feature']} : {row['shap_value']:.4f}"
            for _, row in shap_df.iterrows()
        ])
        instance_details = ", ".join([
            f"{col}: {val}" for col, val in X_test.iloc[instance_index].to_dict().items()
        ])

        # Génération de l'explication en langage naturel
        llm_explanation = self.chain_classification.invoke({
            "context": context,
            "instance_details": instance_details,
            "prediction": nom_classe_predite,
            "probability": proba[pred_class_index],
            "true_class": nom_classe_reelle,
            "shap_features": shap_features_text
        })

        # Affichage de l'explication en markdown
        print("\n=== Explication générée par l'IA ===")
        display(Markdown(llm_explanation))

        # Retourne les informations importantes
        shap_feature_values = list(zip(shap_df['shap_value'], shap_df['feature']))
        return shap_feature_values, nom_classe_predite, proba, llm_explanation

#------------------------------------------------------------------------------------------------------------------#
    def explain_regression(self, X_test, y_test, model,
                           context, instance_index=0, num_features=None,
                           unit="", show_shap_viz=True):
        """
        Explique une prédiction de régression à l'aide de SHAP puis génère un résumé en langage naturel

        Parameters:
        -----------
        model : object
            Modèle de régression entraîné (comme RandomForest, XGBoost, etc.)
        X_test : pandas.DataFrame
            DataFrame contenant les données de test
        y_test : pandas.Series
            Série pandas contenant les valeurs réelles
        context : str
            Description du contexte du jeu de données
        instance_index : int, default=0
            Index de l'observation à analyser
        num_features : int, optional
            Nombre de caractéristiques à inclure dans l'explication (par défaut, toutes sont incluses)
        unit : str, default=""
            Unité de mesure de la variable prédite (ex: "€", "kg", etc.)
        show_shap_viz : bool, default=True
            Afficher la visualisation SHAP waterfall standard

        Returns:
        --------
        tuple
            (valeurs SHAP, valeur prédite, valeur réelle, erreur, explication textuelle)
        """
        # Extraction de l'observation cible
        instance = X_test.iloc[instance_index].values.reshape(1, -1)
        feature_names = X_test.columns

        # Prédiction pour l'observation
        predicted_value = model.predict(instance)[0]
        true_value = y_test.iloc[instance_index]
        error = abs(predicted_value - true_value)

        # Affichage des valeurs de l'observation
        print(f"Instance #{instance_index}")
        print("Valeurs de l'observation :")
        print(pd.DataFrame(X_test.iloc[instance_index]).T)

        # Affichage des valeurs prédite et réelle
        print(f"\nValeur prédite : {predicted_value:.4f} {unit}")
        print(f"Valeur réelle  : {true_value:.4f} {unit}")
        print(f"Erreur absolue : {error:.4f} {unit}")

        # Création de l'explainer SHAP
        explainer_shap = shap.Explainer(model)
        shap_values = explainer_shap.shap_values(X_test)

        # Extraction des valeurs SHAP spécifiques à l'instance
        instance_shap_values = shap_values[instance_index]

        # Construction de l'objet SHAP Explanation pour l'instance
        explanation = shap.Explanation(
            values=instance_shap_values,
            base_values=explainer_shap.expected_value,
            data=X_test.iloc[instance_index].values,
            feature_names=feature_names
        )

        # Affichage du graphique waterfall si demandé
        if show_shap_viz:
            shap.plots.waterfall(explanation, max_display = (num_features + 1))

        # Création du DataFrame des valeurs SHAP
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': instance_shap_values
        }).sort_values(by='shap_value', key=abs, ascending=False)

        # Limiter le nombre de features si num_features est spécifié
        if num_features is not None:
            shap_df = shap_df.head(num_features)

        print("\nValeurs SHAP par ordre d'importance :")
        for _, row in shap_df.iterrows():
            impact = "positif" if row['shap_value'] > 0 else "négatif"
            print(f"- {row['feature']} : {row['shap_value']:.4f} (impact {impact})")

        # Préparation des entrées pour le LLM
        shap_features_text = "\n".join([
            f"- {row['feature']} : {row['shap_value']:.4f}"
            for _, row in shap_df.iterrows()
        ])
        instance_details = ", ".join([
            f"{col}: {val}" for col, val in X_test.iloc[instance_index].to_dict().items()
        ])

        # Génération de l'explication en langage naturel
        llm_explanation = self.chain_regression.invoke({
            "context": context,
            "instance_details": instance_details,
            "prediction": predicted_value,
            "true_value": true_value,
            "error": error,
            "unit": unit,
            "shap_features": shap_features_text
        })

        # Affichage de l'explication en markdown
        print("\n=== Explication générée par l'IA ===")
        display(Markdown(llm_explanation))

        # Retourne les informations importantes
        shap_feature_values = list(zip(shap_df['shap_value'], shap_df['feature']))
        return shap_feature_values, predicted_value, true_value, error, llm_explanation