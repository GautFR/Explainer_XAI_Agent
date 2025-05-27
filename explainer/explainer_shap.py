import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal, Optional, Union, Dict, Any, List, Tuple

# Importations pour différentes interfaces LLM
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM


class ShapExplainerAgent:
    """
    Agent IA qui explique les prédictions de modèles ML à l'aide de SHAP et génère
    un résumé en langage naturel avec différents LLMs.
    Compatible avec les modèles de classification et de régression.
    """

#-----------------------------------------------------------------------------------------------------------------#

    def __init__(self,
                 llm_provider: Literal["ollama", "gemini", "openai", "anthropic"] = "ollama",
                 model_name: str = "llama3.2",
                 api_key: Optional[str] = None,
                 **provider_kwargs):
        """
        Initialise l'agent avec le modèle LLM spécifié, local ou API.

        Parameters:
        -----------
        llm_provider : str, default="ollama"
            Fournisseur de LLM à utiliser ("ollama", "gemini", "openai", "anthropic")
        model_name : str, default="llama3.2"
            Nom du modèle à utiliser (dépend du fournisseur)
        api_key : str, optional
            Clé API nécessaire pour les fournisseurs distants (Gemini, OpenAI, Anthropic)
        provider_kwargs : dict
            Paramètres supplémentaires spécifiques au fournisseur
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = api_key
        self.provider_kwargs = provider_kwargs

        # Initialiser le LLM en fonction du fournisseur choisi
        self.llm = self._initialize_llm()

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
Erreur absolue : {error:.2f} {unit}

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

#-----------------------------------------------------------------------------------------------------------------#

    def _initialize_llm(self) -> Union[BaseChatModel, BaseLLM]:
        """
        Initialise le modèle LLM en fonction du fournisseur spécifié.

        Returns:
        --------
        Union[BaseChatModel, BaseLLM]
            Instance du modèle LLM initialisé
        """
        if self.llm_provider == "ollama":
            return OllamaLLM(model=self.model_name, **self.provider_kwargs)

        elif self.llm_provider == "gemini":
            if not self.api_key:
                raise ValueError("Une clé API est requise pour utiliser Gemini")
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                **self.provider_kwargs
            )

        elif self.llm_provider == "openai":
            if not self.api_key:
                raise ValueError("Une clé API est requise pour utiliser OpenAI")
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                **self.provider_kwargs
            )

        elif self.llm_provider == "anthropic":
            if not self.api_key:
                raise ValueError("Une clé API est requise pour utiliser Anthropic")
            return ChatAnthropic(
                model=self.model_name,
                anthropic_api_key=self.api_key,
                **self.provider_kwargs
            )

        else:
            raise ValueError(f"Fournisseur LLM non pris en charge: {self.llm_provider}")

#-----------------------------------------------------------------------------------------------------------------#

    def set_model(self,
                  llm_provider: Optional[str] = None,
                  model_name: Optional[str] = None,
                  api_key: Optional[str] = None,
                  **provider_kwargs):
        """
        Change dynamiquement le fournisseur et/ou le modèle LLM utilisé par l'agent.

        Parameters:
        -----------
        llm_provider : str, optional
            Fournisseur de LLM à utiliser ("ollama", "gemini", "openai", "anthropic")
        model_name : str, optional
            Nom du modèle à utiliser (dépend du fournisseur)
        api_key : str, optional
            Clé API nécessaire pour les fournisseurs distants
        provider_kwargs : dict
            Paramètres supplémentaires spécifiques au fournisseur
        """
        # Mettre à jour uniquement les paramètres fournis
        if llm_provider:
            self.llm_provider = llm_provider
        if model_name:
            self.model_name = model_name
        if api_key:
            self.api_key = api_key
        if provider_kwargs:
            self.provider_kwargs.update(provider_kwargs)

        try:
            # Réinitialiser le LLM avec les nouveaux paramètres
            self.llm = self._initialize_llm()
            self.chain_classification = self.prompt_classification | self.llm
            self.chain_regression = self.prompt_regression | self.llm
            print(f"Modèle changé avec succès pour {self.llm_provider} / {self.model_name}")
        except Exception as e:
            raise ValueError(f"Impossible d'utiliser le modèle {self.llm_provider}/{self.model_name} : {e}")

#-----------------------------------------------------------------------------------------------------------------#

    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Renvoie la liste des fournisseurs LLM disponibles et leurs modèles recommandés.

        Returns:
        --------
        Dict
            Dictionnaire des fournisseurs avec leurs modèles recommandés et les exigences
        """
        return {
            "ollama": {
                "description": "Serveur LLM local",
                "models_recommandés": ["llama3.2", "mistral", "gemma", "phi3"],
                "nécessite_api_key": False,
                "paramètres_optionnels": {
                    "temperature": "Contrôle la créativité des réponses (0.0-1.0)",
                    "base_url": "URL personnalisée si différente de celle par défaut"
                }
            },
            "gemini": {
                "description": "API Google Gemini",
                "models_recommandés": ["gemini-1.5-pro", "gemini-1.5-flash","gemini-2.0-flash"],
                "nécessite_api_key": True,
                "paramètres_optionnels": {
                    "temperature": "Contrôle la créativité des réponses (0.0-1.0)",
                    "top_p": "Contrôle la diversité des tokens générés (0.0-1.0)"
                }
            },
            "openai": {
                "description": "API OpenAI",
                "models_recommandés": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "nécessite_api_key": True,
                "paramètres_optionnels": {
                    "temperature": "Contrôle la créativité des réponses (0.0-1.0)",
                    "max_tokens": "Limite la longueur de la réponse"
                }
            },
            "anthropic": {
                "description": "API Anthropic",
                "models_recommandés": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "nécessite_api_key": True,
                "paramètres_optionnels": {
                    "temperature": "Contrôle la créativité des réponses (0.0-1.0)",
                    "max_tokens_to_sample": "Limite la longueur de la réponse"
                }
            }
        }

#-----------------------------------------------------------------------------------------------------------------#

    def explain_classification(self, X_test, y_test, model, context,
                               class_names_dict=None, instance_index=0,
                               num_features=None, show_shap_viz=True) -> Tuple:
        """
        Explique une prédiction de classification à l'aide de SHAP puis génère un résumé en langage naturel

        Parameters:
        -----------
        X_test : pandas.DataFrame
            DataFrame contenant les données de test
        y_test : pandas.Series
            Série pandas contenant les classes réelles
        model : object
            Modèle de classification entraîné (comme RandomForest, XGBoost, etc.)
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
        Tuple
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

        # Extraction des valeurs SHAP spécifiques à l'instance et gestion des différentes structures
        if isinstance(shap_values, list):
            # Cas multi-classe où shap_values est une liste de tableaux
            if len(shap_values) > 1:
                # Pour les modèles multiclasses (comme Random Forest)
                values = np.array(shap_values)[pred_class_index][instance_index]
                base_value = explainer_shap.expected_value[pred_class_index]
            else:
                # Pour les modèles binaires
                values = shap_values[0][instance_index]
                base_value = explainer_shap.expected_value
        else:
            # Cas où shap_values est un tableau unique (SHAP TreeExplainer avec certains modèles)
            values = shap_values[instance_index]
            base_value = explainer_shap.expected_value

            # Vérifier si values est multidimensionnel (traiter le cas multiclasse)
            if len(values.shape) > 1:
                values = values[:, pred_class_index]
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[pred_class_index]

        # Construction de l'objet SHAP Explanation
        try:
            explanation = shap.Explanation(
                values=values,
                base_values=base_value,
                data=X_test.iloc[instance_index].values,
                feature_names=feature_names
            )

            # Affichage du graphique waterfall si demandé
            if show_shap_viz:
                plt.figure(figsize=(10, 6))
                shap.plots.waterfall(explanation, max_display=(num_features or len(feature_names)))
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Impossible de créer la visualisation SHAP : {e}")
            print("Utilisation d'un format alternatif pour l'explication SHAP")
            # Si la création de l'explication SHAP échoue, on continue avec les valeurs brutes

        # Création du DataFrame des valeurs SHAP
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': values if isinstance(values, np.ndarray) else np.array(values)
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
        print(f"\n=== Explication générée par l'IA (via {self.llm_provider}/{self.model_name}) ===")

        # Extraction du contenu depuis la réponse du LLM (gestion des différents formats)
        if hasattr(llm_explanation, 'content'):
            explanation_text = llm_explanation.content
        elif isinstance(llm_explanation, dict) and 'content' in llm_explanation:
            explanation_text = llm_explanation['content']
        elif isinstance(llm_explanation, str):
            explanation_text = llm_explanation
        else:
            explanation_text = str(llm_explanation)

        display(Markdown(explanation_text))

        # Retourne les informations importantes
        shap_feature_values = list(zip(shap_df['shap_value'], shap_df['feature']))
        return shap_feature_values, nom_classe_predite, proba, explanation_text

#-----------------------------------------------------------------------------------------------------------------#

    def explain_regression(self, X_test, y_test, model,
                           context, instance_index=0, num_features=None,
                           unit="", show_shap_viz=True) -> Tuple:
        """
        Explique une prédiction de régression à l'aide de SHAP puis génère un résumé en langage naturel

        Parameters:
        -----------
        X_test : pandas.DataFrame
            DataFrame contenant les données de test
        y_test : pandas.Series
            Série pandas contenant les valeurs réelles
        model : object
            Modèle de régression entraîné (comme RandomForest, XGBoost, etc.)
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
        Tuple
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
        print(f"Erreur relative : {abs((predicted_value - true_value) / (true_value if true_value != 0 else 1)) * 100:.2f}%")

        # Création de l'explainer SHAP
        try:
            explainer_shap = shap.Explainer(model)
            shap_values = explainer_shap.shap_values(X_test)

            # Gestion des différents formats possibles de valeurs SHAP
            if isinstance(shap_values, list):
                # Certains modèles retournent une liste pour les modèles de régression
                instance_shap_values = shap_values[instance_index]
            else:
                # Format standard pour la plupart des modèles de régression
                instance_shap_values = shap_values[instance_index]

            # Construction de l'objet SHAP Explanation pour l'instance
            if hasattr(explainer_shap, 'expected_value'):
                base_value = explainer_shap.expected_value
                if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
                    base_value = base_value[0]  # Prendre la première valeur si c'est une liste/array
            else:
                # Si expected_value n'est pas disponible, utiliser la moyenne des valeurs prédites
                base_value = model.predict(X_test).mean()

            explanation = shap.Explanation(
                values=instance_shap_values,
                base_values=base_value,
                data=X_test.iloc[instance_index].values,
                feature_names=feature_names
            )

            # Affichage du graphique waterfall si demandé
            if show_shap_viz:
                plt.figure(figsize=(10, 6))
                shap.plots.waterfall(explanation, max_display=(num_features or len(feature_names)))
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Impossible de créer l'explication SHAP : {e}")
            # Créer un explainer alternatif si le premier a échoué
            try:
                explainer_shap = shap.TreeExplainer(model)
                shap_values = explainer_shap.shap_values(X_test)
                instance_shap_values = shap_values[instance_index]
                print("Utilisation de TreeExplainer comme alternative.")
            except Exception as e2:
                print(f"Échec de l'alternative : {e2}")
                # Créer des valeurs SHAP fictives basées sur les coefficients du modèle si disponibles
                if hasattr(model, 'coef_'):
                    print("Utilisation des coefficients du modèle comme approximation des valeurs SHAP.")
                    instance_shap_values = model.coef_ * X_test.iloc[instance_index].values
                else:
                    print("Impossible de calculer les valeurs SHAP. Utilisation de valeurs aléatoires.")
                    instance_shap_values = np.random.randn(len(feature_names))

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
        print(f"\n=== Explication générée par l'IA (via {self.llm_provider}/{self.model_name}) ===")

        # Extraction du contenu depuis la réponse du LLM (gestion des différents formats)
        if hasattr(llm_explanation, 'content'):
            explanation_text = llm_explanation.content
        elif isinstance(llm_explanation, dict) and 'content' in llm_explanation:
            explanation_text = llm_explanation['content']
        elif isinstance(llm_explanation, str):
            explanation_text = llm_explanation
        else:
            explanation_text = str(llm_explanation)

        display(Markdown(explanation_text))

        # Retourne les informations importantes
        shap_feature_values = list(zip(shap_df['shap_value'], shap_df['feature']))
        return shap_feature_values, predicted_value, true_value, error, explanation_text