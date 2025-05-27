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


class ShapExplainerGlobalAgent:
    """
    Agent IA pour expliquer les graphiques globaux SHAP (summary_plot) de manière pédagogique.
    Génère une explication en langage naturel des principales variables influentes.
    Compatible avec différents types de modèles ML et différents fournisseurs de LLM.
    """

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

        # Template pour les modèles de régression
        self.prompt_template_regression = """Tu es un expert en data science. Tu dois expliquer à un public non technique les résultats globaux d'un modèle d'IA de régression, en particulier à l'aide du graphe SHAP summary_plot.

Contexte : {context}

Voici les principales variables influentes dans le modèle de régression, classées par importance décroissante (valeurs SHAP moyennes absolues) :
{shap_summary}

1. Explique brièvement ce que montre un summary_plot SHAP pour un modèle de régression.
2. Décris l'importance des variables et comment elles influencent la prédiction de la valeur cible (positivement ou négativement).
3. Interprète de manière claire et pédagogique les 3 variables les plus influentes en précisant leur impact sur la valeur continue prédite.
4. Donne une conclusion générale sur les tendances du modèle de régression.

Sois précis, pédagogique, et utilise des analogies simples si besoin."""

        # Template pour les modèles de classification
        self.prompt_template_classification = """Tu es un expert en data science. Tu dois expliquer à un public non technique les résultats globaux d'un modèle d'IA de classification, en particulier à l'aide du graphe SHAP summary_plot.

Contexte : {context}

Voici les principales variables influentes dans le modèle de classification pour la première classe, classées par importance décroissante (valeurs SHAP moyennes absolues) :
{shap_summary}

1. Explique brièvement ce que montre un summary_plot SHAP pour un modèle de classification, spécifiquement pour la première classe.
2. Décris l'importance des variables et comment elles influencent la classification vers cette classe (positivement ou négativement).
3. Interprète de manière claire et pédagogique les 3 variables les plus influentes en précisant leur impact sur la probabilité d'appartenance à cette classe.
4. Donne une conclusion générale sur les tendances du modèle pour cette classe spécifique.

Sois précis, pédagogique, et utilise des analogies simples si besoin."""

        # Initialiser les chaînes de prompt
        self.prompt_regression = ChatPromptTemplate.from_template(self.prompt_template_regression)
        self.prompt_classification = ChatPromptTemplate.from_template(self.prompt_template_classification)

        # Initialiser les chaînes de traitement
        self.chain_regression = self.prompt_regression | self.llm
        self.chain_classification = self.prompt_classification | self.llm

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
            self.chain_regression = self.prompt_regression | self.llm
            self.chain_classification = self.prompt_classification | self.llm
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

    def _prepare_shap_explainer(self, X_train, model, model_type="auto"):
        """
        Méthode privée pour initialiser l'explainer SHAP approprié.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Données d'entraînement
        model : object
            Modèle de ML à expliquer
        model_type : str
            Type d'explainer à utiliser ("auto", "tree", "linear", "deep", "kernel")

        Returns:
        --------
        shap.Explainer
            Instance de l'explainer SHAP initialisé
        """
        try:
            if model_type == "auto":
                explainer = shap.Explainer(model, X_train)
            elif model_type == "tree":
                explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, X_train)
            elif model_type == "deep":
                explainer = shap.DeepExplainer(model, X_train.values)
            elif model_type == "kernel":
                predict_function = model.predict if hasattr(model, 'predict') else model
                explainer = shap.KernelExplainer(predict_function, X_train)
            else:
                raise ValueError("Type de modèle non reconnu.")
        except Exception as e:
            print(f"Erreur initiale: {e}. Fallback sur KernelExplainer.")
            predict_function = model.predict if hasattr(model, 'predict') else model
            explainer = shap.KernelExplainer(predict_function, X_train)

        return explainer

#-----------------------------------------------------------------------------------------------------------------#

    def _compute_shap_values(self, explainer, X_train):
        """
        Méthode privée pour calculer les valeurs SHAP.

        Parameters:
        -----------
        explainer : shap.Explainer
            L'explainer SHAP initialisé
        X_train : pd.DataFrame
            Données pour lesquelles calculer les valeurs SHAP

        Returns:
        --------
        tuple
            (shap_values_for_plot, shap_values_for_calc)
        """
        try:
            shap_values = explainer(X_train)
            # Vérifier si c'est un objet Explanation (nouvelle API SHAP)
            if isinstance(shap_values, shap.Explanation):
                shap_values_for_plot = shap_values
                shap_values_for_calc = shap_values.values
            else:
                # Ancienne API SHAP (liste de numpy arrays)
                shap_values_for_plot = shap_values
                # Pour les modèles à sortie multiple (classification multiclasse), on prend la moyenne
                if isinstance(shap_values, list):
                    shap_values_for_calc = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                else:
                    shap_values_for_calc = shap_values
        except Exception as e:
            print(f"Erreur lors du calcul des valeurs SHAP: {e}")
            print("Essai avec KernelExplainer...")
            # Fallback sur KernelExplainer
            predict_function = model.predict if hasattr(model, 'predict') else model
            explainer = shap.KernelExplainer(predict_function, X_train)
            shap_values = explainer(X_train)
            shap_values_for_plot = shap_values
            shap_values_for_calc = shap_values

        return shap_values_for_plot, shap_values_for_calc

#-----------------------------------------------------------------------------------------------------------------#

    def _create_summary_df(self, X_train, shap_values_for_calc, num_features, class_index=0):
        """
        Méthode privée pour créer un DataFrame résumant les valeurs SHAP.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Données d'entraînement avec les noms des colonnes
        shap_values_for_calc : numpy.ndarray ou shap.Explanation
            Valeurs SHAP calculées
        num_features : int
            Nombre de features à inclure dans le résumé
        class_index : int, default=0
            Index de la classe à utiliser pour la classification (0 = première classe)

        Returns:
        --------
        pd.DataFrame
            DataFrame résumant les valeurs SHAP moyennes absolues
        """
        try:
            # Gestion de différentes structures possibles de valeurs SHAP
            if isinstance(shap_values_for_calc, np.ndarray):
                if len(shap_values_for_calc.shape) == 3:  # Cas multiclasse (échantillons, classes, features)
                    # Utiliser seulement la première classe
                    mean_abs_shap = np.abs(shap_values_for_calc[:, :, class_index]).mean(axis=0)
                else:  # Cas binaire ou régression (échantillons, features)
                    mean_abs_shap = np.abs(shap_values_for_calc).mean(axis=0)
            elif isinstance(shap_values_for_calc, list):
                # Cas où shap_values est une liste de arrays (ancienne API)
                if len(shap_values_for_calc) > class_index:
                    mean_abs_shap = np.abs(shap_values_for_calc[class_index]).mean(axis=0)
                else:
                    mean_abs_shap = np.abs(shap_values_for_calc[0]).mean(axis=0)
            else:
                # Si c'est un objet Explanation
                if hasattr(shap_values_for_calc, 'values'):
                    if len(shap_values_for_calc.values.shape) > 2:  # Multiclasse
                        mean_abs_shap = np.abs(shap_values_for_calc.values[:, :, class_index]).mean(axis=0)
                    else:  # Binaire ou régression
                        mean_abs_shap = np.abs(shap_values_for_calc.values).mean(axis=0)
                else:
                    mean_abs_shap = np.abs(shap_values_for_calc).mean(axis=0)

            # Création du DataFrame de résumé
            summary_df = pd.DataFrame({
                'feature': X_train.columns,
                'mean_abs_shap': mean_abs_shap
            }).sort_values(by='mean_abs_shap', ascending=False).head(num_features)
        except Exception as e:
            print(f"Erreur lors du calcul des moyennes SHAP: {e}")
            # Fallback en cas d'erreur (structure particulière des valeurs SHAP)
            try:
                temp_values = shap_values_for_calc.values if hasattr(shap_values_for_calc, 'values') else shap_values_for_calc
                if isinstance(temp_values, list):
                    # Si c'est une liste (multiclasse), prendre la première classe
                    mean_abs_shap = np.abs(temp_values[class_index]).mean(axis=0)
                elif len(temp_values.shape) > 2:
                    # Si c'est un array 3D (multiclasse)
                    mean_abs_shap = np.abs(temp_values[:, :, class_index]).mean(axis=0)
                else:
                    # Cas standard
                    mean_abs_shap = np.abs(temp_values).mean(axis=0)

                summary_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'mean_abs_shap': mean_abs_shap
                }).sort_values(by='mean_abs_shap', ascending=False).head(num_features)
            except Exception as e2:
                print(f"Erreur secondaire: {e2}")
                # En dernier recours, créer un DataFrame vide
                summary_df = pd.DataFrame(columns=['feature', 'mean_abs_shap'])

        return summary_df

#-----------------------------------------------------------------------------------------------------------------#

    def _get_num_classes(self, shap_values):
        """
        Détermine le nombre de classes à partir des valeurs SHAP.

        Parameters:
        -----------
        shap_values : shap.Explanation ou list
            Valeurs SHAP calculées

        Returns:
        --------
        int
            Nombre de classes détecté
        """
        try:
            if isinstance(shap_values, list):
                # Cas de l'ancienne API SHAP (liste d'arrays - une par classe)
                return len(shap_values)
            elif hasattr(shap_values, 'values'):
                # Nouvelle API SHAP - objet Explanation
                if len(shap_values.values.shape) > 2:
                    # Cas multiclasse - format typique (n_samples, n_features, n_classes)
                    return shap_values.values.shape[2]
                else:
                    # Cas binaire ou régression - simplement 1 classe ou valeur
                    return 1
            else:
                # Cas par défaut
                return 1
        except Exception as e:
            print(f"Erreur lors de la détermination du nombre de classes: {e}")
            return 1

#-----------------------------------------------------------------------------------------------------------------#

    def _display_class_specific_summary_plot(self, shap_values_for_plot, X_train, class_index, num_features):
        """
        Affiche un summary plot SHAP pour une classe spécifique.

        Parameters:
        -----------
        shap_values_for_plot : shap.Explanation ou list
            Valeurs SHAP calculées
        X_train : pd.DataFrame
            Données d'entraînement
        class_index : int
            Index de la classe à visualiser
        num_features : int
            Nombre de variables à afficher
        """
        try:
            # Déterminer le type de valeurs SHAP pour adapter l'affichage
            if isinstance(shap_values_for_plot, list):
                # Ancienne API SHAP - liste d'arrays (une par classe)
                if len(shap_values_for_plot) > class_index:
                    shap.summary_plot(shap_values_for_plot[class_index], X_train, max_display=num_features)
                else:
                    print(f"Classe {class_index} non disponible, affichage pour la classe 0:")
                    shap.summary_plot(shap_values_for_plot[0], X_train, max_display=num_features)
            elif hasattr(shap_values_for_plot, 'values'):
                # Nouvelle API SHAP - objet Explanation
                if len(shap_values_for_plot.values.shape) > 2:
                    # Multiclasse - format (n_samples, n_features, n_classes)
                    class_values = shap_values_for_plot.values[:, :, class_index]

                    # Gérer les base_values qui peuvent avoir différentes formes
                    if hasattr(shap_values_for_plot, 'base_values'):
                        if len(shap_values_for_plot.base_values.shape) > 1:
                            base_values = shap_values_for_plot.base_values[:, class_index]
                        else:
                            base_values = shap_values_for_plot.base_values
                    else:
                        base_values = np.zeros(len(X_train))

                    # Créer un nouvel objet Explanation pour la classe spécifique
                    class_explanation = shap.Explanation(
                        values=class_values,
                        base_values=base_values,
                        data=shap_values_for_plot.data,
                        feature_names=shap_values_for_plot.feature_names if hasattr(shap_values_for_plot, 'feature_names') else X_train.columns
                    )
                    shap.summary_plot(class_explanation, X_train, max_display=num_features)
                else:
                    # Binaire
                    shap.summary_plot(shap_values_for_plot, X_train, max_display=num_features)
            else:
                # Fallback standard
                shap.summary_plot(shap_values_for_plot, X_train, max_display=num_features)
        except Exception as e:
            print(f"Erreur lors de l'affichage du summary plot: {e}")
            print("Tentative de fallback sur le summary plot standard...")
            try:
                shap.summary_plot(shap_values_for_plot, X_train, max_display=num_features)
            except Exception as e2:
                print(f"Échec également sur le fallback: {e2}")

 #-----------------------------------------------------------------------------------------------------------------#

    def explain_global_regression(self, X_train, model, context="", num_features=10, show_plot=True, model_type="auto"):
        """
        Produit une explication globale des résultats SHAP pour un modèle de régression.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Données d'entraînement
        model : object
            Modèle de ML à expliquer (régression)
        context : str, default=""
            Contexte métier du problème pour l'explication
        num_features : int, default=10
            Nombre de variables à inclure dans l'explication
        show_plot : bool, default=True
            Afficher ou non le summary_plot SHAP
        model_type : str, default="auto"
            Type d'explainer à utiliser ("auto", "tree", "linear", "deep", "kernel")

        Returns:
        --------
        str
            Explication textuelle générée par le LLM
        """
        # Initialisation de l'explainer
        explainer = self._prepare_shap_explainer(X_train, model, model_type)

        # Calcul des valeurs SHAP
        shap_values_for_plot, shap_values_for_calc = self._compute_shap_values(explainer, X_train)

        # Afficher le summary plot
        if show_plot:
            try:
                shap.summary_plot(shap_values_for_plot, X_train, max_display=num_features)
            except Exception as e:
                print(f"Erreur lors de l'affichage du summary plot: {e}")

        # Création du DataFrame résumé
        summary_df = self._create_summary_df(X_train, shap_values_for_calc, num_features)

        # Affichage du résumé
        print("\nRésumé des variables les plus influentes (moyenne des valeurs SHAP absolues) :")
        for _, row in summary_df.iterrows():
            print(f"- {row['feature']} : {row['mean_abs_shap']:.4f}")

        # Préparation du texte pour le LLM
        shap_summary_text = "\n".join([
            f"- {row['feature']} : {row['mean_abs_shap']:.4f}"
            for _, row in summary_df.iterrows()
        ])

        # Explication LLM spécifique à la régression
        try:
            llm_explanation = self.chain_regression.invoke({
                "context": context,
                "shap_summary": shap_summary_text
            })

            # Extraction du contenu depuis la réponse du LLM
            if hasattr(llm_explanation, 'content'):
                explanation_text = llm_explanation.content
            elif isinstance(llm_explanation, dict) and 'content' in llm_explanation:
                explanation_text = llm_explanation['content']
            elif isinstance(llm_explanation, str):
                explanation_text = llm_explanation
            else:
                explanation_text = str(llm_explanation)

            print(f"\n=== Explication générée par l'IA pour modèle de RÉGRESSION (via {self.llm_provider}/{self.model_name}) ===")
            display(Markdown(explanation_text))
            return explanation_text

        except Exception as e:
            print(f"Erreur lors de la génération de l'explication: {e}")
            return f"Erreur: {e}"

#-----------------------------------------------------------------------------------------------------------------#

    def explain_global_classification(self, X_train, model, context="", num_features=10,
                                   show_plot=True, model_type="auto", class_names=None):
      """
      Produit une explication globale des résultats SHAP pour un modèle de classification,
      en expliquant toutes les classes disponibles.

      Parameters:
      -----------
      X_train : pd.DataFrame
          Données d'entraînement
      model : object
          Modèle de ML à expliquer (classification multiclasse)
      context : str, default=""
          Contexte métier du problème pour l'explication
      num_features : int, default=10
          Nombre de variables à inclure dans l'explication
      show_plot : bool, default=True
          Afficher ou non les summary_plots SHAP pour chaque classe
      model_type : str, default="auto"
          Type d'explainer à utiliser ("auto", "tree", "linear", "deep", "kernel")
      class_names : list, optional
          Noms des classes à utiliser dans les explications (si None, utilisera les indices)

      Returns:
      --------
      dict
          Dictionnaire avec les explications textuelles générées par le LLM pour chaque classe
      """
      # Initialisation de l'explainer
      explainer = self._prepare_shap_explainer(X_train, model, model_type)

      # Calcul des valeurs SHAP
      shap_values_for_plot, shap_values_for_calc = self._compute_shap_values(explainer, X_train)

      # Déterminer le nombre de classes
      num_classes = self._get_num_classes(shap_values_for_plot)

      # Si aucun nom de classe n'est fourni, utiliser les indices
      if class_names is None:
          class_names = [f"Classe {i}" for i in range(num_classes)]
      elif len(class_names) < num_classes:
          # Compléter avec des indices si pas assez de noms fournis
          class_names = class_names + [f"Classe {i}" for i in range(len(class_names), num_classes)]

      # Stocker toutes les explications
      explanations = {}

      # Pour chaque classe
      for class_index in range(num_classes):
          # Afficher le summary plot spécifique pour la classe courante
          if show_plot:
              try:
                  print(f"\n=== Summary Plot pour {class_names[class_index]} (index {class_index}) ===")
                  self._display_class_specific_summary_plot(shap_values_for_plot, X_train, class_index, num_features)
              except Exception as e:
                  print(f"Erreur lors de l'affichage du summary plot pour {class_names[class_index]}: {e}")

          # Création du DataFrame résumé spécifique à la classe
          summary_df = self._create_summary_df(X_train, shap_values_for_calc, num_features, class_index=class_index)

          # Affichage du résumé
          print(f"\nRésumé des variables les plus influentes pour {class_names[class_index]} :")
          for _, row in summary_df.iterrows():
              print(f"- {row['feature']} : {row['mean_abs_shap']:.4f}")

          # Préparation du texte pour le LLM
          shap_summary_text = "\n".join([
              f"- {row['feature']} : {row['mean_abs_shap']:.4f}"
              for _, row in summary_df.iterrows()
          ])

          # Adapter le prompt pour indiquer la classe spécifique
          class_prompt = ChatPromptTemplate.from_template(
              self.prompt_template_classification.replace(
                  "pour la première classe",
                  f"pour {class_names[class_index]}"
              ).replace(
                  "spécifiquement pour la première classe",
                  f"spécifiquement pour {class_names[class_index]}"
              ).replace(
                  "pour cette classe spécifique",
                  f"pour {class_names[class_index]}"
              )
          )
          class_chain = class_prompt | self.llm

          # Explication LLM spécifique à la classification pour cette classe
          try:
              llm_explanation = class_chain.invoke({
                  "context": context,
                  "shap_summary": shap_summary_text
              })

              # Extraction du contenu depuis la réponse du LLM
              if hasattr(llm_explanation, 'content'):
                  explanation_text = llm_explanation.content
              elif isinstance(llm_explanation, dict) and 'content' in llm_explanation:
                  explanation_text = llm_explanation['content']
              elif isinstance(llm_explanation, str):
                  explanation_text = llm_explanation
              else:
                  explanation_text = str(llm_explanation)

              print(f"\n=== Explication générée pour {class_names[class_index]} (via {self.llm_provider}/{self.model_name}) ===")
              display(Markdown(explanation_text))

              # Stocker l'explication pour cette classe
              explanations[class_names[class_index]] = explanation_text

          except Exception as e:
              print(f"Erreur lors de la génération de l'explication pour {class_names[class_index]}: {e}")
              explanations[class_names[class_index]] = f"Erreur: {e}"

      return explanations