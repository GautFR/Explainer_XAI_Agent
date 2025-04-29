from setuptools import setup, find_packages

setup(
    name='explainer_agent',
    version='0.1.0',
    author='Ton Nom',
    author_email='ton.email@example.com',
    description='Un agent IA pour expliquer des prédictions de modèles ML avec LIME et LangChain',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ton-github/lime_explainer_agent',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'lime',
        'ipython',
        'langchain-core',
        'langchain-ollama'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
