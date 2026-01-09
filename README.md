# MLOps Molecules

## Project Goal

The goal of this project is to develop a machine learning model capable of predicting molecular properties from graph-structured representations of molecules. Molecules are naturally represented as graphs, where atoms correspond to nodes and chemical bonds correspond to edges. By leveraging this structure, the project aims to learn meaningful molecular representations that can be used to accurately predict selected physicochemical properties.

The focus of the project is on building a **functional, accurate, and deployable graph neural network model** rather than on exhaustive architectural experimentation. The model is intended to serve as a foundation that could be further refined or adapted for real-world molecular analysis tasks, such as chemical screening or materials research.

---

## Frameworks and Tools

The project will be implemented using **PyTorch** as the primary deep learning framework, with **PyTorch Lightning** providing a structured and modular approach to model training, validation, and evaluation. PyTorch Lightning reduces boilerplate code and promotes best practices for experiment organization, making the training pipeline easier to maintain and extend.

To handle graph-structured data and message-passing operations, the project uses **PyTorch Geometric (PyG)**. PyG provides optimized data loaders, graph neural network layers, and utility functions that are well-suited for molecular datasets. These frameworks are integrated into a clean codebase designed for reproducibility and clarity, supporting future experimentation or deployment.

---

## Dataset

The project uses the **QM9 dataset**, a widely adopted benchmark in molecular machine learning. QM9 contains approximately 134,000 small organic molecules, each represented as a graph with atom-level and bond-level features. Each molecule is annotated with multiple quantum-chemical properties, enabling supervised learning for regression tasks.

For this project, we will focus on predicting the HOMO-LUMO gap, which is the gap between highest occupied molecular orbital energy and lowest unoccupied molecular orbital energy. The dataset will be split into train and test sets to evaluate model performance and generalization. Although QM9 is used for development and validation, the data pipeline is designed to be flexible, allowing the model to be applied to other molecular datasets in the future.

---

## Model Design

The model developed in this project is a **graph neural network for graph-level regression**. It follows a message-passing paradigm in which node embeddings are iteratively updated based on information from neighboring nodes and edges. After several message-passing layers, a global pooling operation aggregates node-level embeddings into a single fixed-size representation of the molecule.

This graph-level representation is then passed through a prediction head to output the target molecular property. The architecture is chosen to balance expressiveness and computational efficiency, ensuring that the model can be trained and evaluated within reasonable time constraints while still capturing relevant structural information from molecular graphs.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
