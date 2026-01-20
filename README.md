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

## QM9 Property Glossary

The QM9 dataset contains 19 regression targets representing various quantum-chemical properties. Below is a complete reference of all available properties:

| Target | Property | Description | Unit |
|--------|----------|-------------|------|
| 0 | μ | Dipole moment | D |
| 1 | α | Isotropic polarizability | a₀³ |
| 2 | εₕₒₘₒ | Highest occupied molecular orbital energy | Hₐ |
| 3 | εₗᵤₘₒ | Lowest unoccupied molecular orbital energy | Hₐ |
| 4 | Δε | Gap between εₕₒₘₒ and εₗᵤₘₒ | Hₐ |
| 5 | ⟨R²⟩ | Electronic spatial extent | a₀² |
| 6 | ZPVE | Zero point vibrational energy | Hₐ |
| 7 | U₀ | Internal energy at 0K | Hₐ |
| 8 | U | Internal energy at 298.15K | Hₐ |
| 9 | H | Enthalpy at 298.15K | Hₐ |
| 10 | G | Free energy at 298.15K | Hₐ |
| 11 | Cᵥ | Heat capacity at 298.15K | cal/mol·K |
| 12 | U₀ᴬᵀᴼᴹ | Atomization energy at 0K | Hₐ |
| 13 | Uᴬᵀᴼᴹ | Atomization energy at 298.15K | Hₐ |
| 14 | Hᴬᵀᴼᴹ | Atomization enthalpy at 298.15K | Hₐ |
| 15 | Gᴬᵀᴼᴹ | Atomization free energy at 298.15K | Hₐ |
| 16 | A | Rotational constant | GHz |
| 17 | B | Rotational constant | GHz |
| 18 | C | Rotational constant | GHz |

**Note:** This project focuses on **Target 4 (Δε)**, the HOMO-LUMO gap, which is a key indicator of molecular reactivity and stability.

**Unit abbreviations:**
- D = Debye
- a₀ = Bohr radius
- Hₐ = Hartree (atomic unit of energy)

---

## Model Design

The model developed in this project is a **graph neural network for graph-level regression**. It follows a message-passing paradigm in which node embeddings are iteratively updated based on information from neighboring nodes and edges. After several message-passing layers, a global pooling operation aggregates node-level embeddings into a single fixed-size representation of the molecule.

This graph-level representation is then passed through a prediction head to output the target molecular property. The architecture is chosen to balance expressiveness and computational efficiency, ensuring that the model can be trained and evaluated within reasonable time constraints while still capturing relevant structural information from molecular graphs.
## Target Dictionary


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
