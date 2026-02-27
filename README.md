# Diabetes MVP – Machine Learning Predictor

This project includes:
- Data preprocessing and model training
- A trained AI model
- A simple Python UI "UI.py" to run the predictions locally

## How to Run the Project

### Clone the repository

``` bash
git clone https://github.com/joonasojanen/Software_for_IS_and_AI_diabetes_project.git
cd Software_for_IS_and_AI_diabetes_project
```

## Data

- Download the dataset manually from kaggle and place it in the same folder as this file
- Dataset url: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data

## Environment Setup (Kernel)

### Install required tools

Download and install: 
- Anaconda or Miniconda3
- VS Code

### VS Code Extensions

-   Python (Microsoft)
-   Jupyter (Microsoft)

## Create the Conda environment (recommended)

``` bash
conda create -n diabetes-mvp python=3.10 -y
conda activate diabetes-mvp
conda install -y numpy pandas matplotlib scikit-learn jupyter ipykernel streamlit joblib
python -m ipykernel install --user --name diabetes-mvp --display-name "Python (diabetes-mvp)"
```

## Open the project folder and Select the Jupyter kernel

1.  Open the "diabetes_project.ipynb" notebook
2.  Click Select Kernel
3.  Choose Python (diabetes-mvp)

## Run the notebook

1.  Run all the shells in the diabetes_project.ipynb notenook
2.  Note don't run the block "Hyperparameter tuning using GridSearchCV" if not optimizing the parameters its for testing only

## Running the UI after running the script

1. Make sure you have saved the models "diabetes_logreg_pipeline.joblib"

``` bash
conda activate diabetes-mvp
python UI.py
```

## AI Component

Machine learning classification model built using Scikit-learn. The AI component runs through basic python UI.

### Pipeline

-   Data preprocessing
-   Model training
-   Model saved with joblib
-   UI loads model for prediction

## Required Libraries

numpy, pandas, matplotlib, scikit-learn, jupyter, ipykernel, streamlit,
joblib

## Project Structure

    diabetes-mvp/
    ├── UI.py
    ├── diabetes_logreg_pipeline.joblib
    ├── diabetes_project.ipynb
    ├── diabetes_prediction_dataset.csv
    └── README.md
 
