# Titanic ML Package

![Titanic](titanic.jpg)

Welcome to the Titanic ML Package repository! This repository contains a Python package specifically designed to tackle the famous Titanic dataset. The goal of this package is to provide a comprehensive solution for training and evaluating machine learning models on the Titanic dataset, facilitating the exploration of passenger demographics and predicting their survival probabilities.

## About the Titanic Dataset

The Titanic dataset is a well-known and widely used dataset in the field of data analysis and machine learning. It comprises information about passengers who were aboard the RMS Titanic during its ill-fated maiden voyage in 1912. The dataset offers a rich variety of features, including passenger attributes such as age, gender, class, and family relationships. The most crucial feature, "Survived," determines whether a passenger survived the tragic maritime disaster.

## Purpose of the Package

The main purpose of this package is to provide a user-friendly and efficient solution for ML engineers, data scientists to explore, analyze, and build predictive models on the Titanic dataset. The package is built with object-oriented programming principles and adheres to best coding practices, ensuring a robust and maintainable codebase.

## Features and Capabilities

The Titanic ML Package offers the following key features:

- **Command-Line Interface (CLI):** The package comes with a CLI that allows users to interact with the code effortlessly. Through the CLI, users can perform various operations, such as training models, evaluating model performance, and do grid search on one of the 5 classifiers abailable.

- **Model Training and Evaluation:** The package provides a streamlined workflow for training machine learning models on the Titanic dataset. Users can experiment with different algorithms, hyperparameters The package also includes evaluation the Accuracy, recall and F1Score metric. 

- **Exploratory Data Analysis (EDA):** The package incorporates an exploratory data analysis, enabling users to gain insights into the dataset, understand the relationships between features and survival, and visualize patterns using intuitive plots and visualizations.

## Getting Started

To get started with the Titanic ML Package you just need to clone the repository and run the following code: 
### Install : 
1. install anaconda: https://dev.to/waylonwalker/installing-miniconda-on-linux-from-the-command-line-4ad7
2. install environment: 
```bash
conda env create -f environment.yml
```

### Usage: 
To use the model you have 3 possible commands you can use: 
```bash
python run.py --command train
python run.py --command personalized_train
python run.py --command grid_search
```
Each one of this commands will generate a trained model, train will use the default hyperparameters, meanwhile personalized_train will ask the hyperparameters you prefer, lastly grid_search will generate a grid of parameters and give you the best model of the classifier you chose. 
