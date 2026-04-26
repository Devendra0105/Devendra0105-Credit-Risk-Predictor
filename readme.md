# Credit Score Prediction 💳

A machine learning project to predict credit risk using the German Credit Dataset.

## Project Structure

```
├── APP/
│   └── app.py                 # Streamlit application for predictions
├── dataset/
│   └── german_credit_data.csv # Original dataset
├── EDA and ML/
│   ├── EDA.py                 # Exploratory Data Analysis
│   ├── grid_searchCV.py       # Hyperparameter tuning
│   ├── ModelTraining.py       # Model training and evaluation
│   └── PKL/
│       ├── features.pkl       # Feature order
│       └── Model.pkl          # Trained SVC model
├── test_results/
│   ├── balanced_500_data.csv
│   ├── output_500.txt
│   └── test_500_balanced.py
├── About_Model.md             # Model documentation and reports
└── requirments.txt            # Required libraries
```

## Features

- **EDA & Visualization** - Data analysis with pandas, matplotlib, seaborn
- **Multiple ML Models** - Logistic Regression, Random Forest, SVC, Decision Tree, KNN, XGBoost, Gradient Boosting
- **Best Model** - SVC with RBF kernel (~82% accuracy on balanced test set)
- **Interactive App** - Streamlit web app for real-time predictions

## Quick Start

### Install Dependencies
```bash
pip install pandas scikit-learn xgboost streamlit joblib
```

### Run Streamlit App
```bash
cd APP
streamlit run app.py
```

### Run EDA or Model Training
```bash
cd "EDA and ML"
python EDA.py
python ModelTraining.py
```

## Dataset

- **Source**: German Credit Data (UCI Machine Learning Repository)
- **Samples**: 1000
- **Features**: 21 (status_account, month_duration, credit_history, purpose, credit_amount, etc.)
- **Target**: good/bad credit risk

## Models Evaluated

|        Model        | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.775    |
| Random Forest       | 0.780    |
| **SVC (RBF)**       | **~0.82**|
| Decision Tree       | 0.75     |
| KNN                 | 0.73     |
| XGBoost             | 0.805    |
| Gradient Boosting   | 0.795    |

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- XGBoost
- joblib
- Streamlit

---
*Credit Risk Prediction Project*

`` This project is developed for learning and demonstration purposes; while the model shows reasonable performance, it is not intended for deployment in real-world banking systems, it is a learning-oriented project designed to explore machine learning techniques in credit risk prediction; despite promising results, it is not production-ready for real banking applications. ``
