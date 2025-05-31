# Predicting Customer Conversion in Digital Marketing Campaigns

## Overview
This project analyzes factors influencing customer conversion in digital marketing campaigns using machine learning. Leveraging a synthetic dataset of customer demographics, campaign engagement metrics, and historical behavior, we aim to predict conversion outcomes and provide data-driven recommendations to enhance marketing effectiveness.

## Project Objective
- **Proposal Goal**: Predict customer conversion likelihood and identify key drivers to personalize marketing efforts and improve ROI.
- **Final Outcome**: Successfully developed and validated models (XGBoost, Random Forest, etc.) to classify conversion probability and derive actionable insights for campaign optimization.

## Dataset
Sourced from Kaggle, the dataset includes:

- **Demographics**: Age, Gender, Income  
- **Marketing Details**: Campaign channel, campaign type, ad spend, CTR, conversion rate  
- **Customer Behavior**: Website visits, time on site, pages per visit, email opens/clicks  
- **Historical Loyalty**: Loyalty points, previous purchases  
- **Target Variable**: `Conversion` (1 = converted, 0 = not converted)

> 📎 **Dataset Link**: [Kaggle – Predict Conversion in Digital Marketing](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset/data)

## Technologies Used
- **Python** (Jupyter Notebook)
- **Scikit-learn** (Modeling and Evaluation)
- **XGBoost** (Advanced boosting model)
- **Imbalanced-learn (SMOTE)** (Class balancing)
- **Pandas, NumPy** (Data manipulation)
- **Matplotlib, Seaborn** (Visualization)

## How to Run the Project

### 1. Clone the Repository
### 2. Install Dependencies
We recommend creating a virtual environment:
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```
### 3. Run the Notebook
Open the notebook with Jupyter or VS Code:
```
jupyter notebook "Digital marketing.ipynb"
```

## Workflow

### 1. Data Preprocessing
- Dropped irrelevant columns
- Label encoded binary categorical features
- One-hot encoded multi-class features
- Applied SMOTE to balance classes
- Removed multicollinearity using correlation matrix

### 2. Exploratory Data Analysis (EDA)
- Found highest conversion among 36–45 age group
- Referral and PPC channels outperformed other channels
- Email engagement and website visits strongly correlated with conversion

### 3. Modeling
Trained multiple models and compared performance:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

### 4. Model Evaluation
- **Metrics Used**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Best Performing Model**: XGBoost with optimized hyperparameters
- ROC curves used for comparative validation

### 5. Key Features Identified
- **XGBoost**: Campaign Channel, Campaign Type
- **Random Forest**: Campaign Type, Time on Site, Ad Spend, Email Clicks

## Deployment & Monitoring
- Model designed to integrate with CRM systems for real-time predictions
- Suggested quarterly monitoring of actual conversion vs predicted
- Retrain model as customer behavior evolves

## Insights Gained
- Customers with high email and web engagement are more likely to convert
- Campaign personalization by channel and type can improve ROI
- Income, age, and loyalty history significantly influence conversion

## Future Enhancements
- Create a real-time web dashboard to visualize conversion likelihoods
- Include external marketing metrics (competitor pricing, ad rankings)
- Perform segmentation for tailored campaign strategies
