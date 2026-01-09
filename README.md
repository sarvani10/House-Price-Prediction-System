## House Price Prediction System
# Project Overview
The House Price Prediction System is an interactive Machine Learning web application built using Streamlit and the Boston Housing Dataset.  
It predicts house prices and compares the performance of multiple regression models using R² scores.
This project includes:
- Models implemented from scratch
- Models using Scikit-learn
- A multi-page Streamlit web application

Streamlit Web Application
The application contains three pages:

1.Home Page
2.Model Comparison Page
- Comparison of R² scores of four models:
  - Linear Regression (from scratch)
  - Decision Tree Regressor (from scratch)
  - Linear Regression (Scikit-learn)
  - Decision Tree Regressor (Scikit-learn)
3.Prediction Page
- Takes user input features  
- Predicts house prices using trained models  

Dataset
- BostonHousing.csv
- Contains housing-related attributes such as crime rate, number of rooms, accessibility, etc.
- Target variable: Median house price

Models Used
From Scratch
- Linear Regression
- Decision Tree Regressor

Scikit-learn
- Linear Regression
- Decision Tree Regressor



## Project Structure
House-Price-Prediction-System/
├── models/
│ ├── pycache/
│ ├── linear_scratch.py
│ └── tree_scratch.py
│
├── pages/
│ ├── _model_comparison.py
│ ├── _prediction.py
│ └── Home_page.py
│
├── BostonHousing.csv
├── Home_Page.py
└── README.md

---

##  How to Run the Project
1. Clone the repository:

git clone https://github.com/sarvani10/House-Price-Prediction-System.git

2.Navigate to the project folder:
cd House-Price-Prediction-System

3.Run the Streamlit app:
streamlit run Home_Page.py

# Key Learnings
Implementing ML models from scratch improves conceptual understanding
Scikit-learn models are optimized and efficient
Decision Trees handle non-linear data better
Streamlit enables quick ML deployment

# Contributors

Sarvani Gogireddy
Mahathi Popuri

# Conclusion

This project demonstrates an end-to-end Machine Learning workflow, combining manual algorithm implementation with library-based models and deploying them through a Streamlit web interface.
