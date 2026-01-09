import streamlit as st
import numpy as np
import pandas as pd
from models.linear_scratch import LinearRegressionScratch
from models.tree_scratch import DecisionTreeRegressorScratch

st.title("Predictions")

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
df.columns = [c.upper() for c in df.columns]

features = ["LSTAT", "RM", "PTRATIO", "DIS", "CHAS"]
X = df[features].values
y = df["MEDV"].values.reshape(-1, 1)

# Train/test split
np.random.seed(42)
indices = np.random.permutation(len(X))
test_n = int(0.2 * len(X))
test_idx = indices[:test_n]
train_idx = indices[test_n:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Train Gradient Descent Linear Regression
lr = LinearRegressionScratch(lr=0.001, epochs=5000)
lr.fit(X_train, y_train)

# Train Scratch Decision Tree
dt = DecisionTreeRegressorScratch(max_depth=5)
dt.fit(X_train, y_train)

st.subheader("**Enter values**")

col1, col2 = st.columns(2)

with col1:
    lstat = st.text_input("LSTAT (% lower status of population)", "12")
    rm = st.text_input("RM (Average number of rooms)", "6")
    chas = st.text_input("CHAS (1 = next to river, 0 = otherwise)", "0")

with col2:
    ptratio = st.text_input("PTRATIO (Pupil–Teacher Ratio by town)", "15")
    dis = st.text_input("DIS (Distance to employment centers)", "4")
    


if st.button("Predict"):
    try:
        input_values = np.array([[float(lstat), float(rm), float(ptratio), float(dis), int(chas)]])
    except:
        st.error("Enter valid numeric inputs.")
        st.stop()

    lr_pred = lr.predict(input_values)[0][0]
    dt_pred = dt.predict(input_values)[0][0]

    st.success(f"Linear Regression (Scratch): **${lr_pred:.2f}k**")
    st.info(f"Decision Tree (Scratch): **${dt_pred:.2f}k**")
