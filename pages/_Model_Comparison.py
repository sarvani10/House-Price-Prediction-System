import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as SKLinear
from sklearn.tree import DecisionTreeRegressor as SKTree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from models.linear_scratch import LinearRegressionScratch
from models.tree_scratch import DecisionTreeRegressorScratch

st.title("Model Comparison")

# Load data
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
df.columns = [c.upper() for c in df.columns]

features = ["LSTAT", "RM", "PTRATIO", "DIS", "CHAS"]
X = df[features].values
y = df["MEDV"].values.reshape(-1, 1)

# split
np.random.seed(42)
indices = np.random.permutation(len(X))
test_n = int(0.2 * len(X))
test_idx = indices[:test_n]
train_idx = indices[test_n:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Scratch Linear
lr_scratch = LinearRegressionScratch()
lr_scratch.fit(X_train, y_train)
y_lr_scratch = lr_scratch.predict(X_test)

# Sklearn Linear
lr_sk = SKLinear()
lr_sk.fit(X_train, y_train.ravel())
y_lr_sk = lr_sk.predict(X_test).reshape(-1,1)

# Scratch Tree
dt_scratch = DecisionTreeRegressorScratch(max_depth=5)
dt_scratch.fit(X_train, y_train)
y_dt_scratch = dt_scratch.predict(X_test)

# Sklearn Tree
dt_sk = SKTree(max_depth=5, random_state=42)
dt_sk.fit(X_train, y_train.ravel())
y_dt_sk = dt_sk.predict(X_test).reshape(-1,1)

# Metrics function
def compute_metrics(y_true, y_pred):
    return {
        "R²": round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4)
    }

metrics = {
    "Linear (Scratch)": compute_metrics(y_test, y_lr_scratch),
    "Linear (sklearn)": compute_metrics(y_test, y_lr_sk),
    "Tree (Scratch)": compute_metrics(y_test, y_dt_scratch),
    "Tree (sklearn)": compute_metrics(y_test, y_dt_sk),
}

# Convert to DataFrame
metric_df = pd.DataFrame(metrics).T

# Convert to HTML with bold headers
html_table = metric_df.to_html(classes="table", escape=False)
html_table = html_table.replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong>")

st.subheader(" Model Metrics (Test Set)")

# Bold headers manually
html_table = html_table.replace(
    "<th>R²</th>", "<th><strong>R²</strong></th>"
).replace(
    "<th>RMSE</th>", "<th><strong>RMSE</strong></th>"
).replace(
    "<th>MAE</th>", "<th><strong>MAE</strong></th>"
)

st.markdown(html_table, unsafe_allow_html=True)

# BAR GRAPH 
st.subheader(" R² Score Comparison")

models = list(metrics.keys())
r2_vals = [metrics[m]["R²"] for m in models]

fig, ax = plt.subplots(figsize=(7,4))
ax.bar(models, r2_vals)
ax.set_ylim(0, 1.05)
ax.set_ylabel("R² Score")
ax.set_title("R² Comparison: Scratch vs sklearn")
for i, v in enumerate(r2_vals):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

st.pyplot(fig)

st.info("Sklearn models are optimized and usually achieve higher R² scores.")
