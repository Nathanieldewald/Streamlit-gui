import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

import functions as f
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# import data

df = pd.read_csv("telco.csv")

# Column selection
selected_columns = st.multiselect("Select Columns ", df.columns)

# Show selected columns
df_selected = df[selected_columns]

# Show dataframe
st.dataframe(df_selected)

# separate numeric and categorical columns
num_cols = st.multiselect("Select Numeric Columns ", df_selected.columns)
num_cols = df_selected[num_cols]
st.dataframe(num_cols)
# drop columns once selected
df_selected.drop(columns=num_cols, inplace=True)
cat_cols = st.multiselect("Select Categorical Columns ", df_selected.columns)
cat_cols = df_selected[cat_cols]
# Show categorical columns
st.write("Categorical Columns", cat_cols)


# Show numeric columns
num_cols = f.num_nulls(num_cols)
num_cols = f.change_dtype_to_float(num_cols)
st.write("Numeric Columns Cleaned", num_cols)

# Show categorical columns
cat_cols = f.cat_nulls(cat_cols)
cat_cols = f.change_dtype_to_object(cat_cols)
cat_cols = pd.get_dummies(cat_cols, drop_first=True)
st.write("Categorical Columns Cleaned", cat_cols)

# join categorical and numerical columns

df = f.join_cols(num_cols, cat_cols)
st.write("Final Dataframe", df)

# split data into train, validation and test
select_stratify = st.multiselect("Select Stratify ", df.columns)
Y = df[select_stratify]
X = df.drop(columns=select_stratify)

x_train, x_test, x_val, y_train, y_test, y_val = f.split_data(X, Y)

# Show train, validation and test
st.write("Train", x_train)
st.write("Train Shape", x_train.shape)
st.write("Validation", x_val)
st.write("Validation Shape", x_val.shape)
st.write("Test", x_test)
st.write("Test Shape", x_test.shape)

# model selection
select_model = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
if select_model == "Logistic Regression":
    logit = LogisticRegression()
    logit.fit(x_train, y_train)
    y_pred = logit.predict(x_test)
    st.write("Accuracy", accuracy_score(y_test, y_pred))
    st.write("Precision", precision_score(y_test, y_pred))
    st.write("Recall", recall_score(y_test, y_pred))
    st.write("F1 Score", f1_score(y_test, y_pred))
    st.write("Confusion Matrix", confusion_matrix(y_test, y_pred))
    st.write("Classification Report", print(classification_report(y_test, y_pred)))








if select_model == "Random Forest":
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    st.write("Accuracy", accuracy_score(y_test, y_pred))
    st.write("Precision", precision_score(y_test, y_pred))
    st.write("Recall", recall_score(y_test, y_pred))
    st.write("F1 Score", f1_score(y_test, y_pred))
    st.write("Confusion Matrix", confusion_matrix(y_test, y_pred))
    st.write("Classification Report", classification_report(y_test, y_pred))




