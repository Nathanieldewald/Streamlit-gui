from turtle import st

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
def num_nulls(x):
    for col in x.columns:
        if x[col].isnull().sum() > 0:
            replace = x[col].mean()
            x[col] = x[col].fillna(replace)
    return x

def cat_nulls(x):
    for col in x.columns:
        if x[col].isnull().sum() > 0:
            replace = x[col].mode()[0]
            x[col] = x[col].fillna(replace)
    return x

def change_dtype_to_object(x):
    for col in x.columns:
        x[col] = x[col].astype("object")
    return x

def change_dtype_to_float(x):
    for col in x.columns:
        x[col] = x[col].replace(",", "")
        x[col] = x[col].replace(" ", "0")
        x[col] = x[col].astype("float")
    return x

# join categorical and numerical columns
def join_cols(x, y):
    df = pd.concat([x, y], axis=1)
    return df

# split data into train, validation and test
def split_data(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=111, stratify=Y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=111, stratify=y_train)
    return x_train, x_test, x_val, y_train, y_test, y_val

# graphing logistic regression
def graph_logit(x, y, model):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, model.predict_proba(x)[:, 1], color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Logistic Regression")
    st.pyplot(fig)