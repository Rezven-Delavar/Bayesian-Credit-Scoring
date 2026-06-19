from __future__ import annotations

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Column names from german.doc
GERMAN_COLS = [
    "checking_status",        # A11..A14   (categorical)
    "duration_months",        # numeric
    "credit_history",         # A30..A34
    "purpose",                # A40..A410
    "credit_amount",          # numeric
    "savings_status",         # A61..A65
    "employment_since",       # A71..A75
    "installment_rate",       # numeric (1..4)
    "personal_status_sex",    # A91..A95  (gender + marital)
    "other_parties",          # A101..A103
    "residence_since",        # numeric (1..4)
    "property_magnitude",     # A121..A124
    "age_years",              # numeric
    "other_payment_plans",    # A141..A143
    "housing",                # A151..A153
    "existing_credits",       # numeric (1..4)
    "job",                    # A171..A174
    "num_dependents",         # numeric (1..2)
    "telephone",              # A191..A192
    "foreign_worker",         # A201..A202
    "class",              # 1 = good, 2 = bad
]

GERMAN_NUMERIC = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "num_dependents",
]

GERMAN_CATEGORICAL = [c for c in GERMAN_COLS[:-1] if c not in GERMAN_NUMERIC]



data =  pd.read_csv("D:/MSC/Probabilistic machine learning/References/pyro/german credit/ISC18_BNN_Farsi_Conf_revised_v1/code/german_credit.csv", header=0, names=GERMAN_COLS)
data['class'] = data['class'].apply(lambda x: 1.0 if x == 1 else 0.0).astype(int)
print(data)




X = data.drop(columns=["class"])
y = data["class"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), GERMAN_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), GERMAN_CATEGORICAL),
    ]
)

X_train_p = preprocess.fit_transform(X_train).astype(np.float32)
X_test_p  = preprocess.transform(X_test).astype(np.float32)

feature_names = preprocess.get_feature_names_out()

print(X_train_p)
print(X_test_p)
print(feature_names.shape)

print(X_train_p.shape)
print(X_test_p.shape)
