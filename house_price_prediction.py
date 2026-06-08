
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

none_cols = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
             "GarageType","GarageFinish","GarageQual","GarageCond",
             "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
             "BsmtFinType2","MasVnrType"]
for col in none_cols:
    df[col] = df[col].fillna("None")

df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ["MasVnrArea","GarageYrBlt"]:
    df[col] = df[col].fillna(0)
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(["SalePrice","Id"], axis=1)
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"MAE:  ${mean_absolute_error(y_test, predictions):,.0f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, predictions)):,.0f}")
print(f"R2 Score: {r2_score(y_test, predictions):.3f}")
