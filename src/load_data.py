import pandas as pd
from pathlib import Path

data_folder = Path("data/PRSA_Data_20130301-20170228")

files = list(data_folder.glob("*.csv"))

print("Number of files:", len(files))

dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

air_quality = pd.concat(dfs, ignore_index=True)

print("Combined shape:", air_quality.shape)
print(air_quality.head())

print(air_quality.info())
print(air_quality.isnull().sum())

print("Number of stations:", air_quality["station"].nunique())
print(air_quality["station"].unique())


print("Duplicate rows:", air_quality.duplicated().sum())

air_quality["datetime"] = pd.to_datetime(
    air_quality[["year", "month", "day", "hour"]]
)

print(air_quality[["datetime", "station", "PM2.5"]].head())

print(air_quality["PM2.5"].describe())

print(air_quality["PM2.5"].value_counts().sort_index().tail(10))

pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

print(air_quality[pollutants].describe())

print("Start date:", air_quality["datetime"].min())
print("End date:", air_quality["datetime"].max())

print(air_quality["station"].value_counts())

missing_pm25 = air_quality["PM2.5"].isnull().mean() * 100

print("PM2.5 missing:", missing_pm25, "%")

import matplotlib.pyplot as plt

air_quality["PM2.5"].dropna().plot(
    kind="hist",
    bins=50,
    figsize=(10, 5)
)

plt.xlabel("PM2.5")
plt.ylabel("Frequency")
plt.title("PM2.5 Distribution")
plt.show()

station_pm25 = air_quality.groupby("station")["PM2.5"].mean().sort_values(ascending=False)

print(station_pm25)

station_pm25.plot(
    kind="bar",
    figsize=(12, 5)
)

plt.xlabel("Station")
plt.ylabel("Average PM2.5")
plt.title("Average PM2.5 by Station")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

monthly_pm25 = air_quality.groupby(
    air_quality["datetime"].dt.to_period("M")
)["PM2.5"].mean()

print(monthly_pm25.head(12))

monthly_pm25.plot(
    figsize=(12, 5),
    kind="line"
)

plt.xlabel("Month")
plt.ylabel("Average PM2.5")
plt.title("Monthly Average PM2.5 Trend")
plt.tight_layout()
plt.show()

monthly_avg = air_quality.groupby(
    air_quality["datetime"].dt.month
)["PM2.5"].mean()

print(monthly_avg)

monthly_avg.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.xlabel("Month")
plt.ylabel("Average PM2.5")
plt.title("Average PM2.5 by Month")
plt.xticks(
    range(12),
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    rotation=0
)

plt.tight_layout()
plt.show()

print(air_quality[["PM2.5", "PM10"]].corr())

print(air_quality["PM2.5"].quantile([0.25, 0.50, 0.75, 0.90, 0.95]))

pm25_df = air_quality.dropna(subset=["PM2.5"]).copy()

print("Original rows:", len(air_quality))
print("Rows after removing missing PM2.5:", len(pm25_df))

daily_pm25 = (
    pm25_df
    .set_index("datetime")
    .groupby("station")["PM2.5"]
    .resample("D")
    .mean()
    .reset_index()
)

print(daily_pm25.head())

daily_pm25["Health_Risk"] = (
    daily_pm25["PM2.5"] > 15
).astype(int)

print(daily_pm25.head())
print(daily_pm25["Health_Risk"].value_counts())

risk_percentage = daily_pm25["Health_Risk"].mean() * 100

print("Higher-risk station-days:", risk_percentage, "%")

print(
    pm25_df[
        ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
         "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    ].corr()["PM2.5"].sort_values(ascending=False)
)

daily_features = (
    pm25_df
    .set_index("datetime")
    .groupby("station")
    .resample("D")
    .mean(numeric_only=True)
    .reset_index()
)

print(daily_features.shape)
print(daily_features.head())

daily_features = daily_features.merge(
    daily_pm25[["station", "datetime", "Health_Risk"]],
    on=["station", "datetime"],
    how="left"
)

print(daily_features[["station", "datetime", "PM2.5", "Health_Risk"]].head())

print(daily_features.columns.tolist())

features = [
    "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "year", "month", "day"
]

X = daily_features[features]
y = daily_features["Health_Risk"]

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print(X.isnull().sum())

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

print("Missing values handled!")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("Scaling complete!")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

print("Model training complete!")

y_pred = model.predict(X_test_scaled)

print("Prediction complete!")

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Low Risk", "High Risk"],
    yticklabels=["Low Risk", "High Risk"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", roc_auc)

feature_importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

feature_importance = feature_importance.sort_values(
    by="Coefficient",
    ascending=False
)

print(feature_importance)