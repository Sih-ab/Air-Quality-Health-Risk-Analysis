# Air Quality & Health Risk Analysis

## Overview

This project analyzes air quality data from Beijing and uses machine learning to classify daily air-quality health risk.

The project focuses on exploring PM2.5 and other air pollutants, identifying patterns across monitoring stations and months, and building a Logistic Regression model to classify daily health risk.

---

## Dataset

The dataset used in this project is the **Beijing Multi-Site Air Quality Data** from the UCI Machine Learning Repository.

- Time period: 2013–2017
- Monitoring stations: 12
- Records: 420,768 hourly observations
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata)

The dataset contains hourly air-quality measurements and weather information collected from multiple monitoring stations in Beijing.

---

## Objectives

The main objectives of this project are:

- Analyze air-quality data
- Explore PM2.5 levels across stations and months
- Analyze relationships between different pollutants
- Aggregate hourly data into daily measurements
- Create a daily health-risk classification
- Build a machine learning model for health-risk classification
- Evaluate the model using multiple performance metrics

---

## Exploratory Data Analysis

The analysis includes:

- Dataset structure and information
- Missing value analysis
- Duplicate checking
- PM2.5 statistical analysis
- Station-wise PM2.5 analysis
- Monthly PM2.5 analysis
- Time-series analysis
- Pollutant correlation analysis
- PM2.5 quantile analysis
- Daily PM2.5 aggregation

### Key Findings

- PM2.5 showed a strong positive correlation with PM10.
- PM2.5 levels varied across monitoring stations.
- Winter months generally showed higher average PM2.5 levels.
- August had the lowest average monthly PM2.5 among the months analyzed.
- PM10 had the strongest positive correlation with PM2.5 among the analyzed pollutants.

---

## Health Risk Classification

Daily average PM2.5 was used to create a project-specific health-risk classification.

A daily PM2.5 value above **15 µg/m³** was classified as:

- `0` → Lower risk
- `1` → Higher risk

This threshold is based on the WHO 2021 24-hour PM2.5 guideline.

The health-risk classification is used as the target variable for the machine learning model.

---

## Machine Learning

A **Logistic Regression** model was used for daily health-risk classification.

### Features

The model used the following features:

- PM10
- SO2
- NO2
- CO
- O3
- TEMP
- PRES
- DEWP
- RAIN
- WSPM
- Year
- Month
- Day

PM2.5 itself was excluded from the model features to avoid directly using the variable from which the target was created.

### Data Preprocessing

The following preprocessing steps were performed:

- Missing values were handled using median imputation.
- Features were standardized using `StandardScaler`.
- Data was divided into training and testing sets.
- Logistic Regression was used for classification.

---

## Model Performance

The Logistic Regression model achieved:

| Metric | Score |
|---|---:|
| Accuracy | **88.08%** |
| ROC-AUC | **0.9682** |
| Risk Class Precision | **99%** |
| Risk Class Recall | **87%** |
| Risk Class F1-score | **93%** |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0 | 45% | 94% | 61% |
| 1 | 99% | 87% | 93% |

The model performs particularly well in identifying the higher-risk class.

---

## Confusion Matrix

The confusion matrix was:

```text
[[ 325   22]
 [ 396 2764]]