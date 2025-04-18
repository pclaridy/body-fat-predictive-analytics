# **Predicting Body Fat Percentage: A Comparative Study of Regression Techniques**

## **Table of Contents**

1. [Problem Statement](#1-problem-statement)  
2. [Data Source](#2-data-source)  
3. [Data Cleaning & Preprocessing](#3-data-cleaning--preprocessing)  
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)  
5. [Modeling Approach](#5-modeling-approach)  
6. [Evaluation Metrics](#6-evaluation-metrics)  
7. [Outcome](#7-outcome)  
8. [Tools Used](#8-tools-used)  
9. [Business Impact / Use Case](#9-business-impact--use-case)

---

## **1. Problem Statement**

Accurately estimating body fat percentage is essential in both clinical health assessments and personal fitness planning, yet direct measurement techniques can be invasive, expensive, or impractical for routine use. This project aims to build and evaluate regression models that predict body fat percentage from easily measurable physiological features such as circumference measurements, age, weight, and height. By comparing different linear and regularized regression approaches, the analysis seeks to identify the most accurate and interpretable model. The ultimate goal is to support more accessible and reliable body composition assessments that can aid in preventative care, fitness goal tracking, and health risk evaluation.

---

## **2. Data Source**

The dataset [`fat.csv`](https://github.com/pclaridy/body-fat-predictive-analytics/blob/main/fat.csv) contains body composition measurements from 252 adults. Key variables include:

- **Target**: BodyFat (%)  
- **Predictors**: Density, Age, Weight, Height, and circumference measurements (e.g., Neck, Chest, Abdomen, Biceps)

Participants ranged in age from 22 to 81 and weights from 118.5 to 363.15 pounds, offering a diverse and representative sample.

---

## **3. Data Cleaning & Preprocessing**

Preprocessing steps included:

- Removing outliers and entries with implausible values  
- Verifying consistency in units  
- Addressing missing values  
- Preparing variables for regression assumptions (normality, linearity, multicollinearity)

The dataset was normalized and cleaned for robust model performance and valid interpretation.

---

## **4. Exploratory Data Analysis (EDA)**

### Scatter Plots

**Siri vs. Brozek**  
![Siri vs Brozek](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/scatterplot_siri_vs_brozek.png)

**Density vs. Brozek**  
![Density vs Brozek](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/scatterplot_density_vs_brozek.png)

**Biceps vs. Brozek**  
![Biceps vs Brozek](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/scatterplot_biceps_vs_brozek.png)

### Residual Plots

**Siri**  
![Residual Plot - Siri](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/residual_plot_siri.png)

**Density**  
![Residual Plot - Density](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/residual_plot_density.png)

**Biceps**  
![Residual Plot - Biceps](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/residual_plot_biceps.png)

### Q-Q Plots

**Siri**  
![Q-Q Plot - Siri](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/qq_plot_siri.png)

**Density**  
![Q-Q Plot - Density](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/qq_plot_density.png)

**Biceps**  
![Q-Q Plot - Biceps](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/qq_plot_biceps.png)

### Histograms of Residuals

**Siri**  
![Histogram - Siri](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/histogram_of_residuals_siri.png)

**Density**  
![Histogram - Density](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/histogram_of_residuals_density.png)

**Biceps**  
![Histogram - Biceps](https://github.com/pclaridy/body-fat-predictive-analytics/raw/main/figures/histogram_of_residuals_biceps.png)

**Interpretation:**  
Residuals for Siri and Density were nearly normal, indicating good model fit. The Biceps residuals showed a wider spread and irregular distribution, suggesting that a linear model may not capture the relationship as well for that feature.

---

## **5. Modeling Approach**

The following regression models were developed and compared:

- Full Linear Regression  
- Best Subset Selection  
- Stepwise Regression (AIC-based)  
- Ridge Regression  
- LASSO Regression  
- Principal Component Regression (PCR)  
- Partial Least Squares (PLS)

Each model was tuned and validated using mean squared error and Monte Carlo simulations.

---

## **6. Evaluation Metrics**

### MSE on Initial Test Set

| Model              | MSE          |
|--------------------|--------------|
| Stepwise           | **0.0108**   |
| Subset             | 0.0493       |
| Linear Regression  | 0.0495       |
| Ridge              | 1.0285       |
| PLS                | 2.2786       |
| PCR                | 37.3691      |
| LASSO              | 150.2326     |

### Average MSE from Monte Carlo Simulations

| Model              | Average MSE  |
|--------------------|--------------|
| Subset             | **0.0567**   |
| LASSO              | 0.0644       |
| Linear Regression  | 0.0996       |
| Stepwise           | 0.1139       |
| Ridge              | 0.6902       |
| PLS                | 29.4989      |
| PCR                | 33.3529      |

---

## **7. Outcome**

- **Stepwise Regression had the lowest MSE in the initial test set.**  
  It performed best on a fixed validation split, showing strong accuracy for the available data.

- **Best Subset Regression had the lowest average MSE across simulations.**  
  It performed most consistently across different train-test splits, indicating strong generalization.

- **Linear models with variable selection outperformed more complex models.**  
  Simpler techniques like stepwise and subset selection produced stronger results than regularized models or component-based approaches. This suggests that a small number of well-selected variables captured the predictive signal more effectively than global shrinkage or dimensionality reduction.

- **Underperformance of Ridge, LASSO, PCR, and PLS**  
  While multicollinearity was present in the data, it did not significantly hinder traditional linear models. Regularization techniques such as Ridge and LASSO may have over-penalized some of the most informative coefficients, reducing their contribution to the prediction. Similarly, component-based methods like PLS and PCR transform the original features into new, uncorrelated components that can obscure important individual relationships. These techniques are often effective in high-dimensional or noisy settings, but in this case, they appeared to dilute key patterns in the data and limit interpretability without offering gains in accuracy.

---

## **8. Tools Used**

- **Language**: R  
- **Libraries**: `ggplot2`, `dplyr`, `glmnet`, `pls`, `leaps`  
- **Validation**: Monte Carlo simulation  
- **Evaluation Metric**: Mean Squared Error (MSE)  

---

## **9. Business Impact / Use Case**

This project provides a practical, interpretable framework for predicting body fat percentage using affordable and accessible body measurements. It offers tangible benefits to:

- **Healthcare**: Provides low-cost, fast screening for obesity-related risks  
- **Fitness Professionals**: Offers clients data-driven health assessments  
- **Wearable Tech**: Enables predictive modeling features in fitness apps  
- **Insurance Providers**: Supports underwriting decisions with lightweight biometric models  
- **Public Health Researchers**: Tracks body composition trends at scale using reproducible, non-invasive data inputs

These models support proactive health decisions and remain easy to interpret for both technical and non-technical audiences.
