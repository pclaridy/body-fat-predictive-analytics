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

This project explores how various physiological and body composition measurements can be used to predict body fat percentage. The goal is to compare several regression models and determine which method provides the most accurate and interpretable results. This is especially relevant in health and fitness contexts, where reliable body fat estimation can inform preventative care and training decisions.

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
  Simpler techniques like stepwise and subset selection produced stronger results than regularized models or component-based approaches. This suggests that a small number of well-selected variables capture the predictive signal more effectively than global shrinkage or dimensionality reduction.

- **Why did Ridge, LASSO, PLS, and PCR underperform?**  
  In this dataset, multicollinearity was present but manageable, so the penalization in Ridge and LASSO may have unnecessarily shrunk important coefficients. Similarly, component-based methods like PLS and PCR reduced predictors into orthogonal combinations that may have diluted the original structure of the data. These methods can be powerful when there is noise or redundancy, but here they appeared to obscure meaningful relationships between individual measurements and body fat percentage.

---

## **8. Tools Used**

- **Language**: R  
- **Libraries**: `ggplot2`, `dplyr`, `glmnet`, `pls`, `leaps`  
- **Validation**: Monte Carlo simulation  
- **Evaluation Metric**: Mean Squared Error (MSE)  
- **Repository**: [pclaridy/body-fat-predictive-analytics](https://github.com/pclaridy/body-fat-predictive-analytics)  
- **Visuals Folder**: [figures](https://github.com/pclaridy/body-fat-predictive-analytics/tree/main/figures)

---

## **9. Business Impact / Use Case**

This project provides a practical, interpretable framework for predicting body fat percentage using affordable and accessible body measurements. It offers tangible benefits to:

- **Healthcare**: Provides low-cost, fast screening for obesity-related risks  
- **Fitness Professionals**: Offers clients data-driven health assessments  
- **Wearable Tech**: Enables predictive modeling features in fitness apps  
- **Insurance Providers**: Supports underwriting decisions with lightweight biometric models  
- **Public Health Researchers**: Tracks body composition trends at scale using reproducible, non-invasive data inputs

These models support proactive health decisions and remain easy to interpret for both technical and non-technical audiences.
