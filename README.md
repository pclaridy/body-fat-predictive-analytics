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
This project investigates the ability to predict body fat percentage from various physiological and body composition measurements. Using a sample of adult individuals, the objective is to compare several regression modeling techniques to determine which most accurately estimates body fat percentage. The study emphasizes the relationship between model selection, predictor quality, and interpretability in biomedical regression analysis.

---

## **2. Data Source**  
The dataset is derived from [`fat.csv`](https://github.com/pclaridy/body-fat-predictive-analytics/blob/main/fat.csv), which contains body composition measurements from 252 adults. Variables include:

- Target: **BodyFat** (percentage)  
- Predictors: **Density**, **Age**, **Weight**, **Height**, and circumferences of the **Neck**, **Chest**, **Abdomen**, **Hip**, **Thigh**, **Knee**, **Ankle**, **Biceps**, **Forearm**, and **Wrist**

The dataset reflects a broad range of demographics and body types, with participants aged 22 to 81 years and weights ranging from 118.5 to 363.15 pounds.

---

## **3. Data Cleaning & Preprocessing**  
Data preparation steps included:

- Initial inspection and correction of formatting issues  
- Handling missing values  
- Validating physiological plausibility of entries  
- Normalizing or transforming variables as needed for model assumptions  
- Pre-checks for multicollinearity and outlier detection prior to model development

These steps ensured that each model could be fairly compared using consistent and clean data.

---

## **4. Exploratory Data Analysis (EDA)**

### Scatter Plots

- **Siri vs. Brozek**  
  ![Scatter Plot of Siri vs. Brozek](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/scatterplot_siri_vs_brozek.png)

- **Density vs. Brozek**  
  ![Scatter Plot of Density vs. Brozek](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/scatterplot_density_vs_brozek.png)

- **Biceps vs. Brozek**  
  ![Scatter Plot of Biceps vs. Brozek](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/scatterplot_biceps_vs_brozek.png)

### Residual Plots

- **Siri**  
  ![Residual Plot for Siri](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/residual_plot_siri.png)

- **Density**  
  ![Residual Plot for Density](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/residual_plot_density.png)

- **Biceps**  
  ![Residual Plot for Biceps](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/residual_plot_biceps.png)

### Q-Q Plots

- **Siri**  
  ![Q-Q Plot for Siri](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/qq_plot_siri.png)

- **Density**  
  ![Q-Q Plot for Density](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/qq_plot_density.png)

- **Biceps**  
  ![Q-Q Plot for Biceps](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/qq_plot_biceps.png)

### Histograms of Residuals

- **Siri**  
  ![Histogram of Residuals for Siri](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/histogram_of_residuals_siri.png)

- **Density**  
  ![Histogram of Residuals for Density](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/histogram_of_residuals_density.png)

- **Biceps**  
  ![Histogram of Residuals for Biceps](https://github.com/pclaridy/us-housing-market-analytics-2023/raw/main/reports/figures/histogram_of_residuals_biceps.png)

Histograms and Q-Q plots showed that the residuals for Siri and Density were approximately normally distributed, which supports the assumptions of linear regression. The residuals for Biceps were more irregular, with signs of skewness and wider spread, indicating that the model fit was weaker for this variable.

---

## **5. Modeling Approach**

The following regression models were developed and compared:

- Full Linear Model  
- Best Subset Regression  
- Stepwise Regression using AIC  
- Ridge Regression  
- LASSO Regression  
- Principal Component Regression (PCR)  
- Partial Least Squares (PLS)

Models were tuned and validated using Mean Squared Error (MSE) and Monte Carlo simulations for robustness.

---

## **6. Evaluation Metrics**

### Initial Test MSE

| Model              | MSE          |
|--------------------|--------------|
| PLS                | 2.2786       |
| Subset             | 0.0493       |
| PCR                | 37.3691      |
| LASSO              | 150.2326     |
| Ridge              | 1.0285       |
| Stepwise           | **0.0108**   |
| Linear Regression  | 0.0495       |

### Average MSE Across Monte Carlo Simulations

| Model              | Avg MSE      |
|--------------------|--------------|
| PLS                | 29.4989      |
| Subset             | **0.0567**   |
| PCR                | 33.3529      |
| LASSO              | 0.0644       |
| Ridge              | 0.6902       |
| Stepwise           | 0.1139       |
| Linear Regression  | 0.0996       |

---

## **7. Outcome**

The results of this analysis highlight a few important conclusions:

- **Stepwise Regression had the lowest MSE in the initial test set.**  
  When the models were evaluated on a single holdout set, Stepwise Regression produced the most accurate predictions. Its mean squared error was the smallest of all models, indicating it captured the underlying relationships in the data effectively.

- **Best Subset Regression had the lowest average MSE across simulations.**  
  Across multiple iterations using Monte Carlo cross-validation, the Best Subset model consistently yielded the best average performance. This suggests that it generalized better across different subsets of the data and was the most reliable model overall.

- **Linear models with variable selection performed better than regularized and dimension-reduction models.**  
  In this dataset, models that selected the most relevant variables outperformed more complex approaches such as Ridge, LASSO, Principal Component Regression, and Partial Least Squares. This suggests that the signal in the data was strong enough that simple, well-selected predictors were more effective than techniques that added regularization or transformed the feature space.

---

## **8. Tools Used**

- **Language**: R  
- **Libraries**: `ggplot2`, `dplyr`, `glmnet`, `pls`, `leaps`  
- **Model Types**: Linear Regression, Stepwise Regression, Subset Selection, Ridge, LASSO, PCR, PLS  
- **Validation Method**: Monte Carlo simulations with MSE as the evaluation metric  
- **Visualizations**: Diagnostic and performance plots available in the [figures directory](https://github.com/pclaridy/us-housing-market-analytics-2023/tree/main/reports/figures)

---

## **9. Business Impact / Use Case**

Accurately predicting body fat percentage from non-invasive physiological measurements has a wide range of applications:

- **Healthcare Providers** can use these models for early detection of obesity-related risk factors, allowing for proactive patient counseling and interventions without needing expensive tools like DEXA scans.
- **Fitness Professionals** and **Athletic Programs** can implement these models in performance tracking apps to monitor changes in body composition over time using accessible inputs.
- **Insurance Companies** may incorporate predictive models into underwriting processes to assess long-term health risks in a cost-efficient and data-driven manner.
- **Wearable Technology Companies** could integrate this kind of modeling into apps or devices, offering users a more holistic view of their physical health based on simple measurements.
- **Research and Public Health** agencies can use these approaches to evaluate trends in body composition at the population level and identify at-risk groups using readily available data.

This project demonstrates how accessible statistical methods and data science techniques can offer practical, scalable tools for understanding human health and driving evidence-based decisions in both personal and professional domains.
