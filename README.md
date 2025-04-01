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
The dataset is derived from `fat.csv`, which contains body composition measurements from 252 adults. Variables include:

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

These plots visualize the relationships between selected predictors and the target variable (body fat % as estimated using the Brozek formula):

- **Siri vs. Brozek**  
  ![Scatter Plot of Siri vs. Brozek](figures/scatterplot_siri_vs_brozek.png)

- **Density vs. Brozek**  
  ![Scatter Plot of Density vs. Brozek](figures/scatterplot_density_vs_brozek.png)

- **Biceps vs. Brozek**  
  ![Scatter Plot of Biceps vs. Brozek](figures/scatterplot_biceps_vs_brozek.png)

These visualizations confirm a strong linear relationship for `Siri` and `Density`, while `Biceps` displays more variability.

---

### Residual Plots

Residuals were plotted to evaluate homoscedasticity and model fit:

- **Siri**  
  ![Residual Plot for Siri](figures/residual_plot_siri.png)

- **Density**  
  ![Residual Plot for Density](figures/residual_plot_density.png)

- **Biceps**  
  ![Residual Plot for Biceps](figures/residual_plot_biceps.png)

Residuals for `Siri` and `Density` were fairly evenly distributed. The `Biceps` model showed more spread, indicating less stability.

---

### Q-Q Plots

Q-Q plots assess whether residuals are normally distributed:

- **Siri**  
  ![Q-Q Plot for Siri](figures/qq_plot_siri.png)

- **Density**  
  ![Q-Q Plot for Density](figures/qq_plot_density.png)

- **Biceps**  
  ![Q-Q Plot for Biceps](figures/qq_plot_biceps.png)

Deviations from the line were minimal for `Siri` and `Density`, while `Biceps` displayed greater skewness.

---

### Histograms of Residuals

Histograms provide further insight into the distribution of residuals:

- **Siri**  
  ![Histogram of Residuals for Siri](figures/histogram_of_residuals_siri.png)

- **Density**  
  ![Histogram of Residuals for Density](figures/histogram_of_residuals_density.png)

- **Biceps**  
  ![Histogram of Residuals for Biceps](figures/histogram_of_residuals_biceps.png)

The distributions for `Siri` and `Density` residuals were near normal. The histogram for `Biceps` was more irregular.

---

## **5. Modeling Approach**

The following regression models were implemented using R:

- **Full Linear Model**: Includes all predictors  
- **Best Subset Linear Regression**: Selects an optimal subset based on performance  
- **Stepwise Regression (AIC)**: Automatically includes or removes predictors  
- **Ridge Regression**: Penalizes large coefficients to reduce multicollinearity  
- **LASSO Regression**: Shrinks less important coefficients to zero  
- **Principal Component Regression (PCR)**: Uses orthogonal principal components  
- **Partial Least Squares (PLS)**: Balances variance in predictors and target variable

### Validation Strategy
- **Performance Metric**: Mean Squared Error (MSE)  
- **Robustness Check**: Monte Carlo Cross-Validation  
- **Tuning**: Regularization strength and component counts for applicable models

---

## **6. Evaluation Metrics**

MSE was calculated for each model on the test data and averaged across simulations to compare stability and generalization. Results are presented below:

#### MSE Values from Initial Fitting

| Model              | MSE          |
|--------------------|--------------|
| PLS                | 2.2786       |
| Subset             | 0.0493       |
| PCR                | 37.3691      |
| LASSO              | 150.2326     |
| Ridge              | 1.0285       |
| Stepwise           | **0.0108**   |
| Linear Regression  | 0.0495       |

#### Average MSE Across Monte Carlo Simulations

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

- The **Stepwise Regression Model** had the lowest MSE in the initial evaluation (0.0108), indicating high prediction accuracy.  
- The **Best Subset Model** showed the lowest **average MSE** during Monte Carlo simulation (0.0567), suggesting strong consistency across resamples.  
- Regularized models (Ridge, LASSO) were less effective in this case, with LASSO especially underperforming, likely due to multicollinearity and lack of dominant sparse features.

These results show that variable selection-based linear models can outperform penalized methods in clean, well-structured datasets with few noisy predictors.

---

## **8. Tools Used**

- **Language**: R  
- **Packages**: `ggplot2`, `dplyr`, `glmnet`, `pls`, `leaps`  
- **Model Types**: Full Linear, Subset, Stepwise (AIC), Ridge, LASSO, PCR, PLS  
- **Validation**: Monte Carlo simulations  
- **Reporting**: R Markdown and inline commentary  
- **Visuals**: Scatter plots, residual diagnostics, histograms, and Q-Q plots saved in the `/figures` directory

---

## **9. Business Impact / Use Case**

Accurately estimating body fat percentage has wide applications in:

- **Personalized fitness tracking**  
- **Clinical health monitoring**  
- **Military and athletic readiness screening**  
- **Insurance underwriting and risk evaluation**

This project demonstrates how interpretable regression models can be applied to derive reliable body composition insights, avoiding the need for expensive or invasive measurement procedures.
