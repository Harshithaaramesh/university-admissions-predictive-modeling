# 🎓 Predictive Modeling for University Admissions and Enrollment Decisions

A data-driven analytics project built using **R** that applies the **CRISP-DM framework** to predict university admission likelihood, enrollment probability, and expected graduation GPA — providing actionable insights for academic counselors and admissions teams.

> *Course: BUAN 6356 — Business Analytics with R | UT Dallas*

---

## 📌 Table of Contents

- [Overview](#overview)
- [Business Questions](#business-questions)
- [Dataset](#dataset)
- [Models Built](#models-built)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Future Enhancements](#future-enhancements)

---

## Overview

University admissions are increasingly competitive, and students often struggle to evaluate how strong their application is. This project builds interpretable predictive models to assist counselors in providing personalized, data-driven guidance to applicants.

The analysis follows the full **CRISP-DM** lifecycle — from business understanding and data preparation through modeling, evaluation, and deployment strategy.

---

## Business Questions

1. **What is the probability a student will be admitted?**
2. **If admitted, what is the likelihood they will enroll?**
3. **If enrolled, what graduation GPA can the student expect?**

---

## Dataset

17,339 applicant records with the following features:

| Feature | Description |
|---------|-------------|
| HSGPA | High School GPA |
| SAT_ACT | Standardized test score |
| Sex | Gender of applicant |
| Edu_Parent1, Edu_Parent2 | Parental education level |
| White, Asian | Race indicators |
| Admitted | Admission result (Yes/No) |
| Enrolled | Enrollment decision after admission (Yes/No) |
| College_GPA | Final college GPA (for enrolled students only) |

**Data Preparation steps:** missing value imputation, duplicate removal, outlier handling, categorical encoding, and derived variable generation. Cleaned dataset exported as `clean_university_data.csv`.

---

## Models Built

### Admission Prediction
- Logistic Regression
- Classification Tree (pruned using best CP)

### Enrollment Prediction
- Logistic Regression
- Classification Tree (pruned using best CP)

### Graduation GPA Prediction
- Linear Regression (3 models compared)
- Regression Tree (pruned using best CP)

---

## Results Summary

| Task | Model | Performance |
|------|-------|-------------|
| Admission | Logistic Regression | Accuracy: 82.01% |
| Admission | Classification Tree | Accuracy: 85.29% |
| Enrollment | Logistic Regression | Accuracy: 98.46% |
| Enrollment | Classification Tree | Accuracy: 97.78% |
| Graduation GPA | Linear Regression (Model 1) | RMSE: 0.4369, R² = 0.63 |
| Graduation GPA | Regression Tree | RMSE: 0.4735 |

**Key finding:** HSGPA and SAT/ACT are the strongest predictors across all three tasks, accounting for 71% and 25% of variable importance respectively in the GPA regression tree.

---

## Project Structure

```
university-admissions-predictive-modeling/
│
├── data/
│   ├── clean_university_data.csv        # Cleaned dataset
│   └── clean_university_with_AdmitProb.csv  # With predicted probabilities
│
├── scripts/
│   ├── Q1_Logistic_regression_Tree_Admission.R   # Admission models
│   ├── Q2_Logistic_regression_Enrollment.R       # Enrollment models
│   └── Q3_Linear_regression_model.R              # GPA prediction models
│
├── report/
│   └── Group5_Project_Report.pdf        # Full project report
│
└── README.md
```

---

## How to Run

### Prerequisites
- R (version 4.0+)
- RStudio
- Required packages:

```r
install.packages(c("caret", "rpart", "rpart.plot", "gains", "ggplot2", "dplyr"))
```

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Harshithaaramesh/university-admissions-predictive-modeling.git
cd university-admissions-predictive-modeling
```

**2. Open RStudio and set working directory**
```r
setwd("path/to/university-admissions-predictive-modeling")
```

**3. Run scripts in order**
```r
source("scripts/Q1_Logistic_regression_Tree_Admission.R")
source("scripts/Q2_Logistic_regression_Enrollment.R")
source("scripts/Q3_Linear_regression_model.R")
```

---

## Future Enhancements

- Add financial aid, extracurricular, and essay sentiment features
- Implement ensemble methods (Random Forest, XGBoost) for stronger performance
- Build interactive counselor dashboards with what-if simulation tools
- Expand dataset across multiple universities for benchmarking

---

## Author

**Harshitha Bengaluru Rameshbabu**
MS Business Analytics & AI — UT Dallas
Course: BUAN 6356 Business Analytics with R

