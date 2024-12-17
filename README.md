# **EDA and Feature Engineering for Classification Models**

This repository provides a comprehensive exploratory data analysis (EDA) and feature engineering pipeline, focusing on preprocessing, feature scaling, encoding techniques, and feature selection for classification tasks. The dataset used is the "Adult Dataset," which predicts income categories.

---

## **Project Highlights**
1. **Data Exploration and Preprocessing**
   - Initial exploration of data shape, types, and descriptive statistics.
   - Detection of numerical and categorical features.
   - Scaling numerical data using:
     - Min-Max Scaling
     - Standard Scaling

2. **Encoding Techniques**
   - One-Hot Encoding for categorical variables with fewer than 5 categories.
   - Label Encoding for categorical variables with more than 5 categories.
   - A detailed comparison of the pros and cons of both techniques.

3. **Feature Engineering**
   - Creation of new features:
     - `Net_Capital_Gain`: Combines `capital_gain` and `capital_loss` to represent net financial benefits.
     - `Work_hour_category`: Categorizes `hours_per_week` into "Part-Time," "Full-Time," and "Overtime."
   - Transformation of skewed numerical features for improved modeling.

4. **Feature Selection**
   - Outlier detection and removal using the **Isolation Forest** algorithm.
   - Use of **Predictive Power Score (PPS)** for understanding feature relationships and comparing findings with the correlation matrix.

---

## **Key Methods and Libraries**
- **Python Libraries:**
  - `pandas` and `numpy` for data manipulation.
  - `seaborn` and `matplotlib` for data visualization.
  - `sklearn` for scaling, encoding, and outlier detection.
  - `ppscore` for calculating predictive power scores.
  
- **Algorithms Used:**
  - Logistic Regression, Decision Tree, Random Forest, and Support Vector Classifier (discussed for future modeling).

---

## **File Overview**
- `EDA_2.py`: The main script that performs:
  - Data preprocessing (scaling and encoding).
  - Feature engineering.
  - Outlier detection and removal.
  - Visualization of relationships using PPS and correlation matrices.

---

