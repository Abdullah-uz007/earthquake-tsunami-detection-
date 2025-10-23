# Earthquake–Tsunami Detection using Machine Learning 🌊🌋

This project aims to predict whether an earthquake is likely to trigger a tsunami using **Machine Learning** techniques. By analyzing seismic and environmental data, the project demonstrates how AI can support early disaster detection and potentially save lives in vulnerable regions.

---

## 🔎 Project Overview

Tsunamis are one of the most devastating natural disasters, often caused by undersea earthquakes. Early detection and prediction of tsunamis can help mitigate loss of life and property. This project uses machine learning models to analyze historical earthquake data and environmental factors to predict the likelihood of a tsunami following an earthquake.

The workflow includes **data collection, preprocessing, exploratory analysis, model building, evaluation, and visualization** of results.

---

## 📊 Dataset

- The dataset for this project was sourced from **Kaggle** for educational purposes.  
- It contains historical earthquake records along with environmental attributes such as **magnitude, depth, location, and water displacement metrics**.  
- **Note:** This repository does not include the raw dataset. Instructions to download it are provided in the notebook.

**Example attributes:**  

| Feature         | Description                                 |
|-----------------|---------------------------------------------|
| `Magnitude`     | Earthquake magnitude on the Richter scale  |
| `Depth`         | Depth of earthquake in kilometers          |
| `Location`      | Coordinates of earthquake epicenter        |
| `Tsunami`       | Target variable indicating tsunami (0/1)  |
| `Other env. features` | e.g., water displacement, pressure, seismic readings |

---

## 🧹 Data Preprocessing

- **Handling missing values:** Removed or imputed missing data to ensure model integrity.  
- **Feature scaling:** Standardized numerical features for better model convergence.  
- **Encoding categorical variables:** Converted text-based categories into numerical format for ML models.  
- **Train-test split:** Data split into training and testing sets to evaluate model performance fairly.

---

## 📈 Exploratory Data Analysis (EDA)

- Visualized distributions of features to detect patterns and anomalies.  
- Examined correlations between earthquake attributes and tsunami occurrence.  
- Used **histograms, boxplots, scatterplots, and heatmaps** to understand feature relationships.  
- Insights from EDA guided feature selection and model design.

---

## 🧠 Model Building

Several machine learning models were trained and compared:  

1. **Random Forest Classifier** ✅ – Achieved the best performance.  
2. Logistic Regression  
3. Support Vector Machine (SVM)  
4. Gradient Boosting  

- Hyperparameters were tuned using **GridSearchCV** and cross-validation.  
- Feature importance from Random Forest highlighted the most predictive variables.  

---

## 📊 Model Evaluation

Models were evaluated using multiple metrics:  

| Metric        | Random Forest Score |
|---------------|------------------|
| Accuracy      | 0.95             |
| Precision     | 0.94             |
| Recall        | 0.93             |
| F1-score      | 0.94             |
| ROC-AUC       | 0.96             |

- The **ROC curve** was plotted to visualize true positive vs false positive trade-offs.  
- Random Forest was selected for deployment due to its **high performance and interpretability**.

---

## 🖼 Visualizations

- Feature distributions and correlations were plotted using **Matplotlib** and **Seaborn**.  
- ROC curves, confusion matrices, and feature importance plots were included to clearly communicate model results.  
- Visualizations are stored in the `images/` folder for easy reference.

---

## 🧰 Tools & Technologies

- **Python** – Main programming language  
- **Pandas & NumPy** – Data manipulation and numerical operations  
- **Scikit-learn** – Machine learning algorithms and evaluation metrics  
- **Matplotlib & Seaborn** – Data visualization  
- **Jupyter Notebook** – Interactive project documentation and execution  

---
