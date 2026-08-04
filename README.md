# 🏥 Medical Insurance Cost Prediction

A machine learning project that predicts medical insurance charges based on personal attributes such as age, BMI, smoking status, region, and family size. The project covers the full ML workflow — exploratory data analysis, feature engineering, model training/evaluation, and deployment as an interactive web app using Streamlit.

## 📌 Project Overview

Health insurers price policies based on risk factors tied to an individual's demographic and lifestyle attributes. This project builds a regression model that estimates a person's annual medical insurance charges given:

- **Age**
- **Sex**
- **BMI (Body Mass Index)**
- **Number of children/dependents**
- **Smoking status**
- **Region** (northeast, northwest, southeast, southwest)

The trained model is served through a simple, user-friendly Streamlit web application that allows anyone to input their details and receive an instant cost estimate.

## 📂 Project Structure

```
├── Medical_Insurance_Cost_Prediction_using_Machine_Learning.ipynb   # EDA, experimentation & model comparison
├── train_model.py                                                    # Final training pipeline & model export
├── app.py                                                             # Streamlit web application
├── insurance_model.pkl                                                # Serialized trained model (generated after running train_model.py)
└── README.md
```

## 🔍 Dataset

The dataset (`insurance.csv`) contains 1,338 records with the following columns:

| Column     | Description                                   |
|------------|------------------------------------------------|
| `age`      | Age of the individual                          |
| `sex`      | Gender (male/female)                           |
| `bmi`      | Body Mass Index                                 |
| `children` | Number of dependents                            |
| `smoker`   | Smoking status (yes/no)                         |
| `region`   | Residential region in the US                    |
| `charges`  | Individual medical insurance cost (target)      |

## 🧪 Exploratory Data Analysis

Performed in the Jupyter notebook, including:
- Distribution analysis of categorical variables (gender count plot)
- Relationship between BMI and charges (regression plot)
- Impact of smoking status on charges (box plot) — smokers show significantly higher and more variable costs
- Correlation analysis between numerical features and the target variable

## ⚙️ Modeling Approach

Multiple models were built and compared to arrive at the final pipeline:

| Model                                            | R² Score |
|---------------------------------------------------|----------|
| Linear Regression (smoker only)                   | ~0.62    |
| Linear Regression (all features, one-hot encoded)  | ~0.75    |
| Linear Regression + Polynomial Features (degree 2) | ~0.85    |
| Ridge Regression + Polynomial Features (test set)  | ~0.86    |

**Final pipeline (`train_model.py`):**
1. **Preprocessing** — `ColumnTransformer` combining:
   - `StandardScaler` on numerical features (`age`, `bmi`, `children`)
   - `OneHotEncoder` on categorical features (`sex`, `smoker`, `region`)
2. **Feature Engineering** — `PolynomialFeatures` (degree 2) to capture non-linear relationships and interaction effects
3. **Model** — `Ridge` regression (alpha=0.1) for regularization and improved generalization
4. All steps are bundled into a single `sklearn.pipeline.Pipeline` and serialized with `joblib` for consistent use at inference time

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn joblib streamlit
```

### 1. Train the Model
Update the CSV path in `train_model.py` to point to your local `insurance.csv`, then run:
```bash
python train_model.py
```
This generates `insurance_model.pkl` in the project directory.

### 2. Run the Web App
```bash
streamlit run app.py
```
Open the local URL provided by Streamlit (typically `http://localhost:8501`) and enter your details to get a predicted insurance cost.

## 🖥️ App Preview

The Streamlit app provides a simple two-column form for:
- Age, Sex, BMI
- Number of children, Region, Smoking status

Clicking **"Predict Cost"** returns an estimated dollar value for annual insurance charges.

## 🛠️ Tech Stack

- **Python**
- **pandas / NumPy** — data manipulation
- **scikit-learn** — preprocessing, pipeline, modeling
- **Matplotlib / Seaborn** — visualization
- **Streamlit** — web app deployment
- **joblib** — model serialization
