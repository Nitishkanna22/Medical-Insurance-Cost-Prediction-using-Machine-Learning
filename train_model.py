import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib  # This library helps us save the model

# 1. Load your datasetC:\Users\vigee\OneDrive\ドキュメント\Model Projects\insurance.csv"# Make sure the csv file is in the same folder as this script
df = pd.read_csv(r"C:\Users\vigee\OneDrive\ドキュメント\Model Projects\insurance.csv") 

# 2. Separate Features (X) and Target (y)
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# 3. Create a Preprocessing Pipeline
# This automatically handles the "get_dummies" and scaling for us
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# 4. Bundle Preprocessing, Polynomial Features, and Model together
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2)),
    ('classifier', Ridge(alpha=0.1))
])

# 5. Train the model
print("Training the model...")
model_pipeline.fit(X, y)
print(f"Model Accuracy (R2 Score): {model_pipeline.score(X, y):.4f}")

# 6. Save the trained model to a file
joblib.dump(model_pipeline, 'insurance_model.pkl')
print("Model saved as 'insurance_model.pkl'")