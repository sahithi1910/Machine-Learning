# Install first if needed:
# pip install lime

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import StackingClassifier

from lime.lime_tabular import LimeTabularExplainer

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("Feature_data.csv")
X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 2. Define Base Models
# -------------------------------
base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

# -------------------------------
# 3. Define Stacking Model
# -------------------------------
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)

# -------------------------------
# 4. Create Pipeline
# -------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('stacking', stack_model)
])

# -------------------------------
# 5. Train Model
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate
# -------------------------------
accuracy = pipeline.score(X_test, y_test)
print("Accuracy:", accuracy)

# -------------------------------
# 7. LIME Explainer
# -------------------------------
explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['Malignant', 'Benign'],
    discretize_continuous=True
)

# Choose one test sample
i = 5

# Explain prediction
exp = explainer.explain_instance(
    X_test[i],
    pipeline.predict_proba,
    num_features=10
)

# Show explanation
exp.show_in_notebook(show_table=True)

# Save explanation (optional)
exp.save_to_file("lime_explanation.html")
