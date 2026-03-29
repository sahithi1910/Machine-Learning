# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import ast

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("Text_coherence.csv")   # change filename if needed

# Convert embedding column from string to list
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Convert to numpy array
X = np.vstack(df['embedding'].values)
y = df['label']

# OPTIONAL: Reduce dataset size if system is slow (uncomment if needed)
# df = df.sample(n=2000, random_state=42)

# =========================
# 3. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4. EVALUATION FUNCTION
# =========================
def evaluate(model):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred, average='weighted'),
        "Recall": recall_score(y_test, test_pred, average='weighted'),
        "F1 Score": f1_score(y_test, test_pred, average='weighted')
    }

# =========================
# 5. SVM (SIMPLIFIED)
# =========================
svm_params = {
    'C': [1, 10],
    'kernel': ['linear']   # lighter than rbf
}

svm_search = RandomizedSearchCV(
    SVC(),
    svm_params,
    n_iter=2,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    random_state=42
)
svm_search.fit(X_train, y_train)

# =========================
# 6. DECISION TREE
# =========================
dt_params = {
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

dt_search = RandomizedSearchCV(
    DecisionTreeClassifier(),
    dt_params,
    n_iter=2,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    random_state=42
)
dt_search.fit(X_train, y_train)

# =========================
# 7. RANDOM FOREST
# =========================
rf_params = {
    'n_estimators': [100],
    'max_depth': [10, None],
    'min_samples_split': [2]
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(),
    rf_params,
    n_iter=2,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    random_state=42
)
rf_search.fit(X_train, y_train)

# =========================
# 8. ADABOOST
# =========================
ada_params = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 1]
}

ada_search = RandomizedSearchCV(
    AdaBoostClassifier(),
    ada_params,
    n_iter=2,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    random_state=42
)
ada_search.fit(X_train, y_train)

# =========================
# 9. MLP
# =========================
mlp_params = {
    'hidden_layer_sizes': [(100,), (50,)],
    'activation': ['relu'],
    'learning_rate_init': [0.001]
}

mlp_search = RandomizedSearchCV(
    MLPClassifier(max_iter=200),
    mlp_params,
    n_iter=2,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    random_state=42
)
mlp_search.fit(X_train, y_train)

# =========================
# 10. NAIVE BAYES
# =========================
nb = GaussianNB()
nb.fit(X_train, y_train)

# =========================
# 11. EVALUATE ALL MODELS
# =========================
results = {}

results['SVM'] = evaluate(svm_search.best_estimator_)
results['Decision Tree'] = evaluate(dt_search.best_estimator_)
results['Random Forest'] = evaluate(rf_search.best_estimator_)
results['AdaBoost'] = evaluate(ada_search.best_estimator_)
results['MLP'] = evaluate(mlp_search.best_estimator_)
results['Naive Bayes'] = evaluate(nb)

# =========================
# 12. FINAL RESULTS TABLE
# =========================
results_df = pd.DataFrame(results).T

print("\n===== FINAL MODEL COMPARISON =====\n")
print(results_df)

# Save results
results_df.to_csv("classification_results.csv")
