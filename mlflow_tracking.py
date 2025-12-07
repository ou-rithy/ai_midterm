# Import necessary libraries
import mlflow
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# ... import other models and metrics

# Set an experiment name
mlflow.set_experiment("Midterm_ML_Pipeline_Iris_Classification")

# 1. Load and Split Data (Same as Task 1)
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Define Model Configurations and run MLflow
models_config = [
    {'model': LogisticRegression(max_iter=200, random_state=42), 'params': {'max_iter': 200, 'solver': 'lbfgs'}},
    {'model': SVC(kernel='linear', C=1.0, random_state=42), 'params': {'kernel': 'linear', 'C': 1.0}},
    {'model': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), 'params': {'n_estimators': 100, 'max_depth': 5}}
]

best_f1 = -1
best_model = None

for config in models_config:
    model = config['model']
    
    with mlflow.start_run():
        model_name = type(model).__name__
        mlflow.set_tag("model_type", model_name)

        # Log Parameters
        mlflow.log_params(config['params'])

        # Train and Evaluate (use the function from Task 1, or re-implement here)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log the model (MLflow will handle serialization)
        mlflow.sklearn.log_model(model, "model")
        
        # Check for the best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            
# 3. Save the actual best model object for deployment
if best_model:
    joblib.dump(best_model, "best_model.pkl")
    print(f"Best model ({type(best_model).__name__}) saved to best_model.pkl")

# To run the MLflow UI: Open terminal and type `mlflow ui`