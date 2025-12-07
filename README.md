# 🚀 Midterm ML Pipeline: Iris Classification Deployment

This repository hosts an end-to-end Machine Learning pipeline focusing on classifying Iris species. It demonstrates key stages of an ML project: **Model Development**, **Experiment Tracking (MLflow)**, and **API Deployment (FastAPI)**.

## 1\. 📂 Project Structure

This structure adheres to standard MLOps practices, separating training/tracking logic from the serving application.

```
.
├── app.py                      # FastAPI application to serve model predictions
├── mlflow_tracking.py          # Executes Task 1 (Training) and Task 2 (MLflow Logging)
├── midterm_training.ipynb      # Jupyter Notebook for initial data exploration (Task 1 Documentation)
├── README.md                   # Project documentation (this file)
├── requirements.txt            # List of all Python dependencies
├── mlruns/                     # MLflow tracking data folder (automatically created)
└── best_model.pkl              # The serialized best-performing model (saved via joblib)
```

-----

## 2\. 🛠️ Setup and Installation

### A. Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### B. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ou-rithy/ai_midterm
    cd ai_midterm
    ```

2.  **Install dependencies:**
    The project relies on `scikit-learn`, `mlflow`, `fastapi`, `uvicorn`, and `joblib`.

    ```bash
    pip install -r requirements.txt
    ```

-----

## 3\. 🧪 Training and Experiment Tracking (Tasks 1 & 2)

The `mlflow_tracking.py` script handles the core ML tasks: data loading (Iris dataset), splitting, training three models (Logistic Regression, SVC, Random Forest), evaluating metrics (Accuracy, F1 Score), and logging everything to MLflow.

### A. Run the Experiments

Execute the tracking script:

```bash
python mlflow_tracking.py
```

  * **Output:** This script trains the models, logs parameters and metrics, and automatically saves the best-performing model (based on the highest **F1 Score**) to the file **`best_model.pkl`**.

### B. View MLflow Results

To compare the performance of the three models and review the hyperparameters logged:

1.  Start the MLflow UI server from the project root directory:
    ```bash
    mlflow ui
    ```
2.  Open your browser and navigate to **`http://localhost:5000`**. You can compare runs side-by-side to determine which model was truly the best.

-----

## 4\. 🌐 Model Deployment (Task 3)

The best model (`best_model.pkl`) is deployed as a high-performance REST API using **FastAPI** for real-time predictions.

### A. Run the API Server

Start the application using Uvicorn. The `--reload` flag is helpful for development.

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### B. Test the Prediction Endpoint

The easiest way to test the API is by using the interactive documentation provided by FastAPI/Swagger UI.

1.  **Access Swagger UI:** Navigate to **`http://127.0.0.1:8000/docs`**.

2.  Find the **POST** endpoint `/predict/` and click **"Try it out"**.

3.  Enter a sample request body representing the four Iris features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).

    **Sample Request Body (for Iris Setosa):**

    ```json
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
    ```

### C. Example Response

The API returns the predicted class name, the numerical label, and the prediction probabilities:

```json
{
  "model_used": "RandomForestClassifier", 
  "prediction_label": 0,
  "predicted_class": "setosa",
  "probabilities": {
    "setosa": 0.985,
    "versicolor": 0.010,
    "virginica": 0.005
  }
}
```

-----

## 5\. 📚 Documentation (Jupyter Notebook)

The `midterm_training.ipynb` notebook serves as the detailed, step-by-step documentation for **Task 1**, showing the initial data loading, exploration, and metric calculations before the logic was moved into the MLflow script.

-----

## 6\. 🤝 Submission Requirements

The following files meet the assignment requirements:

| Requirement | File | Purpose |
| :--- | :--- | :--- |
| Jupyter Notebook Script | `midterm_training.ipynb` | Detailed initial development and evaluation. |
| Python Script MLflow Tracking | `mlflow_tracking.py` | Experimentation, training, and tracking. |
| Python Deployment Code | `app.py` | FastAPI application for serving predictions. |
| Documentation | `README.md` | Provides instructions on running the application. |
