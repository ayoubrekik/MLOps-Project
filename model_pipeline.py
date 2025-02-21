# import the necessary library
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score, precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

mlflow.set_tracking_uri("http://localhost:5001")
def print_red(text):
    print(f"\033[91m{text}\033[00m")


# cap the outliers to be replaced by upper bound or lower bound values


def cap_outliers(data, col_num):

    for col_ in col_num:
        Q1 = data[col_].quantile(0.25)
        Q3 = data[col_].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data[col_] = data[col_].clip(lower=lower_bound, upper=upper_bound)

    return data


# we will define the function to encode columns
# we will use LabelEncoder for the categorical columns


def encode_categorical_features(data):

    encoded_data = data.copy()
    label_encoders = {}

    # Label encode binary features
    binary_features = ["International plan", "Voice mail plan", "Churn"]
    for feature in binary_features:
        le = LabelEncoder()
        encoded_data[feature] = le.fit_transform(encoded_data[feature])
        label_encoders[feature] = le

    le_state = LabelEncoder()
    encoded_data["State"] = le_state.fit_transform(encoded_data["State"])
    label_encoders["State"] = le_state

    return encoded_data, label_encoders


def prepare_data():
    train_data = pd.read_csv("churn-bigml-80.csv")
    test_data = pd.read_csv("churn-bigml-20.csv")
    col_num = [
        "Account length",
        "Number vmail messages",
        "Total day minutes",
        "Total day calls",
        "Total day charge",
        "Total eve minutes",
        "Total eve calls",
        "Total eve charge",
        "Total night minutes",
        "Total night calls",
        "Total night charge",
        "Total intl minutes",
        "Total intl calls",
        "Total intl charge",
        "Customer service calls",
    ]

    # apply the capping to our training dataset
    train_data = cap_outliers(train_data, col_num)
    # apply the capping to our testing dataset
    test_data = cap_outliers(test_data, col_num)

    # we will apply the encoding to the training and testing dataset
    train_data, label_encoders = encode_categorical_features(train_data)
    test_data, label_encoders1 = encode_categorical_features(test_data)

    # we will drop the 4 columns from both the training and testing sets
    train_data = train_data.drop(
        columns=[
            "Total day charge",
            "Total eve charge",
            "Total night charge",
            "Total intl charge",
            "Voice mail plan",
        ]
    )
    test_data = test_data.drop(
        columns=[
            "Total day charge",
            "Total eve charge",
            "Total night charge",
            "Total intl charge",
            "Voice mail plan",
        ]
    )

    # Create X and Y for training set
    X_train = train_data.drop(columns=["Churn"])
    y_train = train_data["Churn"]
    X_test = test_data.drop(columns=["Churn"])
    y_test = test_data["Churn"]

    # Now we will complete the feature selection phase and we will try to identify the most and least important features.
    x_log = sm.add_constant(X_train)
    reg_log = sm.Logit(y_train, x_log)
    results_log = reg_log.fit()

    # Extract p-values and identify significant features (p-value < 0.05)
    significant_features = results_log.pvalues[results_log.pvalues < 0.05].index
    X_train = X_train[significant_features.drop("const")]
    X_test = X_test[significant_features.drop("const")]

    # we will use StandardScaler
    # Now, let's standardize both our X_train and X_test.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_st = scaler.transform(X_train)
    X_test_st = scaler.transform(X_test)
    # Save the datasets as CSV files
    pd.DataFrame(X_train_st, columns=X_train.columns).to_csv("X_train.csv", index=False)
    pd.DataFrame(X_test_st, columns=X_test.columns).to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)



def train_model(X_train_st, y_train):
    # Initialize the model
    mlp = MLPClassifier()
    # Fit the model
    mlp.fit(X_train_st, y_train)
    # Save the model
    save_model(mlp)
    # Log the default hyperparameters to MLflow
    mlflow.log_param("hidden_layer_sizes", mlp.get_params()['hidden_layer_sizes'])
    mlflow.log_param("activation", mlp.get_params()['activation'])
    mlflow.log_param("solver", mlp.get_params()['solver'])
    mlflow.log_param("max_iter", mlp.get_params()['max_iter']) 
    # Log model to MLflow
    mlflow.sklearn.log_model(mlp, "model")


def evaluate_model(model, X_test_st, y_test):
    # Predict the model's output
    y_pred = model.predict(X_test_st)
    
    # Generate confusion matrix
    cm_nn = confusion_matrix(y_test, y_pred)
    # Log the default hyperparameters to MLflow
    mlflow.log_param("hidden_layer_sizes", model.get_params()['hidden_layer_sizes'])
    mlflow.log_param("activation", model.get_params()['activation'])
    mlflow.log_param("solver", model.get_params()['solver'])
    mlflow.log_param("max_iter", model.get_params()['max_iter']) 
    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # Log the confusion matrix as an artifact
    with open("confusion_matrix.txt", "w") as f:
        f.write(str(cm_nn))
    mlflow.log_artifact("confusion_matrix.txt")

    # Log evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')  # Adjust `average` if needed
    recall = recall_score(y_test, y_pred, average='binary')  # Adjust `average` if needed
    f1 = f1_score(y_test, y_pred, average='binary')  # Adjust `average` if needed

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Output for user with red text
    print_red("Confusion Matrix:")
    print(str(cm_nn))

    print_red("Accuracy:")
    print(str(accuracy))

    print_red("Precision:")
    print(str(precision))

    print_red("Recall:")
    print(str(recall))

    print_red("F1 Score:")
    print(str(f1))

def improve_model(X_train_st, y_train):
    param_grid = {
        "hidden_layer_sizes": [(5, 5), (5, 6), (5, 5, 5)],
        "activation": ["logistic", "relu", "tanh"],
        "solver": ["adam", "sgd", "lbfgs"],
        "max_iter": [1000, 2000],  # Include a range of iterations.
        "random_state": [42],  # Multiple values for consistency checks.
        "alpha": [0.0001, 0.001, 0.01, 0.2],  # Expanded range for L2 regularization.
    }
    mlp = MLPClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring="f1"
    )

    grid_search.fit(X_train_st, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best score found: ", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    return best_model


def retraine(hidden_layers, activation, solver, alpha, max_iter, random_state):
    # Load datasets
    try:
        x_train = pd.read_csv("X_train.csv").values
        x_test = pd.read_csv("X_test.csv").values
        y_train = pd.read_csv("y_train.csv").values
        y_test = pd.read_csv("y_test.csv").values
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None  # Exit function if datasets fail to load

    # Initialize and train the model
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        validation_fraction=0.2,
        verbose=False  # Set to True if you want training logs
    )
    model.fit(x_train, y_train)
    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)  
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Log metrics to MLflow
    # Start MLflow logging
    with mlflow.start_run(run_name="Retraining Model"):
        mlflow.log_param("hidden_layer_sizes", model.get_params()["hidden_layer_sizes"])
        mlflow.log_param("activation", model.get_params()["activation"])
        mlflow.log_param("solver", model.get_params()["solver"])
        mlflow.log_param("max_iter", model.get_params()["max_iter"])  
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
    # Save the model
    dump(model, "my_model.joblib")

    # Return response
    return accuracy, precision, recall, f1

def save_model(model):
    # Save the model
    model_name_save = "model_NN.joblib"
    dump(model, model_name_save)
    print("Model saved successfully.")


def load_model():
    # Save the model
    model_path = "model_NN.joblib"
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        loaded_model = load(model_path)
        print("Model loaded successfully.")
        return loaded_model
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return loaded_model
