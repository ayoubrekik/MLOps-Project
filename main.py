import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    improve_model,
    save_model,
    load_model,
    log_system_metrics,
    log_requirements,
    log_data,
)


def main(args):
    X_train_st, X_test_st, y_train, y_test = None, None, None, None
    model = None
    mlflow.set_tracking_uri("http://localhost:5001")

    if args.train_model or args.evaluate_model or args.improve_model:
        # Load datasets
        try:
            X_train_st = pd.read_csv("X_train.csv").values
            X_test_st = pd.read_csv("X_test.csv").values
            y_train = pd.read_csv("y_train.csv").values
            y_test = pd.read_csv("y_test.csv").values
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return

        # Check if datasets are loaded
        if X_train_st is None or X_test_st is None or y_train is None or y_test is None:
            print("Error: One or more datasets could not be loaded.")

    # Prepare the data when requested
    if args.prepare_data:
        with mlflow.start_run(
            run_name="Preparing data", log_system_metrics=True
        ) as run:
            run_id = run.info.run_id
            print("Preparing the data...")
            prepare_data()

            # Log system metrics
            # log_system_metrics()

            # Log requirements.txt
            log_requirements()

            # Log dataset (update with actual data file path)
            data_file_path = (
                "data/dataset.csv"  # Change this to your actual data file path
            )
            log_data("X_train.csv")
            log_data("X_test.csv")
            log_data("y_train.csv")
            log_data("y_test.csv")
            print("Data prepared Successfully!")

    # Train the model
    if args.train_model:
        # Check if datasets are loaded
        if X_train_st is None or X_test_st is None or y_train is None or y_test is None:
            print("Error: One or more datasets could not be loaded.")
        else:
            print("Training the model...")
            with mlflow.start_run(
                run_name="Training model", log_system_metrics=True
            ) as run:
                run_id = run.info.run_id
                train_model(X_train_st, y_train)
                # log_system_metrics()
                print(run.info)  # Prints metadata about the run
                model_uri = f"runs:/{run_id}/model"
                model_name = "Churn_Prediction_Model"
                mlflow.register_model(model_uri, model_name)
                print(f"Model registered as '{model_name}'.")
                # run.run
    # Load a saved model
    if args.loaded_model or args.evaluate_model:
        model = load_model(None)
    # Evaluate the model
    if args.evaluate_model:
        if model is None:
            print("Error: Model has not been trained. Please train the model first.")
            return

        # Check if datasets are loaded
        if X_test_st is None or y_test is None:
            print("Error: One or more datasets could not be loaded.")
        else:
            print("Evaluating the model...")
            with mlflow.start_run(
                run_name="Evaluation", log_system_metrics=True
            ) as run:
                run_id = run.info.run_id
                # log_system_metrics()
                evaluate_model(model, X_test_st, y_test)

    # Improve the model
    if args.improve_model:
        if X_train_st is None or X_test_st is None or y_train is None:
            print(
                "Error: Data not available for improvement. Please train the model first."
            )
            return
        print("Improving the model...")

        best_model = improve_model(X_train_st, y_train)
        with mlflow.start_run(
            run_name="Improving model", log_system_metrics=True
        ) as run:
            run_id = run.info.run_id
            # log_system_metrics()
            evaluate_model(best_model, X_test_st, y_test)
            model_uri = f"runs:/{run_id}/model"
            model_name = "Churn_Prediction_Model"
            mlflow.register_model(model_uri, model_name)
        print(f"Model registered as '{model_name}'.")
    # Save model if requested
    if args.save_model:
        print("Saving the best improved model...")
        save_model(best_model)

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Pipeline Automation")
    parser.add_argument("--prepare_data", action="store_true", help="Prepare the data")
    parser.add_argument("--train_model", action="store_true", help="Train the model")
    parser.add_argument(
        "--evaluate_model", action="store_true", help="Evaluate the model"
    )
    parser.add_argument(
        "--improve_model", action="store_true", help="Improve the model"
    )
    parser.add_argument("--save_model", action="store_true", help="Save the model")
    parser.add_argument("--loaded_model", action="store_true", help="Load the model")

    args = parser.parse_args()
    main(args)
