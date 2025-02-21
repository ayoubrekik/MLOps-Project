from flask import request, render_template, jsonify
import joblib
import numpy as np
from flask import request, jsonify
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import pandas as pd
import pickle
from joblib import dump
from model_pipeline import retraine
#modelneural = joblib.load("model_NN.joblib")
scaler = joblib.load("scaler.pkl")


def configure_routes(app):
    @app.route("/")
    def default():
        return render_template("home.html")

    @app.route("/home")
    def home():
        return render_template("home.html")

    @app.route("/team")
    def team():
        return render_template("team.html")
    @app.route("/hyperparam")
    def hyperparam():
        return render_template("hyperparam.html")

    @app.route("/neuralnetwork")
    def neural_open():
        return render_template("neural.html")

    @app.route("/predictneural", methods=["POST"])
    def predictneural():
        selected_model = request.form.get("model")  # Assuming 'model' is passed in the form data
        international_plan = int(request.form["international_plan"])
        international_plan = int(request.form["international_plan"])
        number_vmail_messages = float(request.form["number_vmail_messages"])
        total_day_minutes = float(request.form["total_day_minutes"])
        total_eve_minutes = float(request.form["total_eve_minutes"])
        total_night_minutes = float(request.form["total_night_minutes"])
        total_intl_minutes = float(request.form["total_intl_minutes"])
        total_intl_calls = int(request.form["total_intl_calls"])
        customer_service_calls = int(request.form["customer_service_calls"])

        data = {
            "International plan": [international_plan],
            "Number vmail messages": [number_vmail_messages],
            "Total day minutes": [total_day_minutes],
            "Total eve minutes": [total_eve_minutes],
            "Total night minutes": [total_night_minutes],
            "Total intl minutes": [total_intl_minutes],
            "Total intl calls": [total_intl_calls],
            "Customer service calls": [customer_service_calls],
        }

        input_df = pd.DataFrame(data)

        input_df = scaler.transform(input_df)
        model = joblib.load(selected_model)
        prediction = model.predict(input_df)

        if prediction == 0:
            result = "The customer will not churn."
        else:
            result = "The customer is likely to churn."
        return jsonify({"result": result})


    # Assume x_train_st, x_test_st, y_train, y_test are already defined
    @app.route('/retrain', methods=['POST'])
    def retrain():
        # Extract hyperparameters from form data
        hidden_layers = tuple(map(int, request.form.get("hidden_layers", "100").split(",")))
        activation = request.form.get("activation", "relu")
        solver = request.form.get("solver", "adam")
        alpha = float(request.form.get("alpha", 0.0001))
        max_iter = int(request.form.get("max_iter", 200))
        random_state = int(request.form.get("random_state", 42))

        # Call the retrain function
        accuracy, precision, recall, f1 = retraine(
            hidden_layers, activation, solver, alpha, max_iter, random_state
        )

        # Return response
        return jsonify({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
