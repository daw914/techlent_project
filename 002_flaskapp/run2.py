from flask import Flask, jsonify, request, render_template, redirect, url_for, session
import numpy as np
import pickle
import pandas as pd
import shap
import io
from trs import Transformer

# Load the pre-trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect form inputs
        gender = request.form["gender"]
        SeniorCitizen = request.form["SeniorCitizen"]
        Partner = request.form["Partner"]
        Dependents = request.form["Dependents"]
        tenure = request.form["tenure"]
        PhoneService = request.form["PhoneService"]
        MultipleLines = request.form["MultipleLines"]
        InternetService = request.form["InternetService"]
        OnlineSecurity = request.form["OnlineSecurity"]
        OnlineBackup = request.form["OnlineBackup"]
        DeviceProtection = request.form["DeviceProtection"]
        TechSupport = request.form["TechSupport"]
        StreamingTV = request.form["StreamingTV"]
        StreamingMovies = request.form["StreamingMovies"]
        Contract = request.form["Contract"]
        PaperlessBilling = request.form["PaperlessBilling"]
        PaymentMethod = request.form["PaymentMethod"]
        MonthlyCharges = request.form["MonthlyCharges"]
        TotalCharges = request.form["TotalCharges"]

        # Prepare the input for the model
        input_data = np.array([[str(gender), float(SeniorCitizen), str(Partner), str(Dependents), float(tenure), 
                                str(PhoneService), str(MultipleLines), str(InternetService), str(OnlineSecurity),
                                str(OnlineBackup), str(DeviceProtection), str(TechSupport), str(StreamingTV),
                                str(StreamingMovies), str(Contract), str(PaperlessBilling), str(PaymentMethod),
                                float(MonthlyCharges), float(TotalCharges)]])

        # Convert input to DataFrame
        features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                    'MonthlyCharges', 'TotalCharges']
        X = pd.DataFrame(input_data, columns=features)

        # Make prediction
        pred = model.predict_proba(X)[0][1]
        print(f"Prediction: {pred}")  # Debugging output

        # Get the tree-based model (e.g., the model pipeline’s 'cat' step)
        tree_model = model.named_steps['cat']

        # Initialize SHAP TreeExplainer
        explainer = shap.TreeExplainer(tree_model)

        # Transform input features as per your pipeline's transformation step
        X_shap = model.named_steps['tf'].transform(X)

        if isinstance(X_shap, pd.DataFrame):
            X_shap = X_shap.to_numpy()

        # Get SHAP values for the first instance
        shap_values = explainer.shap_values(X_shap)

        # Generate SHAP force plot
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            X_shap[0]
        )

        # Save SHAP plot to an HTML string
        buf = io.StringIO()
        shap.save_html(buf, force_plot)
        shap_html = buf.getvalue()

        # Debug output to check if SHAP HTML is generated
        print(f"SHAP HTML: {shap_html[:500]}...")  # Print first 500 chars

        # Instead of redirecting, pass the data directly to render the result page
        return render_template("result.html", pred=pred, shap_html=shap_html)

    return render_template("index.html")


@app.route('/result')
def result():
    # Render the result page (without session)
    return "This should not be accessed directly."


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)