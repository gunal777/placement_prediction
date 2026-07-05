from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("model/placement_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "gender": request.form["gender"],
        "age": int(request.form["age"]),
        "city_tier": request.form["city_tier"],
        "ssc_percentage": float(request.form["ssc_percentage"]),
        "ssc_board": request.form["ssc_board"],
        "hsc_percentage": float(request.form["hsc_percentage"]),
        "hsc_board": request.form["hsc_board"],
        "hsc_stream": request.form["hsc_stream"],
        "degree_percentage": float(request.form["degree_percentage"]),
        "degree_field": request.form["degree_field"],
        "mba_percentage": float(request.form["mba_percentage"]),
        "internships_count": int(request.form["internships_count"]),
        "projects_count": int(request.form["projects_count"]),
        "certifications_count": int(request.form["certifications_count"]),
        "technical_skills_score": float(request.form["technical_skills_score"]),
        "soft_skills_score": float(request.form["soft_skills_score"]),
        "aptitude_score": float(request.form["aptitude_score"]),
        "communication_score": float(request.form["communication_score"]),
        "work_experience_months": int(request.form["work_experience_months"]),
        "leadership_roles": int(request.form["leadership_roles"]),
        "extracurricular_activities": int(request.form["extracurricular_activities"]),
        "backlogs": int(request.form["backlogs"])
    }

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    result = "Likely to be Placed ✅" if prediction == 1 else "Not Likely to be Placed ❌"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)