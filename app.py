from pathlib import Path
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).parent / "academic_advisor_model.pkl"

with MODEL_PATH.open("rb") as file:
    model = pickle.load(file)

ADVICE = {
    "At Risk": "Speak with your course adviser, attend classes consistently, and make a weekly recovery plan for difficult courses.",
    "Needs Improvement": "Increase focused study time, submit assignments early, and ask questions about topics you do not understand.",
    "Good": "Keep your current study routine, revise consistently, and identify one or two areas where you can improve.",
    "Excellent": "Maintain your habits, help peers when appropriate, and set challenging but realistic academic goals."
}

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            cgpa = float(request.form["cgpa"])
            attendance = float(request.form["attendance"])
            study_hours = float(request.form["study_hours"])
            assignment_score = float(request.form["assignment_score"])

            if not (0 <= cgpa <= 5):
                raise ValueError("CGPA must be between 0 and 5.")
            if not (0 <= attendance <= 100 and 0 <= assignment_score <= 100):
                raise ValueError("Attendance and assignment score must be between 0 and 100.")
            if not (0 <= study_hours <= 24):
                raise ValueError("Study hours must be between 0 and 24.")

            features = np.array([[cgpa, attendance, study_hours, assignment_score]])
            prediction = model.predict(features)[0]
            result = {"category": prediction, "advice": ADVICE.get(prediction, "Discuss your progress with your academic adviser.")}
        except ValueError as exc:
            error = str(exc) or "Please enter valid numeric values."
        except (KeyError, TypeError):
            error = "Please complete every field with a valid number."

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
