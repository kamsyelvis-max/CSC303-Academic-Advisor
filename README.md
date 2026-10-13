# CSC 303 — Academic Advisor System

A small AI-assisted academic advising prototype built with Python, scikit-learn, Flask, HTML/CSS, Git and GitHub.

## Inputs
- CGPA on a 0–5 scale
- Attendance percentage
- Average study hours per day
- Assignment score percentage

## Output
A predicted academic category and general study advice.

## Run locally
1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Place the trained `academic_advisor_model.pkl` file in the same folder as `app.py`.
4. Run `python app.py`.
5. Open `http://127.0.0.1:5000` in a browser.

## Model note
The Colab notebook uses generated demonstration data with rule-based labels to show the ML workflow. This is a classroom prototype, not a validated model for making real academic decisions.
