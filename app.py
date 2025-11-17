import os

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
model_path = os.path.join("model", "mental_load_model.pkl")
model = joblib.load(model_path)


def level_to_text(level: int) -> str:
    """Map numeric level to a short text message."""
    if level == 0:
        return "Low load – your day looks light and manageable."
    if level == 1:
        return "Balanced – you seem in a good spot to study normally."
    if level == 2:
        return "High load – try to add more breaks and reduce pressure."
    return "Overloaded – consider resting and reducing tasks for today."


@app.route("/predict", methods=["POST"])
def predict():
    """Predict mental load level from JSON input."""
    data = request.get_json(force=True)

    try:
        features = [
            float(data["sleep_hours"]),
            float(data["study_hours"]),
            float(data["screen_time_hours"]),
            float(data["caffeine_cups"]),
            float(data["stress_level_1_5"]),
            float(data["mood_level_1_5"]),
            float(data["exercise_today"]),
            float(data["heavy_tasks_today"]),
        ]
    except KeyError as exc:
        return jsonify({"error": f"Missing field: {exc}"}), 400
    except ValueError:
        return jsonify({"error": "All fields must be numeric."}), 400

    x = np.array(features).reshape(1, -1)
    pred = model.predict(x)[0]

    return jsonify(
        {
            "level": int(pred),
            "message": level_to_text(int(pred)),
        }
    )


if __name__ == "__main__":
    # Run local development server
    app.run(debug=True)
