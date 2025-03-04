import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

ort_session = ort.InferenceSession('voting_ensemble_model.onnx')

app = Flask(__name__)

CORS(app)

@app.route("/")
def home():
    return render_template('index.html') , 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        height = request.form.get('height')
        weight = request.form.get('weight')
        bmi = request.form.get('bmi')
        physical = request.form.get('physical')

        import numpy as np
        input_array = np.array([height,weight,bmi,physical]).astype(np.float32)

        inputs = {ort_session.get_inputs()[0].name: input_array.reshape(1, -1)}

        prediction = ort_session.run(None, inputs)

        return render_template('predict.html',prediction=prediction[0][0]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/line", methods=["POST"])
def predict_line():
    try:
        data = request.get_json()
        height = data['Height']
        weight = data['Weight']
        bmi = data['BMI']
        physical = data['PhysicalActivityLevel']

        import numpy as np
        input_array = np.array([height,weight,bmi,physical]).astype(np.float32)
        inputs = {ort_session.get_inputs()[0].name: input_array.reshape(1, -1)}
        prediction = ort_session.run(None, inputs)

        return jsonify({"prediction": prediction[0][0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
