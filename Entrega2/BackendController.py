from flask import Flask, request, jsonify
import PipelineController
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

if os.path.exists(os.path.join(".", "Entrega2", "Model", "model.pkl")):
    pipeline = joblib.load(os.path.join(".", "Entrega2", "Model", "model.pkl"))
else:
    pipeline = PipelineController.getTrainedPipeline()
    joblib.dump(pipeline, os.path.join(".", "Entrega2", "Model", "model.pkl"))

@app.route("/predict", methods=['POST'])
def predict():
    dataString = request.get_json()
    dataJson = json.loads(dataString)
    
    documents = []
    for key in dataJson.keys():
        documents.append(dataJson[key])

    documents = pd.Series(documents)

    preds = pipeline.predict_proba(documents)

    return jsonify(preds.tolist())


@app.route("/retrain", methods=['POST'])
def retrain():
    dataString = request.get_json()
    dataJson = json.loads(dataString)

    targetVariable = dataJson["targetVariable"]

    df = dataJson["data"]
    df = json.loads(df)
    df = pd.DataFrame(df)

    documentsKey = [key for key in df.keys() if key != targetVariable][0]

    pipeline, classReport = PipelineController.retrainPipeline(df[documentsKey], df[targetVariable])
    joblib.dump(pipeline, os.path.join(".", "Entrega2", "Model", "model.pkl"))

    return {"message": classReport}

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)