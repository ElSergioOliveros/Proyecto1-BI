from flask import Flask, request, jsonify
import PipelineController
import pandas as pd
import json

app = Flask(__name__)

pipeline = PipelineController.getTrainedPipeline()

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)