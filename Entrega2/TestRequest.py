import requests
import PipelineController
from sklearn.model_selection import train_test_split
import json
import numpy as np
import pandas as pd

data = pd.read_csv("documentsVal.csv")

data = data[data.columns[0]].to_json()
url = " http://127.0.0.1:5000/predict"

response = requests.post(url, json=data)

print(np.array(json.loads(response.text)))