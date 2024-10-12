import requests
import PipelineController
from sklearn.model_selection import train_test_split
import json
import numpy as np
import pandas as pd

data = pd.read_excel(r"C:\Users\yo\Documents\Andes\7mo Semestre\BI\P1\Proyecto1-BI\Entrega2\Data\ODScat_345.xlsx")

data = data.to_json()

jsonMsg = json.dumps({"data":data, "targetVariable":"sdg"})


url = "http://127.0.0.1:5000/retrain"

response = requests.post(url, json=jsonMsg)

print(json.loads(response.text)["message"])
