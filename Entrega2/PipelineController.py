import pandas as pd
import os
from TextPreprocessor import TextPreprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

def readOriginalData(path = os.path.join(".","Entrega2","Data","ODScat_345.xlsx")):
    odsData = pd.read_excel(path)
    odsData["Textos_espanol"] = odsData["Textos_espanol"].str.replace("Ã¡","á")
    odsData["Textos_espanol"] = odsData["Textos_espanol"].str.replace("Ã©","é")
    odsData["Textos_espanol"] = odsData["Textos_espanol"].str.replace("Ã³","ó")
    odsData["Textos_espanol"] = odsData["Textos_espanol"].str.replace("Ãº","ú")
    odsData["Textos_espanol"] = odsData["Textos_espanol"].str.replace("Ã±","ñ")
    odsData["Textos_espanol"] = odsData["Textos_espanol"].str.replace("Ã","í")

    return odsData

def getPipeline():
    pipeline = Pipeline([
        ("Preprocessor", TextPreprocessor()),
        ("model", RandomForestClassifier(random_state=42, criterion="entropy", max_depth=None, n_estimators=1250))
    ])
    return pipeline

def getTrainedPipeline():
    odsData = readOriginalData()

    xTrain, xVal, yTrain, yVal = train_test_split(odsData["Textos_espanol"],odsData["sdg"],test_size=0.2, random_state=42, stratify=odsData["sdg"])
    
    pipeline = getPipeline()

    pipeline.fit(xTrain, yTrain)

    return pipeline

def retrainPipeline(documents, labels):
    pipeline = getPipeline()
    xTrain, xVal, yTrain, yVal = train_test_split(documents, labels, test_size=0.2, random_state=42, stratify=labels)

    pipeline.fit(xTrain, yTrain)

    preds = pipeline.predict(xVal)

    classReport = classification_report(yVal, preds)

    return pipeline, classReport
    

if __name__ == "__main__":
    odsData = readOriginalData()

    xTrain, xVal, yTrain, yVal = train_test_split(odsData["Textos_espanol"],odsData["sdg"],test_size=0.2, random_state=42, stratify=odsData["sdg"])

    pipeline = getPipeline()

    pipeline.fit(xTrain, yTrain)

    preds = pipeline.predict(xVal)
    print(classification_report(yVal, preds))