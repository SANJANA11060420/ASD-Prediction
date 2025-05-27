
import json
import sys
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def generate_mock_data():
    
    X = np.random.randint(0, 5, size=(1000, 10))
    y = np.random.randint(0, 5, size=1000)
    
    
    for i in range(1000):
        if X[i, 0] > 3 and X[i, 1] > 3:  
            y[i] = min(4, y[i] + 2)
        if sum(X[i]) > 25: 
            y[i] = 4
    return X, y

def predict_qchat(answers):
    X = np.array([answers])
    
    X_train, y_train = generate_mock_data()
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    prediction = clf.predict(X)[0]
    probas = clf.predict_proba(X)[0]
    
    risk_levels = ["No Risk", "Low Risk", "Medium Risk", "High Risk", "Autistic"]
    
    _, X_test, _, y_test = train_test_split(X_train, y_train, test_size=0.2)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        "prediction": risk_levels[prediction],
        "probability": {risk_levels[i]: float(probas[i]) for i in range(5)},
        "accuracy": f"{accuracy:.1%}",
        "score": sum(answers)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1 and len(sys.argv[1:]) == 10:
        answers = list(map(int, sys.argv[1:]))
    else:
        answers = [1, 2, 0, 1, 0, 2, 1, 0, 1, 3]
    
    answer_dict = {f"q{i+1}": ans for i, ans in enumerate(answers)}
    results = predict_qchat(answers)
    output = {
        "qchatScore": results["score"],
        "answers": answer_dict,
        "svmPrediction": results["prediction"],
        "svmAccuracy": results["accuracy"],
        "riskProbability": results["probability"]
    }
    
    with open("qchat_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to qchat_results.json")