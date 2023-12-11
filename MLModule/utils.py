import csv
import numpy as np


def compare_predict_and_expected(predicts: list[list[float]], expected: list[list[float]])-> list[bool]:

    results = []
    for pred, exp in zip(predicts, expected):
        res = pred.index(max(pred))
        expected_res = exp.index(max(exp))
        results.append(res==expected_res)
    
    print(f"{sum(results)} / {len(results)} corrects")
    return results

#work only for iris.csv for now
def get_data_from_csv(csvpath: str = "data/iris.csv")-> (list[list[float]], list[list[float]]):

    Xs, ys = [], []

    class_mapping = {
        'Iris-setosa': [1,0,0], 
        'Iris-versicolor': [0,1,0], 
        'Iris-virginica': [0,0,1]
        }

    with open(csvpath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            features = list(map(float, row[:-1]))
            Xs.append(features)
            class_label = row[-1]
            numeric_label = class_mapping[class_label]
            ys.append(numeric_label)

    return Xs, ys
