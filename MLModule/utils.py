import csv
import numpy as np
from MLModule.model import NeuralNetwork


def compare_predict_and_expected(
    predicts: list[list[float]], expected: list[list[float]]
) -> list[bool]:

    results = []
    for pred, exp in zip(predicts, expected):
        res = pred.index(max(pred))
        expected_res = exp.index(max(exp))
        results.append(res == expected_res)

    return results


def print_result(
    model: NeuralNetwork, Xs: list[list[float]], ys: list[list[float]]
) -> None:
    results = []
    for x, y in zip(Xs, ys):
        pred = model.predict(x)
        res = pred.index(max(pred))
        expected_res = y.index(max(y))
        results.append(res == expected_res)

    print(
        f"{sum(results)} / {len(results)} corrects, so {round(sum(results) * 100 / len(results),2)} % accuracy"
    )
    return None


# work only for iris.csv for now
def get_data_from_csv(
    csvpath: str = "data/iris.csv", header=False, label_idx: int = -1
) -> tuple[list[list[float]], list[list[float]]]:

    Xs, ys = [], []

    class_set = set()
    with open(csvpath, "r") as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if header:
                header = False
            else:
                features = list(map(float, row[:-1]))
                Xs.append(features)
                class_label = row[label_idx]

                class_set.add(class_label)
                ys.append(class_label)

    class_mapping = {}
    idx = 0
    for label in class_set:
        res = []
        for i in range(len(class_set)):
            res.append(0 if i != idx else 1)
        class_mapping[label] = res
        idx += 1
    ys = list(map(lambda x: class_mapping[x], ys))

    return Xs, ys


def reduce_dataset(
    Xs: list[list[float]], ys: list[list[float]], reduce_rate=0.75
) -> tuple[list[list[float]], list[list[float]]]:
    if reduce_rate <= 0 or reduce_rate > 1:
        reduce_rate = 0.75
    nb_datapoint_wanted = int(len(Xs) * reduce_rate)

    indices = np.random.choice(len(Xs), size=nb_datapoint_wanted, replace=False)
    Xs_train_small = [Xs[i] for i in indices]
    ys_train_small = [ys[i] for i in indices]

    return Xs_train_small, ys_train_small


def print_confusion_matrix(
    model: NeuralNetwork, full_Xs: list[list[float]], full_ys: list[list[float]]
):

    corrects = 0
    nb_possible_output = len(full_ys[0])
    matrix = []
    for i in range(nb_possible_output):
        matrix.append([0] * nb_possible_output)

    for x, y in zip(full_Xs, full_ys):
        pred_array = model.predict(x)
        predicted_output = pred_array.index(max(pred_array))
        expected_output = y.index(max(y))
        matrix[expected_output][predicted_output] += 1
        corrects += expected_output == predicted_output

    print("Expected (---) / Predicted (|||)")
    for line in matrix:
        print(line)

    idx = 0
    for line in matrix:
        print(f"{idx} : {round(line[idx] * 100 / sum(line), 2)} %")
        idx += 1

    print(f"Overall accuracy : {round(corrects * 100 / len(full_ys),2)} %")
