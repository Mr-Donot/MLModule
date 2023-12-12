import matplotlib.pyplot as plt
from MLModule.model import NeuralNetwork
from MLModule.utils import compare_predict_and_expected
from MLModule.costHistory import get_costs

def plot_datapoints(data_points, colors):
    # Extraire les coordonnées x et y des datapoints
    x_values = [item[0] for item in data_points]
    y_values = [item[1] for item in data_points]
    colors = list(map(lambda x: "#0000ff" if x[0]==1 else "#ff0000", colors))
    # Créer un plot avec des couleurs associées
    plt.scatter(x_values, y_values, c=colors, marker='o')

    # Ajouter des labels et un titre
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Datapoints with Colors')

    # Afficher le plot
    plt.show()


def plot_model_prediction(model : NeuralNetwork, Xs :list[list[float]], ys: list[list[float]]):

    predictions = []
    for x in Xs:
        predict = model.predict(x)
        predictions.append(predict)
    
    comparaison = compare_predict_and_expected(predictions, ys)
    colors_values = []
    valid_color = "#00aa00"
    possible_colors = ["#0000ff", "#ff0000", "#ffff00", "#ff00ff", "#00ffff"]
    for c in range(len(comparaison)) :
        if comparaison[c] :
            colors_values.append(valid_color)
        else:
            colors_values.append(possible_colors[ys[c].index(1)])
            


    x_values = [item[0] for item in Xs]
    y_values = [item[1] for item in Xs]

    # Créer un plot avec des couleurs associées
    plt.scatter(x_values, y_values, c=colors_values, marker='o')

    # Ajouter des labels et un titre
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Datapoints with Colors')

    # Afficher le plot
    plt.show()


def plot_cost_history(model_name=""):
    
    cost_values = list(map(float,get_costs(model_name)))

    plt.plot([i for i in range(len(cost_values))], cost_values)
    plt.show()
    

