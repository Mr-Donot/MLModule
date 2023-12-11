from MLModule.utils import get_data_from_csv
from MLModule.model import NeuralNetwork
from MLModule.display import plot_model_prediction


nb_input = 4
hidden_layers = [8, 4]
nb_output = 3
PATH = "model/iris_3.pkl"
model = NeuralNetwork(nb_input=nb_input,hidden_layers=hidden_layers,nb_output=nb_output)
model.load_weights(path=PATH)


nb_datapoint = 100
low_bound = 0.0
high_bound = 10.0

Xs, ys = get_data_from_csv("data/iris.csv")

cost = model.cost(Xs, ys)
print(f"Initial Cost : {cost}")
nb_training = 10000
#plot_model_prediction(model, Xs, ys)

for i in range(nb_training):
    if (i+1)%(nb_training//10) == 0 : 
        print(f"training {i+1} / {nb_training}")
        cost = model.cost(Xs, ys)
        print(f"Cost : {cost}")
        model.save_weights(path=PATH)
    model.learn2(Xs, ys)

cost = model.cost(Xs, ys)
print(f"Final Cost : {cost}")

plot_model_prediction(model, Xs, ys)

model.save_weights(path=PATH)
