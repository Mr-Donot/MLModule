from MLModule.model import NeuralNetwork
from MLModule.random_data import generate_datapoints
from MLModule.display import plot_model_prediction

nb_input = 4
hidden_layers = [2, 3]
nb_output = 2
PATH = "model/iris.pkl"
model = NeuralNetwork(nb_input=nb_input,hidden_layers=hidden_layers,nb_output=nb_output)
#model.load_weights(path=PATH)


nb_datapoint = 100
low_bound = 0.0
high_bound = 10.0

Xs, ys = generate_datapoints(nb_datapoint, low_bound, high_bound)

cost = model.cost(Xs, ys)
print(f"Initial Cost : {cost}")
nb_training = 1000
plot_model_prediction(model, Xs, ys)

for i in range(nb_training):
    if (i+1)%100 == 0 : 
        print(f"training {i+1} / {nb_training}")
        plot_model_prediction(model, Xs, ys)
    model.learn(Xs, ys)

cost = model.cost(Xs, ys)
print(f"Final Cost : {cost}")

plot_model_prediction(model, Xs, ys)

model.save_weights(path=PATH)