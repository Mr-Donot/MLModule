from MLModule.utils import get_data_from_csv
from MLModule.model import NeuralNetwork
from MLModule.display import plot_model_prediction, plot_cost_history
from MLModule.utils import print_result, reduce_dataset


model_name = "mnist1.5k"
DATA_PATH = "data/"+model_name+".csv"


full_Xs, full_ys = get_data_from_csv(DATA_PATH, header=True)

nb_input = len(full_Xs[0])
hidden_layers = [128]
nb_output = len(full_ys[0])


PATH = "model/"+model_name+".pkl"
keeping_track_cost = True

model = NeuralNetwork(
    nb_input=nb_input,
    hidden_layers=hidden_layers,
    nb_output=nb_output,
    name=model_name
    )


nb_datapoint = 100
low_bound = 0.0
high_bound = 10.0




nb_training = 10
nb_to_print = 1
#plot_model_prediction(model, Xs, ys)
cost = None
for i in range(nb_training):
    Xs, ys = reduce_dataset(full_Xs, full_ys, 0.1) #mini-batch technique, to avoid overfitting and to reduce time for each training
    if (i+1)%(nb_to_print) == 0 : 
        print(f"training {i+1} / {nb_training}")
        print(f"Cost : {cost}" if cost is not None else f"Not keeping track of the cost during the training")
        model.save_weights(path=PATH)
    cost = model.learn2(Xs, ys, 0.01, keeping_track_cost, 0.3)

cost = model.cost(Xs, ys)
print(f"Final Cost : {cost}")

print_result(model, Xs, ys)

#plot_model_prediction(model, Xs, ys)

model.save_weights(path=PATH)

plot_cost_history(model_name)
