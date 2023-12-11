from random import uniform

def generate_datapoints(nb_datapoint: int, low_bound: float, high_bound: float) -> (list[list[float]], list[list[float]]):
    data_points = []
    for i in range(nb_datapoint):
        data_points.append([uniform(low_bound, high_bound), uniform(low_bound, high_bound)])
    colors = []
    for d in data_points:
        colors.append([[1.0, 0.0], [0.0, 1.0]][d[0]>d[1]])
    return data_points, colors