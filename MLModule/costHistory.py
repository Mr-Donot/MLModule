def write_new_cost(value, model_name=""):
    file_path = "model/"+model_name+"_costs.txt"
    try:
        with open(file_path, 'a') as file:
            file.write(str(value) + '\n')
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            file.write(str(value) + '\n')


def get_costs(model_name=""):
    file_path = "model/"+model_name+"_costs.txt"
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                return [line.strip() for line in lines]
            else:
                return []
    except FileNotFoundError:
        return []
    
