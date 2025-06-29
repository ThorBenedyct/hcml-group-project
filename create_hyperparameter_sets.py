hyper_parameter_sets = []


def create_float_range(low, high, step):
    return [{"m": x/100, "f": 1-(x/100)} for x in range(5, 95, 5)]

def class_weight_ranges():
    return [{"m": x/100, "f": 1-(x/100)} for x in range(5, 95, 5)]
# ["scaler", "pca", "scalar+pca", "none"]:


for model in ["SVC", "LogReg"]:

    for normalization in ["scaler", "none"]:
        for class_weight in class_weight_ranges():
            for kernel in ["poly", "rbf"]:
                for gamma in [x/100000 for x in range(10, 200, 10)]:
                    hyper_parameter_sets.append({
                        "model": "SVC",
                        "class_weight": class_weight,
                        "kernel": kernel,
                        "gamma": gamma,
                        "normalization": normalization,
                    })
            # if model == "LogReg":
            #     hyper_parameter_sets.append({
            #         "model": "LogReg",
            #         "class_weight": class_weight,
            #     })