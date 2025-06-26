hyper_parameter_sets = []

for model in ["SVC", "LogReg"]:
    for class_weight in ['balanced']:
        if model == "SVC":
            for kernel in ["poly", "rbf"]:
                for gamma in ["auto", "scale", 1/650, 1/600, 1/512, 1/420, 1/400, 1/360]:
                    hyper_parameter_sets.append({
                        "model": "SVC",
                        "class_weight": class_weight,
                        "kernel": kernel,
                        "gamma": gamma,
                    })
        # if model == "LogReg":
        #     hyper_parameter_sets.append({
        #         "model": "LogReg",
        #         "class_weight": class_weight,
        #     })