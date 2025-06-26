hyper_parameter_sets = []

for model in ["SVC", "LogReg"]:
    for class_weight in ["balanced", None]:
        if model == "SVC":
            for kernel in ["rbf"]: #["linear", "poly", "rbf"]
                for gamma in ["auto", "scale", 1/512, 1/400, 1/256, 1/128, 1/75, 1/64]:
                    hyper_parameter_sets.append({
                        "model": "SVC",
                        "class_weight": class_weight,
                        "kernel": kernel,
                        "gamma": gamma,
                    })
        if model == "LogReg":
            hyper_parameter_sets.append({
                "model": "LogReg",
                "class_weight": class_weight,
            })