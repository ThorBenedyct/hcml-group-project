hyper_parameter_sets = []

for model in ["SVC", "LogReg"]:
    for class_weight in ["balanced", None]:
        if model == "SVC":
            for kernel in ["linear", "poly", "rbf"]:
                hyper_parameter_sets.append({
                    "model": "SVC",
                    "class_weight": class_weight,
                    "kernel": kernel
                })
        if model == "LogReg":
            hyper_parameter_sets.append({
                "model": "LogReg",
                "class_weight": class_weight,
            })