import run_model
import deepnn

model = deepnn.DeepNN(1, 442, 3)
run_model.train_and_test(model, "DeepNN")
