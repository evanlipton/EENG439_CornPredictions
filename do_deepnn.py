import run_model
import deepnn

model = deepnn.DeepNN(3, 441, 3)
run_model.train_and_test(model, "DeepNN")
