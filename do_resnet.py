import run_model
import resnet

model = resnet.ResNet18(3)
run_model.train_and_test(model, "ResNet18", "data_in_matrix.pkl")
