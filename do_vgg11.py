import run_model
import vgg

model = vgg.VGG('VGG11', 3)
run_model.train_and_test(model, "VGG11", "data_in_matrix.pkl")
