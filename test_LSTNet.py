import LSTNet

from types import SimpleNamespace


def create_LSTNet(num_input_params, num_classes):
    args = SimpleNamespace(
        hidCNN=100,
        hidRNN=100,
        window=24*7,
        CNN_kernel=6,
        highway_window=24,
        horizon=12,
        skip=24,
        hidSkip=5)
