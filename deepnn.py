import torch
import torch.nn as nn

class DeepNN(nn.Module):
    def __init__(self, num_layers, num_input, num_output):
        super(DeepNN, self).__init__()
        layers = [nn.Linear(num_input, 256), nn.ReLU()]
        layers += self._make_layers(num_layers)
        self.features = nn.Sequential(*layers)

        classifier_layers = [nn.Linear(256, num_output), nn.Sigmoid()]
        self.classifier = nn.Sequential(*classifier_layers)

    def _make_layers(self, num_layers):
        layers = []
        while num_layers > 0:
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
            num_layers -= 1
        return layers

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out
