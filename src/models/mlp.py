import torch.nn as nn

__all__ = ["MLP_NN", "MLP_RESIDUAL_NN"]

class MLP_NN(nn.Module):
    def __init__(self, embedding, layers, num_classes, dropout_prob=0.25):
        super(MLP_NN, self).__init__()
        self.layers = layers

        # first linear layer
        self.fc1 = nn.Linear(embedding, self.layers[0], bias=True)
        self.bn1 = nn.BatchNorm1d(self.layers[0])
        self.relu = nn.ReLU()

        # hidden layers
        self.layers_mods = nn.ModuleList([])
        for i in range(1, len(layers)):
            self.layers_mods.append(nn.Linear(self.layers[i - 1], self.layers[i], bias=True))
            self.layers_mods.append(nn.BatchNorm1d(self.layers[i]))
            self.layers_mods.append(nn.ReLU())
            if i == len(layers) - 1:
                self.layers_mods.append(nn.Dropout(dropout_prob))

        # final classifier
        self.out = nn.Linear(self.layers[-1], out_features=num_classes, bias=True)


    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        for layer in self.layers_mods:
            x = layer(x)
        x = self.out(x)
        return x


class MLP_RESIDUAL_NN(nn.Module):
    def __init__(self, embedding, layers_residual, layers, num_classes, dropout_prob=0.25):
        super(MLP_RESIDUAL_NN, self).__init__()
        self.layers = layers
        self.layers_residual = layers_residual

        # first linear layer
        self.fc1 = nn.Linear(embedding, self.layers_residual[0], bias=True)
        self.bn1 = nn.BatchNorm1d(self.layers_residual[0])
        self.relu = nn.ReLU()

        # hidden residual layers
        self.layers_mod_residual = nn.ModuleList([])
        for i in range(1, len(layers_residual)):
            self.layers_mod_residual.append(nn.Linear(self.layers_residual[i - 1], self.layers_residual[i], bias=True))
            self.layers_mod_residual.append(nn.BatchNorm1d(self.layers_residual[i]))
            self.layers_mod_residual.append(nn.ReLU())

        # hidden layers
        self.layers_mods = nn.ModuleList([])
        self.fc2 = nn.Linear(self.layers_residual[-1], self.layers[0], bias=True)
        self.bn2 = nn.BatchNorm1d(self.layers[0])
        self.relu2 = nn.ReLU()
        for i in range(1, len(layers)):
            self.layers_mods.append(nn.Linear(self.layers[i - 1], self.layers[i], bias=True))
            self.layers_mods.append(nn.BatchNorm1d(self.layers[i]))
            self.layers_mods.append(nn.ReLU())
            if i == len(layers) - 1:
                self.layers_mods.append(nn.Dropout(dropout_prob))

        # final classifier
        self.out = nn.Linear(self.layers[-1], out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        for i in range(0, len(self.layers_mod_residual), 3):
            identity = x
            out = self.layers_mod_residual[i](identity)
            out = self.layers_mod_residual[i + 1](out)
            out = self.layers_mod_residual[i + 2](out)
            x = out + identity

        x = self.relu2(self.bn2(self.fc2(x)))
        for layer in self.layers_mods:
            x = layer(x)
        x = self.out(x)
        return x
