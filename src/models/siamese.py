import torch
import torch.nn as nn

__all__ = ["SiameseResidualNetwork", "SiameseNetwork"]

class SiameseNetwork(nn.Module):
    def __init__(self, num_spectra, layers, embedding_size, dropout_prob=0.25, len_spectrum=1000,
                 dropout_final_layer_one=False, noise_std=1e-6):
        super(SiameseNetwork, self).__init__()
        self.num_spectra = num_spectra
        self.layers = layers

        # Concatenate all spectrums together to have dim `num_spectra * len_spectrum`
        print("Spectra ", self.num_spectra * len_spectrum)
        self.fc1 = nn.Linear(self.num_spectra * len_spectrum, self.layers[0])
        self.bn1 = nn.BatchNorm1d(self.layers[0])
        self.relu = nn.ReLU()  # nn.LeakyReLU()

        self.layers_mods = nn.ModuleList([])
        for i in range(1, len(layers)):
            self.layers_mods.append(nn.Linear(self.layers[i - 1], self.layers[i]))
            self.layers_mods.append(nn.BatchNorm1d(self.layers[i]))
            r"""
            Use Parametetric RELU because it can be identity at the beginning nn.PReLU(init=1.0)

            Use RelU if you already knowits  
            """
            self.layers_mods.append(nn.ReLU())  # nn.LeakyReLU())
            if not dropout_final_layer_one:
                self.layers_mods.append(nn.Dropout(dropout_prob))  # 0.5))
            else:
                # Only apply dropout right befor ethe final layer.
                if i == len(layers) - 1:
                    self.layers_mods.append(nn.Dropout(dropout_prob))
        self.out = nn.Linear(self.layers[-1], out_features=embedding_size)

        # now initialize
        self._init_weights(noise_std)

    def _init_weights(self, noise_std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 1) if square, do exact identity + tiny noise
                if m.weight.size(0) == m.weight.size(1):
                    nn.init.zeros_(m.weight)
                    m.weight.data += torch.randn_like(m.weight) * noise_std
                else:
                    # 2) otherwise orthogonal (norm preserving)
                    raise RuntimeError("Siamese Neural network should be identical")
                # zero bias
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                # BN start with weight=1, bias=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_once(self, x):
        # X has shape (B, C, 1000) B=batch-size, C=numb of spectra
        x = x.view(x.size(0), x.size(1) * x.size(2)) # Shape (B, C*1000)
        x = self.relu(self.bn1(self.fc1(x)))
        for layer in self.layers_mods:
            x = layer(x)
        x = self.out(x)      # (B, 1)
        # x = x.squeeze()      #(B,)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2


class SiameseResidualNetwork(nn.Module):
    def __init__(self, num_spectra, layers, embedding_size, dropout_prob=0.25, len_spectrum=1000,
                 dropout_final_layer_one=False, noise_std=1e-6):
        super(SiameseResidualNetwork, self).__init__()
        self.num_spectra = num_spectra
        self.layers = layers

        # Concatenate all spectrums together to have dim `num_spectra * len_spectrum`
        print("Spectra ", self.num_spectra * len_spectrum)
        self.fc1 = nn.Linear(self.num_spectra * len_spectrum, self.layers[0])
        self.bn1 = nn.BatchNorm1d(self.layers[0])
        self.relu = nn.ReLU()  # nn.LeakyReLU()

        self.layers_mods = nn.ModuleList([])
        for i in range(1, len(layers)):
            self.layers_mods.append(nn.Linear(self.layers[i - 1], self.layers[i]))
            self.layers_mods.append(nn.BatchNorm1d(self.layers[i]))
            r"""
            Use Parametetric RELU because it can be identity at the beginning nn.PReLU(init=1.0)

            Use RelU if you already knowits  
            """
            self.layers_mods.append(nn.ReLU())  # nn.LeakyReLU())
            if not dropout_final_layer_one:
                self.layers_mods.append(nn.Dropout(dropout_prob))  # 0.5))
            else:
                # Only apply dropout right befor ethe final layer.
                if i == len(layers) - 1:
                    self.layers_mods.append(nn.Dropout(dropout_prob))
        self.out = nn.Linear(self.layers[-1], out_features=embedding_size)

        # now initialize
        self._init_weights(noise_std)

    def _init_weights(self, noise_std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 1) if square, do exact identity + tiny noise
                print(m.weight.size(0), m.weight.size(1))
                if m.weight.size(0) == m.weight.size(1):
                    nn.init.eye_(m.weight)
                    m.weight.data += torch.randn_like(m.weight) * noise_std
                else:
                    # 2) otherwise orthogonal (norm preserving)
                    raise RuntimeError("Siamese Neural network should be identical")
                    nn.init.orthogonal_(m.weight, gain=1.0)
                # zero bias
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                # BN start with weight=1, bias=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # def forward_once(self, x):
    #     # X has shape (B, C, 1000) B=batch-size, C=numb of spectra
    #     x = x.view(x.size(0), x.size(1) * x.size(2)) # Shape (B, C*1000)
    #     x = self.relu(self.bn1(self.fc1(x)))
    #     for layer in self.layers_mods:
    #         x = layer(x)
    #     x = self.out(x)      # (B, 1)
    #     # x = x.squeeze()      #(B,)
    #     return x

    def forward_once(self, x):
        # x: shape [B, C, 1000] => flatten to [B, C*1000]
        x = x.view(x.size(0), -1)  # (batch_size, input_dim)

        # -- Residual block 1 (fc1 + bn1 + relu) --
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Add skip if dimensions match (assumed in this example):
        out = out + identity

        # -- Subsequent residual blocks --
        #   each block has 4 modules: [Linear, BN, ReLU, Dropout]
        for i in range(0, len(self.layers_mods), 4):
            identity = out
            # Linear
            out = self.layers_mods[i](out)
            # BatchNorm
            out = self.layers_mods[i + 1](out)
            # ReLU
            out = self.layers_mods[i + 2](out)
            # Dropout (or Identity)
            out = self.layers_mods[i + 3](out)
            # Add skip
            out = out + identity

        # Final output layer (no skip here, but you could add one if dims matched)
        out = self.out(out)
        return out

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2
