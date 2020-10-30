import torch
from torch import Tensor, nn
import torch.nn.functional as F
import pytorch_lightning as pl

import src.set_module as sm


class PatchSetsClassification(pl.LightningModule):
    def __init__(self, patch_size, hidden_n, feature_n, output_n):
        super().__init__()

        self.squeeze_from_set = sm.SqueezeFromSet()

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(patch_size ** 2, hidden_n)
        self.linear_2 = nn.Linear(hidden_n, feature_n)

        self.set_pooling = sm.SetPooling('sum')

        self.linear_3 = nn.Linear(feature_n, hidden_n)
        self.linear_4 = nn.Linear(hidden_n, output_n)

    def encode_feature(self, batch_set: sm.BatchSetType) -> Tensor:
        x, index = self.squeeze_from_set(batch_set)

        x = self.flatten(x)

        x = self.linear_1(x).relu()
        x = self.linear_2(x)

        x, index = self.set_pooling(x, index)

        feature = x
        return feature

    def classify_from_feature(self, feature: Tensor) -> Tensor:
        x = self.linear_3(feature).relu()
        output = self.linear_4(x)
        return output

    def forward(self, batch_set: sm.BatchSetType) -> Tensor:
        feature = self.encode_feature(batch_set)
        output = self.classify_from_feature(feature)
        return output


if __name__ == '__main__':
    patch_size = 3
    hidden_n = 64
    feature_n = 8
    output_n = 2

    model = PatchSetsClassification(patch_size, hidden_n, feature_n, output_n)
    x = [
        torch.full([i, *[patch_size] * 2], i, dtype=torch.float32) for i in range(1, 4)
    ]

    y = model(x)

    print('x', *x, '', sep='\n')
    print('y', *y, '', sep='\n')
