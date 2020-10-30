from typing import Any, Dict, Tuple, List
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data


class Model(pl.LightningModule):
    def __init__(
        self,
        seed: int,
        batch_size: int,
        num_workers: int,
        max_epochs: int,
        min_epochs: int,
        patience: int,
        optimizer: str,
        lr: float,
        image_size: int,
        hidden_n: int,
        output_n: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(image_size ** 2, hidden_n)
        self.linear_2 = nn.Linear(hidden_n, hidden_n)
        self.linear_3 = nn.Linear(hidden_n, output_n)

        self.accuracy = pl.metrics.Accuracy()

    @auto_move_data
    def forward(self, batch_set: Tensor) -> Tensor:
        x = self.flatten(batch_set)
        x = self.linear_1(x).relu()
        x = self.linear_2(x).relu()
        x = self.linear_3(x)
        output = x
        return output

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = getattr(optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr
        )
        return optimizer

    def _step(self, batch: List[Tensor]) -> Dict[str, Any]:
        x, y = batch
        batch_size = len(y)
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        return {'batch_size': batch_size, 'loss': loss, 'accuracy': accuracy}

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        items = self._step(batch)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('train_loss', loss, on_step=True)
        self.log('train_accuracy', accuracy, on_step=True)
        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Dict[str, Any]:
        items = self._step(batch)
        batch_size = items['batch_size']
        loss = items['loss'] * batch_size
        accuracy = items['accuracy'] * batch_size
        return {
            'batch_size': batch_size,
            'loss': loss,
            'accuracy': accuracy,
        }

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Dict[str, Any]:
        items = self._step(batch)
        batch_size = items['batch_size']
        loss = items['loss'] * batch_size
        accuracy = items['accuracy'] * batch_size
        return {
            'batch_size': batch_size,
            'loss': loss,
            'accuracy': accuracy,
        }

    def _epoch_end(self, step_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        data_size = sum([i['batch_size'] for i in step_outputs])
        loss = sum([i['loss'] for i in step_outputs]) / data_size
        accuracy = sum([i['accuracy'] for i in step_outputs]) / data_size
        return {'loss': loss, 'accuracy': accuracy}

    def validation_epoch_end(self, step_outputs: List[Dict[str, Any]]):
        items = self._epoch_end(step_outputs)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('valid_loss', loss, prog_bar=True)
        self.log('valid_accuracy', accuracy, prog_bar=True)

    def test_epoch_end(self, step_outputs: List[Dict[str, Any]]):
        items = self._epoch_end(step_outputs)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)


if __name__ == '__main__':
    image_size = 3
    hidden_n = 64
    output_n = 2

    model = Model(image_size, hidden_n, output_n)
    x = torch.rand([2, *[image_size] * 2])

    y = model(x)

    print('x', *x, '', sep='\n')
    print('y', *y, '', sep='\n')
