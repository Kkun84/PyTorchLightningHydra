import hydra
import hydra.utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from logging import getLogger

from src.model import Model
from src.dataset import MNISTDataModule


logger = getLogger(__name__)


@hydra.main(config_path='../config', config_name='config.yaml')
def main(config) -> None:
    logger.info(f"\n{config.pretty()}")

    pl.seed_everything(config.hparams.seed)

    datamodule = MNISTDataModule(
        data_dir=config.dataset.path,
        batch_size=config.hparams.batch_size,
        num_workers=config.hparams.num_workers,
    )

    trainer = pl.Trainer(
        **config.trainer,
        checkpoint_callback=ModelCheckpoint(**config.model_checkpoint),
        callbacks=[EarlyStopping(**config.early_stopping)]
        + [hydra.utils.instantiate(i) for i in config.callbacks],
        logger=[hydra.utils.instantiate(i) for i in config.loggers],
        auto_lr_find=config.hparams.lr == 0,
    )

    model = Model(**config.hparams)

    trainer.tune(model, datamodule=datamodule)
    assert model.hparams.lr > 0, f'model.hparams.lr > 0={model.hparams.lr > 0}'
    config.hparams.lr = model.hparams.lr

    # 更新したhparamsをlogに出力するには↓がいる
    model = Model(**config.hparams)

    trainer.fit(model, datamodule=datamodule)
    trainer.test()


if __name__ == "__main__":
    main()
