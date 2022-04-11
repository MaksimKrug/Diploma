import time

import pytorch_lightning as pl
import segmentation_models_pytorch as smp


def get_callbacks(params):
    """
    Return callbacks: EarlyStopping, Checkpoint and Logger
    """
    logger = pl.loggers.TensorBoardLogger(
        save_dir="../data/logs/",
        name=f"{params['arch']}_{params['encoder_name']}_{params['loss_name']}",
        version="",
    )
    callbacks = [
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_f1", min_delta=0.0001, patience=3, verbose=False, mode="max"
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath="../data/models/",
            filename=f"{params['arch']}_{params['encoder_name']}_{params['loss_name']}",
            monitor="val_f1",
            save_last=True,
        ),
    ]

    return callbacks, logger

