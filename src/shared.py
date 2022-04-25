import time

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch_optimizer


def get_callbacks(params):
    """
    Return callbacks: EarlyStopping, Checkpoint and Logger
    """
    logger = pl.loggers.TensorBoardLogger(
        save_dir="../data/logs/",
        name=f"{params['arch']}_{params['encoder_name']}_{params['optimizer_name']}_{params['loss_name']}",
        version="",
    )
    callbacks = [
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_f1", min_delta=0.001, patience=5, verbose=False, mode="max"
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath="../data/models/",
            filename=f"{params['arch']}_{params['encoder_name']}_{params['optimizer_name']}_{params['loss_name']}",
            monitor="val_f1",
        ),
    ]

    return callbacks, logger


def get_optimizer(optimizer_name, model_params, lr, weight_decay):
    if optimizer_name in ["SGD", "Adam"]:
        optimizer = getattr(torch.optim, optimizer_name)(
            model_params, lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name in [
        "AdaBelief",
        "AdaBound",
        "AdamP",
        "MADGRAD",
        "Apollo",
        "Adahessian",
    ]:
        optimizer = getattr(torch_optimizer, optimizer_name)(
            model_params, lr=lr, weight_decay=weight_decay
        )

    return optimizer
