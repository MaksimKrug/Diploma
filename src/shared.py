import time

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch_optimizer


def get_callbacks(logger_name, logger_save_path, callback_name, callback_save_path):
    """
    Return callbacks: Logger, EarlyStopping and Checkpoint 
    """
    logger = pl.loggers.TensorBoardLogger(
        save_dir=logger_save_path,
        name=logger_name,
        version="",
        default_hp_metric=False,
    )
    callbacks = [
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_f1", min_delta=0.001, patience=3, verbose=False, mode="max"
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=callback_save_path,
            filename=callback_name,
            monitor="val_f1",
            mode="max",
        ),
    ]

    return callbacks, logger


def get_optimizer(optimizer_name, model_params, lr, weight_decay):
    """
    Get optimizer from torch.optim or from torch_optimizer
    """
    if optimizer_name in ["SGD", "Adam"]:
        optimizer = getattr(torch.optim, optimizer_name)(
            model_params, lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name in ["AdamP", "MADGRAD", "AdaBelief", "Adahessian"]:
        optimizer = getattr(torch_optimizer, optimizer_name)(
            model_params, lr=lr, weight_decay=weight_decay
        )

    return optimizer


def calculate_inference_time(dataloader, model):
    """
    Make 1000 predictions and calculate average inference time for >10 preds
    """
    # utils
    batch_size = dataloader.batch_size
    time_val = 0
    iters = 1

    # device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Inference with: {device}")
    model = model.to(device)

    for x, _ in dataloader:
        x = x.to(device)
        # pred
        start_time = time.time()
        model(x)
        end_time = time.time() - start_time

        # skip for first 10 iters
        if iters > 10:
            time_val += end_time
        iters += 1

        # stop on 1000 image
        if iters >= 1000:
            break

    return (time_val / (iters - 10)) * 1000 / batch_size
