import time

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from losses import get_loss
from shared import calculate_inference_time, get_callbacks, get_optimizer

activation = {}


def train_loop(
    params,
    test_metrics,
    logger_name,
    logger_save_path,
    callback_name,
    callback_save_path,
    epochs,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    continue_training=False,
):
    # callbacks
    callbacks, logger = get_callbacks(
        logger_name, logger_save_path, callback_name, callback_save_path
    )

    # init model
    if continue_training:
        model = PretrainedModel.load_from_checkpoint(
            f"{callback_save_path}/{callback_name}.ckpt"
        )
    else:
        model = PretrainedModel(**params)
        model_parameters = round(
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1
        )

    # train model
    trainer = pl.Trainer(gpus=1, max_epochs=epochs, logger=logger, callbacks=callbacks,)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

    # save model
    # torch.save(model.state_dict(), f"{callback_save_path}/{callback_name}.pt")

    # evaluate
    model = PretrainedModel.load_from_checkpoint(
        f"{callback_save_path}/{callback_name}.ckpt"
    )
    model.eval()
    trainer = pl.Trainer(gpus=1, logger=logger)
    model_metrics = trainer.test(model, test_dataloader)

    # get inference time
    inference_time_gpu, inference_time_cpu = calculate_inference_time(
        test_dataloader, model
    )

    # update test metrics
    test_metrics = test_metrics.append(
        {
            "model": params["arch"],
            "backbone": params["encoder_name"],
            "loss": params["loss_name"],
            "optimizer": params["optimizer_name"],
            "model_parameters": model_parameters,
            "modules": len([l for l in model.modules()]),
            "IoU": model_metrics[0]["test_iou"],
            "F1": model_metrics[0]["test_f1"],
            "Inference Time (GPU), ms": inference_time_gpu,
            "Inference Time (CPU), ms": inference_time_cpu,
        },
        ignore_index=True,
    )

    # delete model
    del model

    return test_metrics


def get_activation(name):
    # activation for BiasLoss
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


class PretrainedModel(pl.LightningModule):
    def __init__(
        self,
        weight,
        arch,
        encoder_name,
        loss_name,
        optimizer_name,
        lr,
        weight_decay,
        automatic_optimization=True,
        scheduler_type="Plateau",
        scheduler_patience=2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = automatic_optimization

        # utils
        self.weight = weight
        self.arch = arch
        self.encoder_name = encoder_name
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.scheduler_patience = scheduler_patience

        # init pretrained model from https://github.com/qubvel/segmentation_models.pytorch
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=3, classes=11,
        )
        self.model.decoder.register_forward_hook(get_activation("4"))

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # calculate loss
        self.loss_fn = get_loss(loss_name, self.weight)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std

        # get pred
        mask = self.model(image)

        return mask

    def shared_step(self, batch, stage):
        # get data
        image, mask = batch[0], batch[1]

        # get height and width
        h, w = image.shape[2:]

        # aserts
        assert h % 32 == 0 and w % 32 == 0  # dimension should be divisable by 32

        # get preds
        logits_mask = self.forward(image)

        # calculate loss
        if self.loss_name == "BiasLoss":
            loss = self.loss_fn(activation["4"], logits_mask, mask)
        else:
            loss = self.loss_fn(logits_mask, mask)

        # metrics
        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask > 0.5
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary",
        )

        # backward for custom optimizers
        if stage == "train" and self.automatic_optimization == False:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss, create_graph=True)
            opt.step()

        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs, stage):
        # utils for metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # metrics
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

        # track metrics
        metrics = {
            f"{stage}_iou": iou,
            f"{stage}_f1": f1,
        }
        if stage == "test":
            metrics["hp_metric"] = f1

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.optimizer_name, self.parameters(), self.lr, self.weight_decay
        )
        if self.scheduler_type == "Plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.scheduler_patience,
                factor=0.5,
                verbose=True,
                mode="max",
            )
        elif self.scheduler_type == "Step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.scheduler_patience, gamma=0.5, verbose=True
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }

