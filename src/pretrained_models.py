from functools import lru_cache

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from losses import get_loss


class PretrainedModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, loss_name, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        # utils
        self.arch = arch
        self.encoder_name = encoder_name
        self.loss_name = loss_name
        self.lr = lr
        self.weight_decay = weight_decay

        # init pretrained model from https://github.com/qubvel/segmentation_models.pytorch
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=3, classes=11,
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = get_loss(loss_name)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        # get data
        image, mask = batch[0], batch[1]

        # get height and width
        h, w = image.shape[2:]

        # aserts
        assert image.ndim == 4 and mask.ndim == 4  # [batch, channels, height, width]
        assert h % 32 == 0 and w % 32 == 0  # dimension should be divisable by 32
        assert mask.max() <= 1.0 and mask.min() >= 0  # mask values [0, 1]

        # get preds
        logits_mask = self.forward(image)

        # calculate loss
        loss = self.loss_fn(logits_mask, mask)

        # metrics
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs, stage):
        # utils for metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # macro
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
        # self.logger.log_metrics(metrics)

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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.1, verbose=False
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }
        return
