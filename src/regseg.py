import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn

from losses import get_loss
from shared import calculate_inference_time, get_callbacks, get_optimizer


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
):
    # callbacks
    callbacks, logger = get_callbacks(
        logger_name, logger_save_path, callback_name, callback_save_path
    )
    # init model
    model = RegSeg(
        loss_name=params["loss_name"],
        optimizer_name=params["optimizer_name"],
        weight=params["weight"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        automatic_optimization=params["automatic_optimization"],
        scheduler_type=params["scheduler_type"],
        scheduler_patience=params["scheduler_patience"],
        num_classes = params["num_classes"]
    )

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
    print(f"{callback_save_path}/{callback_name}.ckpt")
    model = model.load_from_checkpoint(f"{callback_save_path}/{callback_name}.ckpt")
    model.eval()
    trainer = pl.Trainer(gpus=1, logger=logger)
    model_metrics = trainer.test(model, test_dataloader)

    # get inference time
    inference_time_gpu, inference_time_cpu = calculate_inference_time(test_dataloader, model)

    # update test metrics
    test_metrics = test_metrics.append(
        {
            "model": "regseg",
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

activation_util = {}

def get_activation(name):
    # activation for BiasLoss
    def hook(model, input, output):
        activation_util[name] = output.detach()

    return hook

def activation():
    return nn.ReLU(inplace=True)


def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        apply_act=True,
    ):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.bn = norm2d(out_channels)
        if apply_act:
            self.act = activation()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self, w, dilations, group_width, stride, bias):
        super().__init__()
        num_splits = len(dilations)
        assert w % num_splits == 0
        temp = w // num_splits
        assert temp % group_width == 0
        groups = temp // group_width
        convs = []
        for d in dilations:
            convs.append(
                nn.Conv2d(
                    temp,
                    temp,
                    3,
                    padding=d,
                    dilation=d,
                    stride=stride,
                    bias=bias,
                    groups=groups,
                )
            )
        self.convs = nn.ModuleList(convs)
        self.num_splits = num_splits

    def forward(self, x):
        x = torch.tensor_split(x, self.num_splits, dim=1)
        res = []
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res, dim=1)


class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.act1(self.conv1(y))
        y = self.act2(self.conv2(y))
        return x * y


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg = nn.AvgPool2d(2, 2, ceil_mode=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.avg = None
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.avg is not None:
            x = self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dilations, group_width, stride, attention="se"
    ):
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm2d(out_channels)
        self.act1 = activation()
        if len(dilations) == 1:
            dilation = dilations[0]
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                groups=groups,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        else:
            self.conv2 = DilatedConv(
                out_channels,
                dilations,
                group_width=group_width,
                stride=stride,
                bias=False,
            )
        self.bn2 = norm2d(out_channels)
        self.act2 = activation()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm2d(out_channels)
        self.act3 = activation()
        if attention == "se":
            self.se = SEModule(out_channels, in_channels // 4)
        elif attention == "se2":
            self.se = SEModule(out_channels, out_channels // 4)
        else:
            self.se = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x


def generate_stage2(ds, block_fun):
    blocks = []
    for d in ds:
        blocks.append(block_fun(d))
    return blocks


class RegSegBody(nn.Module):
    def __init__(self, ds):
        super().__init__()
        gw = 16
        attention = "se"
        self.stage4 = DBlock(32, 48, [1], gw, 2, attention)
        self.stage8 = nn.Sequential(
            DBlock(48, 128, [1], gw, 2, attention),
            DBlock(128, 128, [1], gw, 1, attention),
            DBlock(128, 128, [1], gw, 1, attention),
        )

        self.stage16 = nn.Sequential(
            DBlock(128, 256, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1, attention)),
            DBlock(256, 320, ds[-1], gw, 1, attention),
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        return {"4": x4, "8": x8, "16": x16}

    def channels(self):
        return {"4": 48, "8": 128, "16": 320}


class Exp2_LRASPP(nn.Module):
    # LRASPP
    def __init__(self, num_classes, channels, inter_channels=128):
        super().__init__()
        channels8, channels16 = channels["8"], channels["16"]
        self.cbr = ConvBnAct(channels16, inter_channels, 1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels16, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(channels8, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, x):
        # intput_shape=x.shape[-2:]
        x8, x16 = x["8"], x["16"]
        x = self.cbr(x16)
        s = self.scale(x16)
        x = x * s
        x = F.interpolate(x, size=x8.shape[-2:], mode="bilinear", align_corners=False)
        x = self.low_classifier(x8) + self.high_classifier(x)
        return x


class Exp2_Decoder4(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8, channels16 = channels["8"], channels["16"]
        self.head8 = ConvBnAct(channels8, 32, 1)
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.conv = ConvBnAct(128 + 32, 128, 3, 1, 1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        # intput_shape=x.shape[-2:]
        x8, x16 = x["8"], x["16"]
        x16 = self.head16(x16)
        x16 = F.interpolate(
            x16, size=x8.shape[-2:], mode="bilinear", align_corners=False
        )
        x8 = self.head8(x8)
        x = torch.cat((x8, x16), dim=1)
        x = self.conv(x)
        x = self.classifier(x)
        return x


class Exp2_Decoder10(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8, channels16 = channels["8"], channels["16"]
        self.head8 = ConvBnAct(channels8, 32, 1)
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.conv = DBlock(128 + 32, 128, [1], 16, 1, "se")
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        # intput_shape=x.shape[-2:]
        x8, x16 = x["8"], x["16"]
        x16 = self.head16(x16)
        x16 = F.interpolate(
            x16, size=x8.shape[-2:], mode="bilinear", align_corners=False
        )
        x8 = self.head8(x8)
        x = torch.cat((x8, x16), dim=1)
        x = self.conv(x)
        x = self.classifier(x)
        return x


class Exp2_Decoder12(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8, channels16 = channels["8"], channels["16"]
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.conv = ConvBnAct(128, 128, 1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        # intput_shape=x.shape[-2:]
        x8, x16 = x["8"], x["16"]
        x16 = self.head16(x16)
        x16 = F.interpolate(
            x16, size=x8.shape[-2:], mode="bilinear", align_corners=False
        )
        x8 = self.head8(x8)
        x = x8 + x16
        x = self.conv(x)
        x = self.classifier(x)
        return x


class Exp2_Decoder14(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels8, channels16 = channels["8"], channels["16"]
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.conv = ConvBnAct(128, 128, 3, 1, 1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        # intput_shape=x.shape[-2:]
        x8, x16 = x["8"], x["16"]
        x16 = self.head16(x16)
        x16 = F.interpolate(
            x16, size=x8.shape[-2:], mode="bilinear", align_corners=False
        )
        x8 = self.head8(x8)
        x = x8 + x16
        x = self.conv(x)
        x = self.classifier(x)
        return x


class Exp2_Decoder26(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.head4 = ConvBnAct(channels4, 8, 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64 + 8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16 = x["4"], x["8"], x["16"]
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = F.interpolate(
            x16, size=x8.shape[-2:], mode="bilinear", align_corners=False
        )
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class Exp2_Decoder29(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]
        self.head16 = ConvBnAct(channels16, 256, 1)
        self.head8 = ConvBnAct(channels8, 256, 1)
        self.head4 = ConvBnAct(channels4, 16, 1)
        self.conv8 = ConvBnAct(256, 128, 3, 1, 1)
        self.conv4 = ConvBnAct(128 + 16, 128, 3, 1, 1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        x4, x8, x16 = x["4"], x["8"], x["16"]
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = F.interpolate(
            x16, size=x8.shape[-2:], mode="bilinear", align_corners=False
        )
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class RegSeg(pl.LightningModule):
    # exp48_decoder26 is what we call RegSeg in our paper
    # exp53_decoder29 is a larger version of exp48_decoder26
    # all the other models are for ablation studies
    def __init__(
        self,
        weight=[1,1,1,1,1,1,1,1,1,1,1],
        name="exp48_decoder26",
        num_classes=11,
        pretrained="",
        ablate_decoder=False,
        change_num_classes=False,
        loss_name="BCEWeighted",
        optimizer_name="Adam",
        lr=1e-4,
        weight_decay=0,
        automatic_optimization=False,
        scheduler_type="Plateau",
        scheduler_patience=2,
    ):
        super().__init__()
        self.stem = ConvBnAct(3, 32, 3, 2, 1)

        # utils
        self.optimizer_name = optimizer_name
        self.weight = weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.automatic_optimization = automatic_optimization
        self.scheduler_type = scheduler_type
        self.scheduler_patience = scheduler_patience
        self.num_classes = num_classes
        self.loss_name = loss_name

        self.save_hyperparameters()

        # calculate loss
        self.loss_fn = get_loss(loss_name, self.weight, self.num_classes)

        body_name, decoder_name = name.split("_")
        if "exp30" == body_name:
            self.body = RegSegBody(5 * [[1, 4]] + 8 * [[1, 10]])
        elif "exp43" == body_name:
            self.body = RegSegBody(
                [[1], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10]] + 7 * [[1, 12]]
            )
        elif "exp46" == body_name:
            self.body = RegSegBody(
                [[1], [1, 2], [1, 4], [1, 6], [1, 8]] + 8 * [[1, 10]]
            )
        elif "exp47" == body_name:
            self.body = RegSegBody(
                [[1], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 12]] + 6 * [[1, 14]]
            )
        elif "exp48" == body_name:
            self.body = RegSegBody([[1], [1, 2]] + 4 * [[1, 4]] + 7 * [[1, 14]])
        elif "exp49" == body_name:
            self.body = RegSegBody([[1], [1, 2]] + 6 * [[1, 4]] + 5 * [[1, 6, 12, 18]])
        elif "exp50" == body_name:
            self.body = RegSegBody(
                [[1], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10]] + 7 * [[1, 3, 6, 12]]
            )
        elif "exp51" == body_name:
            self.body = RegSegBody(
                [[1], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10]] + 7 * [[1, 4, 8, 12]]
            )
        elif "exp52" == body_name:
            self.body = RegSegBody([[1], [1, 2], [1, 4]] + 10 * [[1, 6]])
        else:
            raise NotImplementedError()

        if "decoder4" == decoder_name:
            self.decoder = Exp2_Decoder4(num_classes, self.body.channels())
        elif "decoder10" == decoder_name:
            self.decoder = Exp2_Decoder10(num_classes, self.body.channels())
        elif "decoder12" == decoder_name:
            self.decoder = Exp2_Decoder12(num_classes, self.body.channels())
        elif "decoder14" == decoder_name:
            self.decoder = Exp2_Decoder14(num_classes, self.body.channels())
        elif "decoder26" == decoder_name:
            self.decoder = Exp2_Decoder26(num_classes, self.body.channels())
        elif "decoder29" == decoder_name:
            self.decoder = Exp2_Decoder29(num_classes, self.body.channels())
        # elif "BisenetDecoder"==decoder_name:
        #     self.decoder=BiseNetDecoder(num_classes,self.body.channels())
        # elif "SFNetDecoder"==decoder_name:
        #     self.decoder=SFNetDecoder(num_classes,self.body.channels())
        # elif "FaPNDecoder"==decoder_name:
        #     self.decoder=FaPNDecoder(num_classes,self.body.channels())
        else:
            raise NotImplementedError()
        if pretrained != "" and not ablate_decoder:
            dic = torch.load(pretrained, map_location="cpu")
            if type(dic) == dict and "model" in dic:
                dic = dic["model"]
            if change_num_classes:
                current_model = self.state_dict()
                new_state_dict = {}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size() == current_model[k].size():
                        new_state_dict[k] = dic[k]
                    else:
                        print(k)
                        new_state_dict[k] = current_model[k]
                self.load_state_dict(new_state_dict, strict=True)
            else:
                self.load_state_dict(dic, strict=True)

        self.decoder.register_forward_hook(get_activation("conv4"))

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.stem(x)
        x = self.body(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x

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
            loss = self.loss_fn(activation_util["conv4"], logits_mask, mask)
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
                mode="max",
                patience=self.scheduler_patience,
                factor=0.5,
                verbose=True,
            )
        elif self.scheduler_type == "Step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.scheduler_patience, gamma=0.5, verbose=True
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }

