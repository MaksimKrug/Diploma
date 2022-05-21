from collections import OrderedDict

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
    use_pretrained=False,
    model_type="ddrnet_23_slim",
):
    # callbacks
    callbacks, logger = get_callbacks(
        logger_name, logger_save_path, callback_name, callback_save_path
    )
    # init model
    if model_type == "ddrnet_23":
        model = DualResNet(
            BasicBlock,
            [2, 2, 2, 2],
            planes=64,
            spp_planes=128,
            head_planes=128,
            augment=False,
            loss_name=params["loss_name"],
            optimizer_name=params["optimizer_name"],
            weight=params["weight"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            automatic_optimization=params["automatic_optimization"],
            scheduler_type=params["scheduler_type"],
            scheduler_patience=params["scheduler_patience"],
            num_classes=params["num_classes"],
        )
    elif model_type == "ddrnet_23_slim":
        model = DualResNet(
            BasicBlock,
            [2, 2, 2, 2],
            planes=32,
            spp_planes=128,
            head_planes=64,
            augment=False,
            loss_name=params["loss_name"],
            optimizer_name=params["optimizer_name"],
            weight=params["weight"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            automatic_optimization=params["automatic_optimization"],
            scheduler_type=params["scheduler_type"],
            scheduler_patience=params["scheduler_patience"],
            num_classes=params["num_classes"],
        )

    # load ImageNet pretrained
    if use_pretrained:
        checkpoint = torch.load(
            "../data/models/DDRNet23s_imagenet.pth", map_location="cpu"
        )
        new_state_dict = OrderedDict()
        for key in checkpoint.keys():
            new_state_dict[key[7:]] = checkpoint[key]

        model.load_state_dict(new_state_dict, strict=False)

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
    model = model.load_from_checkpoint(f"{callback_save_path}/{callback_name}.ckpt")
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
            "model": model_type,
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


BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False
            ),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False
            ),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False
            ),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False
            ),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):

        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1(
                (
                    F.interpolate(self.scale1(x), size=[height, width], mode="bilinear")
                    + x_list[0]
                )
            )
        )
        x_list.append(
            (
                self.process2(
                    (
                        F.interpolate(
                            self.scale2(x), size=[height, width], mode="bilinear"
                        )
                        + x_list[1]
                    )
                )
            )
        )
        x_list.append(
            self.process3(
                (
                    F.interpolate(self.scale3(x), size=[height, width], mode="bilinear")
                    + x_list[2]
                )
            )
        )
        x_list.append(
            self.process4(
                (
                    F.interpolate(self.scale4(x), size=[height, width], mode="bilinear")
                    + x_list[3]
                )
            )
        )

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=8):
        super().__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(
            inplanes, interplanes, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            interplanes, outplanes, kernel_size=1, padding=0, bias=True
        )
        self.scale_factor = scale_factor

    def forward(self, x):

        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode="bilinear")

        return out


class DualResNet(pl.LightningModule):
    def __init__(
        self,
        block,
        layers,
        num_classes=19,
        planes=64,
        spp_planes=128,
        head_planes=128,
        augment=False,
        loss_name="BCEWeighted",
        optimizer_name="Adam",
        weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        lr=1e-4,
        weight_decay=0,
        automatic_optimization=False,
        scheduler_type="Plateau",
        scheduler_patience=2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # utils
        highres_planes = planes * 2
        self.augment = augment
        self.optimizer_name = optimizer_name
        self.weight = weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.automatic_optimization = automatic_optimization
        self.scheduler_type = scheduler_type
        self.scheduler_patience = scheduler_patience
        self.num_classes = num_classes

        # calculate loss
        self.loss_fn = get_loss(loss_name, self.weight, self.num_classes)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, planes * 2, planes * 4, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, planes * 4, planes * 8, layers[3], stride=2
        )

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
            BatchNorm2d(highres_planes, momentum=bn_mom),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
            BatchNorm2d(highres_planes, momentum=bn_mom),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(
                highres_planes,
                planes * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(
                highres_planes,
                planes * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False
            ),
            BatchNorm2d(planes * 8, momentum=bn_mom),
        )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)

        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3(self.relu(layers[2])),
            size=[height_output, width_output],
            mode="bilinear",
        )
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[3])),
            size=[height_output, width_output],
            mode="bilinear",
        )

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode="bilinear",
        )

        x_ = self.final_layer(x + x_)

        if self.augment:
            x_extra = self.seghead_extra(temp)
            return [x_, x_extra]
        else:
            return x_

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

