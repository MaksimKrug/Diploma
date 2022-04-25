import time

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch_optimizer

from losses import get_loss
from shared import get_optimizer

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


class PretrainedModel(pl.LightningModule):
    def __init__(
        self, weight, arch, encoder_name, loss_name, optimizer_name, lr, weight_decay
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # utils
        self.weight = weight
        self.arch = arch
        self.encoder_name = encoder_name
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay

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

        if stage == "train":
            start_time = time.time()
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss, create_graph=True)
            # loss.backward()
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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.1, verbose=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }


import math

import torch
from torch.optim.optimizer import Optimizer


class Adahessian(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1). You can also try 0.5. For some tasks we found this to result in better performance.
        single_gpu (Bool, optional): Do you use distributed training or not "torch.nn.parallel.DistributedDataParallel" (default: True)
    """

    def __init__(
        self,
        params,
        lr=0.15,
        betas=(0.9, 0.999),
        eps=1e-4,
        weight_decay=0,
        hessian_power=1,
        single_gpu=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
        self.single_gpu = single_gpu
        super(Adahessian, self).__init__(params, defaults)

    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError(
                    "Gradient tensor {:} does not have grad_fn. When calling\n".format(
                        i
                    )
                    + "\t\t\t  loss.backward(), make sure the option create_graph is\n"
                    + "\t\t\t  set to True."
                )

        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]

        # this is for distributed setting with single node and multi-gpus,
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for v1 in v:
                math.dist.all_reduce(v1)
        if not self.single_gpu:
            for v_i in v:
                v_i[v_i < 0.0] = -1.0
                v_i[v_i >= 0.0] = 1.0

        hvs = torch.autograd.grad(
            grads, params, grad_outputs=v, only_inputs=True, retain_graph=True
        )

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                # Hessian diagonal block size is 1 here.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = hv.abs()

            elif len(param_size) == 4:  # Conv kernel
                # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            hutchinson_trace.append(tmp_output)

        # this is for distributed setting with single node and multi-gpus,
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for output1 in hutchinson_trace:
                math.dist.all_reduce(output1 / torch.cuda.device_count())

        return hutchinson_trace

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        groups = []
        grads = []

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # get the Hessian diagonal

        hut_traces = self.get_trace(params, grads)

        for (p, group, grad, hut_trace) in zip(params, groups, grads, hut_traces):

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data)
                # Exponential moving average of Hessian diagonal square values
                state["exp_hessian_diag_sq"] = torch.zeros_like(p.data)

            exp_avg, exp_hessian_diag_sq = (
                state["exp_avg"],
                state["exp_hessian_diag_sq"],
            )

            beta1, beta2 = group["betas"]

            state["step"] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_hessian_diag_sq.mul_(beta2).addcmul_(
                hut_trace, hut_trace, value=1 - beta2
            )

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            # make the square root, and the Hessian power
            k = group["hessian_power"]
            denom = (
                (exp_hessian_diag_sq.sqrt() ** k) / math.sqrt(bias_correction2) ** k
            ).add_(group["eps"])

            # make update
            p.data = p.data - group["lr"] * (
                exp_avg / bias_correction1 / denom + group["weight_decay"] * p.data
            )

        return loss
