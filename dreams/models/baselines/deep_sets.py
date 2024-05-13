import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryJaccardIndex
from abc import abstractmethod
from dreams.models.layers.fourier_features import FourierFeatures
from dreams.models.layers.feed_forward import FeedForward
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryConfusionMatrix


class DeepSets(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        """
        :param phi: Element-wise function.
        :param rho: Single-element function after sum pooling of phi outputs.
        """
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        x = self.phi(x)
        x = torch.sum(x, dim=-2)  # (batch_size, num_tokens, d) -> (batch_size, d)
        out = self.rho(x)
        return out


class DeepSetsPeaks(pl.LightningModule):
    def __init__(self, phi_dim, phi_depth, rho_dim, rho_depth, out_dim, lr, dropout=0, fourier_strategy=None,
                 fourier_num_freqs=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # if fourier_strategy:
        #     self.fourier_enc = FourierFeatures(strategy=fourier_strategy, num_freqs=fourier_num_freqs,
        #                                        x_min=1e-4,
        #                                        x_max=1000,
        #                                        trainable=False)

        self.phi = FeedForward(in_dim=2, hidden_dim=phi_dim, out_dim=phi_dim, depth=phi_depth, dropout=dropout)
        self.rho = FeedForward(in_dim=phi_dim, hidden_dim=rho_dim, out_dim=out_dim, depth=rho_depth, dropout=dropout,
                               act_last=False)
        self.deep_sets = DeepSets(phi=self.phi, rho=self.rho)

    def forward(self, spec, charge):
        return self.deep_sets(spec)

    @abstractmethod
    def step(self, data, batch_idx):
        pass

    def training_step(self, data, batch_idx):
        _, loss = self.step(data, batch_idx)
        self.log('Train loss', loss, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx):
        label_pred, loss = self.step(data, batch_idx)
        self.log('Val loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class DeepSetsPeaksFingerprint(DeepSetsPeaks):

    def __init__(self, lr=3e-4, fp_str='fp_morgan_2048'):
        self.fp_size = int(fp_str.split('_')[-1])
        super().__init__(phi_dim=768, rho_dim=768, phi_depth=3, rho_depth=3, out_dim=self.fp_size, lr=lr)
        self.iou = BinaryJaccardIndex()

    def step(self, data, batch_idx):
        pred = self(data['spec'], data['charge'])
        pred = F.sigmoid(pred)
        loss = F.binary_cross_entropy(pred.flatten(), data['label'].flatten())
        return pred, loss

    def validation_step(self, data, batch_idx):
        pred, loss = self.step(data, batch_idx)
        self.log('Val loss', loss, sync_dist=True)
        self.log('IoU', self.iou(pred, data['label']), sync_dist=True)
        return loss


class DeepSetsPeakReg(DeepSetsPeaks):

    def __init__(self, lr, sigmoid=False, out_dim=1):
        super().__init__(phi_dim=768, rho_dim=768, phi_depth=3, rho_depth=3, out_dim=out_dim, lr=lr)
        self.sigmoid = nn.Sigmoid() if sigmoid else None
        self.out_dim = out_dim

    def step(self, data, batch_idx):
        label_pred = self(data['spec'], data['charge'])
        if self.sigmoid:
            label_pred = self.sigmoid(label_pred)

        if self.out_dim == 10:
            loss = F.mse_loss(label_pred.squeeze(), torch.stack(list(data['label'].values()), dim=1))
        elif self.out_dim == 1:
            loss = F.mse_loss(label_pred.squeeze(), data['label'])
        return label_pred, loss

    def validation_step(self, data, batch_idx):
        label_pred, loss = self.step(data, batch_idx)
        self.log('Val loss', loss, sync_dist=True)
        if self.out_dim == 10:
            prop = list(data['label'].keys())
            for i in range(10):
                self.log(f'MAE {prop[i]}', F.l1_loss(label_pred[:, i], data['label'][prop[i]]).item(), sync_dist=True)
        elif self.out_dim == 1:
            self.log('MAE', F.l1_loss(label_pred.squeeze(), data['label']).item(), sync_dist=True)
        return loss


class DeepSetsPeakIntReg(DeepSetsPeakReg):

    def __init__(self, lr):
        super().__init__(lr=lr, sigmoid=False)

    def validation_step(self, data, batch_idx):
        label_pred, loss = self.step(data, batch_idx)
        self.log('Val loss', loss, sync_dist=True)
        self.log('MAE', F.l1_loss(label_pred.squeeze(), data['label']).item(), sync_dist=True)
        self.log('Accuracy', (torch.round(label_pred.squeeze()) == data['label']).sum() / torch.numel(data['label']), sync_dist=True)
        return loss


class DeepSetsPeakBinCls(DeepSetsPeaks):

    def __init__(self, lr):
        super().__init__(phi_dim=768, rho_dim=768, phi_depth=3, rho_depth=3, out_dim=1, lr=lr)
        self.acc = BinaryAccuracy()
        self.prec = BinaryPrecision()
        self.recall = BinaryRecall()

    def forward(self, spec, charge):
        spec = self.deep_sets(spec)
        return F.sigmoid(spec)

    def step(self, data, batch_idx):
        label_pred = self(data['spec'], data['charge'])
        loss = F.binary_cross_entropy(label_pred.squeeze(), data['label'])
        return label_pred, loss

    def validation_step(self, data, batch_idx):
        label_pred, loss = self.step(data, batch_idx)
        self.log('Val loss', loss, sync_dist=True)
        self.log('Accuracy', self.acc(label_pred.squeeze(), data['label']).item(), sync_dist=True)
        self.log('Precision', self.prec(label_pred.squeeze(), data['label']).item(), sync_dist=True)
        self.log('Recall', self.recall(label_pred.squeeze(), data['label']).item(), sync_dist=True)
        return loss
