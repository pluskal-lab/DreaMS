import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinarySpecificity, BinaryAUROC,
    BinaryROC, BinaryPrecisionRecallCurve, BinaryAveragePrecision)
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.aggregation import SumMetric
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Any, Union
import itertools
import pandas as pd
import plotly.graph_objects as go
import dreams.utils.mols as mu
import dreams.utils.io as io
from dreams.models.dreams.dreams import DreaMS
from dreams.models.baselines.deep_sets import DeepSets
from dreams.models.layers.feed_forward import FeedForward
from dreams.models.optimization.losses_metrics import SmoothIoULoss, CosSimLoss, FingerprintMetrics, FocalLoss
from dreams.utils.annotation import FingerprintInChIRetrieval
from dreams.definitions import *


class FineTuningHead(pl.LightningModule):
    """
    Base class for fine-tuning heads in the DreaMS model.

    This class provides a foundation for creating various fine-tuning heads
    that can be attached to a pre-trained DreaMS backbone for specific tasks.

    Attributes:
        backbone (Union[DreaMS, pl.LightningModule]): The pre-trained backbone model.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        precursor_emb (bool): Whether to use precursor embeddings.
        unfreeze_backbone_at_epoch (int): Epoch at which to unfreeze the backbone.
        head (nn.Module): The fine-tuning head (to be implemented in subclasses).
    """

    def __init__(self, backbone: Union[Path, DreaMS], lr, weight_decay, backbone_cls=DreaMS, unfreeze_backbone_at_epoch=0,
                 precursor_emb=True):
        """
        Initialize the FineTuningHead.

        Args:
            backbone (Union[Path, DreaMS]): Path to the pre-trained backbone or the backbone itself.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            backbone_cls (type): Class of the backbone model (default: DreaMS).
            unfreeze_backbone_at_epoch (int): Epoch at which to unfreeze the backbone (default: 0).
            precursor_emb (bool): Whether to use only precursor embeddings (default: True).
        """
        super(FineTuningHead, self).__init__()
        self.save_hyperparameters()

        if isinstance(backbone, Path):
            self.backbone = backbone_cls.load_from_checkpoint(
                backbone,
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
        else:
            self.backbone = backbone

        self.lr = lr
        self.weight_decay = weight_decay
        self.precursor_emb = precursor_emb
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.head = NotImplementedError('Fine tuning head must be implemented in the child subclass.')

    def forward(self, spec, charge=None, no_head=False):
        """
        Forward pass through the model.

        Args:
            spec (torch.Tensor): Input spectrum.
            charge (torch.Tensor, optional): Charge information.
            no_head (bool): If True, return embeddings without passing through the head.

        Returns:
            torch.Tensor: Output of the model.
        """
        # Get backbone embeddings
        embs = self.backbone(spec, charge)

        if self.precursor_emb:
            # Output projection from precursor peak
            embs = embs[:, 0, ...]

        if no_head:
            return embs

        return self.head(embs)

    @abstractmethod
    def step(self, data, batch_idx):
        """
        Perform a single step (to be implemented in subclasses).

        Args:
            data (dict): Input data.
            batch_idx (int): Index of the current batch.

        Returns:
            tuple: Predicted labels and loss.
        """
        pass

    def training_step(self, data, batch_idx):
        """
        Perform a single training step.

        Args:
            data (dict): Input data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        _, loss = self.step(data, batch_idx)
        self.log('Train loss', loss, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx, dataloader_idx=0):
        """
        Perform a single validation step.

        Args:
            data (dict): Input data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            torch.Tensor: The computed loss.
        """
        label_pred, loss = self.step(data, batch_idx)
        self.log('Val loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_train_epoch_start(self):
        """
        Callback method called at the start of each training epoch.
        Handles freezing/unfreezing of the backbone.
        """
        if self.trainer.current_epoch == self.unfreeze_backbone_at_epoch:
            self.backbone.unfreeze()
        elif self.trainer.current_epoch == 0:
            self.backbone.freeze()

    def _update_metric(
        self,
        name: str,
        metric_class: type[Metric],
        update_args: Any,
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        metric_kwargs: Optional[dict] = None,
        log: bool = True,
        log_n_samples: bool = False
    ) -> None:
        """
        Update a metric and optionally log it.

        Args:
            name (str): Name of the metric.
            metric_class (type[Metric]): Class of the metric.
            update_args (Any): Arguments to update the metric.
            batch_size (int, optional): Batch size for logging.
            prog_bar (bool): Whether to show the metric in the progress bar.
            metric_kwargs (dict, optional): Additional keyword arguments for the metric.
            log (bool): Whether to log the metric.
            log_n_samples (bool): Whether to log the number of samples.
        """
        # Log total number of samples for debugging
        if log_n_samples:
            self._update_metric(
                name=name + '_n_samples',
                metric_class=SumMetric,
                update_args=(len(update_args[0]),),
                batch_size=1
            )

        # Init metric if does not exits yet
        if hasattr(self, name):
            metric = getattr(self, name)
        else:
            if metric_kwargs is None:
                metric_kwargs = dict()
            metric = metric_class(**metric_kwargs).to(self.device)
            setattr(self, name, metric)

        # Update
        metric(*update_args)

        # Log
        if log:
            self.log(
                name,
                metric,
                prog_bar=prog_bar,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                metric_attribute=name  # Suggested by a torchmetrics error
            )

    def _plot_curve(self, x, y, threshold=None, name='', xaxis_range=(0, 1), yaxis_range=(0, 1)):
        """
        Plot a curve using plotly.

        Args:
            x (array-like): X-axis values.
            y (array-like): Y-axis values.
            threshold (float, optional): Threshold value to plot.
            name (str): Name of the curve.
            xaxis_range (tuple): Range for x-axis.
            yaxis_range (tuple): Range for y-axis.

        Returns:
            go.Figure: The plotly figure object.
        """
        fig = go.Figure()

        # Curve
        fig.add_trace(go.Scatter(x=x, y=y,
                                 mode='lines',
                                 name=name,
                                 line=dict(color='blue', width=2)))
        fig.update_layout(xaxis_range=xaxis_range)
        fig.update_layout(yaxis_range=yaxis_range)

        return fig


class RegressionHead(FineTuningHead):
    """
    Regression head for fine-tuning tasks.

    This class implements a regression head that can be used for various
    regression tasks on top of the DreaMS backbone.

    Attributes:
        head (nn.Module): The regression head module.
        out_dim (int): Output dimension of the regression.
        sigmoid (nn.Module): Sigmoid activation (if used).
        mol_props_calc (mu.MolPropertyCalculator): Calculator for molecular properties.
    """

    def __init__(self, backbone: Union[Path, DreaMS], lr, weight_decay, sigmoid=True, out_dim=1,
                 mol_props_calc: mu.MolPropertyCalculator = None, head_depth=1, dropout=0):
        """
        Initialize the RegressionHead.

        Args:
            backbone (Union[Path, DreaMS]): Path to the pre-trained backbone or the backbone itself.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            sigmoid (bool): Whether to use sigmoid activation (default: True).
            out_dim (int): Output dimension of the regression (default: 1).
            mol_props_calc (mu.MolPropertyCalculator, optional): Calculator for molecular properties.
            head_depth (int): Depth of the regression head (default: 1).
            dropout (float): Dropout rate (default: 0).
        """
        super().__init__(backbone=backbone, lr=lr, weight_decay=weight_decay)
        self.head = FeedForward(in_dim=self.backbone.d_model, out_dim=out_dim, depth=head_depth, act_last=False,
                                hidden_dim=self.backbone.d_model, dropout=dropout)
        self.out_dim = out_dim
        self.sigmoid = nn.Sigmoid() if sigmoid else None
        self.mol_props_calc = mol_props_calc

    def step(self, data, batch_idx):
        """
        Perform a single step of regression.

        Args:
            data (dict): Input data.
            batch_idx (int): Index of the current batch.

        Returns:
            tuple: Predicted labels and loss.
        """
        label_pred = self(data['spec'], data['charge'])
        if self.sigmoid:
            label_pred = self.sigmoid(label_pred)

        if self.mol_props_calc is not None:
            loss = F.mse_loss(label_pred.squeeze(), torch.stack(list(data['label'].values()), dim=1))
        else:
            loss = F.mse_loss(label_pred.squeeze(), data['label'])
        return label_pred, loss

    def validation_step(self, data, batch_idx, dataloader_idx=0):
        """
        Perform a single validation step.

        Args:
            data (dict): Input data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            torch.Tensor: The computed loss.
        """
        label_pred, loss = self.step(data, batch_idx)
        self.log('Val loss', loss, sync_dist=True)

        if self.mol_props_calc is not None:
            prop_names = self.mol_props_calc.prop_names
            for i, name in enumerate(prop_names):
                mae = F.l1_loss(label_pred[:, i], data['label'][name], reduction='none').detach()
                self.log(f'MAE {name}', self.mol_props_calc.denormalize_prop(mae, name, do_not_add_min=True).mean(), sync_dist=True)
        else:
            self.log('MAE', F.l1_loss(label_pred.squeeze(), data['label']).item(), sync_dist=True)
        return loss


class IntRegressionHead(RegressionHead):
    """
    Integer Regression head for fine-tuning tasks.

    This class implements a regression head specifically for integer-valued outputs.

    Attributes:
        Inherits all attributes from RegressionHead.
    """

    def __init__(self, backbone_pth: Path, lr, weight_decay, out_dim=1):
        """
        Initialize the IntRegressionHead.

        Args:
            backbone_pth (Path): Path to the pre-trained backbone.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            out_dim (int): Output dimension of the regression (default: 1).
        """
        super().__init__(backbone_pth=backbone_pth, lr=lr, weight_decay=weight_decay, sigmoid=False, out_dim=out_dim)

    def validation_step(self, data, batch_idx):
        """
        Perform a single validation step.

        Args:
            data (dict): Input data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        label_pred, loss = self.step(data, batch_idx)
        self.log('val_loss', loss, sync_dist=True)
        self.log('MAE', F.l1_loss(label_pred.squeeze(), data['label']).item(), sync_dist=True)
        self.log('Accuracy', (torch.round(label_pred.squeeze()) == data['label']).sum() / torch.numel(data['label']), sync_dist=True)
        return loss


class BinClassificationHead(FineTuningHead):
    """
    Binary Classification head for fine-tuning tasks.

    This class implements a binary classification head that can be used for
    various classification tasks on top of the DreaMS backbone.

    Attributes:
        head (nn.Module): The classification head module.
        metrics (dict): Dictionary to store various classification metrics.
        loss (nn.Module): The loss function (Focal Loss).
    """

    def __init__(self, backbone_pth: Path, lr, weight_decay, head_depth=1, head_phi_depth=0, dropout=0,
                 focal_loss_alpha=None, focal_loss_gamma=0):
        """
        Initialize the BinClassificationHead.

        Args:
            backbone_pth (Path): Path to the pre-trained backbone.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            head_depth (int): Depth of the classification head (default: 1).
            head_phi_depth (int): Depth of the phi network in DeepSets (default: 0).
            dropout (float): Dropout rate (default: 0).
            focal_loss_alpha (float, optional): Alpha parameter for Focal Loss.
            focal_loss_gamma (float): Gamma parameter for Focal Loss (default: 0).
        """
        super().__init__(backbone=backbone_pth, lr=lr, weight_decay=weight_decay, precursor_emb=head_phi_depth == 0)
        self.head = nn.Sequential(nn.Linear(self.backbone.d_model, 1), nn.Sigmoid())
        self.metrics = {}

        # TODO: refactor
        self.train_acc, self.val_acc = BinaryAccuracy(), BinaryAccuracy()
        self.train_prec, self.val_prec = BinaryPrecision(), BinaryPrecision()
        self.train_rec, self.val_rec = BinaryRecall(), BinaryRecall()
        self.train_f1, self.val_f1 = BinaryF1Score(), BinaryF1Score()
        self.train_auroc, self.val_auroc = BinaryAUROC(), BinaryAUROC()
        self.train_auprc, self.val_auprc = BinaryAveragePrecision(), BinaryAveragePrecision()
        self.head_depth = head_depth
        self.head_phi_depth = head_phi_depth
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.loss = FocalLoss(alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, binary=True)

        # TODO: move head definition to the parent class

        # Define head for the backbone
        if self.head_phi_depth > 1:
            raise NotImplementedError
        if self.head_phi_depth == 1:
            self.head = DeepSets(
                phi=nn.Sequential(
                    nn.Linear(self.backbone.d_model, self.backbone.d_model, bias=False),
                    nn.Dropout(dropout)
                ),
                rho=nn.Sequential(
                    nn.Linear(self.backbone.d_model, 1, bias=False),
                    nn.Sigmoid()
                )
            )
        else:
            self.head = FeedForward(
                in_dim=self.backbone.d_model, out_dim=1, hidden_dim='interpolated', depth=self.head_depth,
                act_last=True, act=nn.Sigmoid, dropout=dropout, bias=False
            )

    def step(self, data, batch_idx):
        """
        Perform a single step of computation.

        Args:
            data (dict): A dictionary containing the input data.
            batch_idx (int): The index of the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the predicted labels and the loss.
        """
        label_pred = self(data['spec'], data['charge'])
        loss = self.loss(label_pred.squeeze(), data['label'])
        return label_pred, loss

    def training_step(self, data, batch_idx):
        """
        Perform a single training step.

        Args:
            data (dict): A dictionary containing the input data.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for this step.
        """
        label_pred, loss = self.step(data, batch_idx)
        self.log('Train loss', loss, sync_dist=True)
        self.train_acc(label_pred.squeeze(), data['label'])
        self.train_prec(label_pred.squeeze(), data['label'])
        self.train_rec(label_pred.squeeze(), data['label'])
        self.train_f1(label_pred.squeeze(), data['label'])
        self.train_auroc(label_pred.squeeze(), data['label'])
        self.train_auprc(label_pred.squeeze(), data['label'].long())
        self.log('Train acc', self.train_acc, sync_dist=True)
        self.log('Train prec', self.train_prec, sync_dist=True)
        self.log('Train rec', self.train_rec, sync_dist=True)
        self.log('Train f1', self.train_f1, sync_dist=True)
        self.log('Train AUROC', self.train_auroc, sync_dist=True)
        self.log('Train AUPRC', self.train_auprc, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx, dataloader_idx=0):
        """
        Perform a single validation step.

        Args:
            data (dict): A dictionary containing the input data.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader. Defaults to 0.

        Returns:
            torch.Tensor: The computed loss for this step.
        """
        label_pred, loss = self.step(data, batch_idx)

        self.log('Val loss', loss, sync_dist=True, add_dataloader_idx=False)
        self.val_acc(label_pred.squeeze(), data['label'])
        self.val_prec(label_pred.squeeze(), data['label'])
        self.val_rec(label_pred.squeeze(), data['label'])
        self.val_f1(label_pred.squeeze(), data['label'])
        self.val_auroc(label_pred.squeeze(), data['label'])
        self.val_auprc(label_pred.squeeze(), data['label'].long())
        self.log('Val acc', self.val_acc, sync_dist=True, add_dataloader_idx=False)
        self.log('Val prec', self.val_prec, sync_dist=True, add_dataloader_idx=False)
        self.log('Val rec', self.val_rec, sync_dist=True, add_dataloader_idx=False)
        self.log('Val f1', self.val_f1, sync_dist=True, add_dataloader_idx=False)
        self.log('Val AUROC', self.val_auroc, sync_dist=True, add_dataloader_idx=False)
        self.log('Val AUPRC', self.val_auprc, sync_dist=True, add_dataloader_idx=False)

        # ROC
        self._update_metric(
            f'val_roc',
            BinaryROC,
            (label_pred.squeeze(), data['label'].long()),
            label_pred.size(0),
            log=False
        )
        # PR curve
        self._update_metric(
            f'val_pr',
            BinaryPrecisionRecallCurve,
            (label_pred.squeeze(), data['label'].long()),
            label_pred.size(0),
            log=False
        )

        return loss

    def on_validation_epoch_end(self):
        """
        Perform operations at the end of each validation epoch.
        This method computes and logs ROC and PR curves.
        """

        # ROC curve
        name = f'val_roc'
        if not hasattr(self, name):
            setattr(self, name, BinaryROC())
        roc = getattr(self, name)
        fpr, tpr, threshold = roc.compute()
        fpr = fpr.cpu().detach().numpy()
        tpr = tpr.cpu().detach().numpy()
        threshold = threshold.cpu().detach().numpy()
        fig = self._plot_curve(fpr, tpr, threshold, name=name)
        if self.global_rank == 0:
            self.trainer.logger.experiment.log({name: fig})
        roc.reset()

        # PR curve
        name = f'val_pr'
        if not hasattr(self, name):
            setattr(self, name, BinaryPrecisionRecallCurve())
        pr_curve = getattr(self, name)
        p, r, threshold = pr_curve.compute()
        p = p.cpu().detach().numpy()
        r = r.cpu().detach().numpy()
        threshold = threshold.cpu().detach().numpy()
        fig = self._plot_curve(p, r, threshold, name=name)
        if self.global_rank == 0:
            self.trainer.logger.experiment.log({name: fig})
        pr_curve.reset()


class FingerprintHead(FineTuningHead):

    def __init__(self, backbone: Path, fp_str: str, lr, batch_size, weight_decay, dropout=0, loss='cos',
                 retrieval_val_pth=None, retrieval_epoch_freq=10, unfreeze_backbone_at_epoch=0,
                 head_depth=1, store_val_out_dir: Path = None, head_phi_depth: int = 0):
        """
        Initialize the FingerprintHead.

        Args:
            backbone (Path): Path to the pre-trained backbone.
            fp_str (str): String representation of the fingerprint.
            lr (float): Learning rate.
            batch_size (int): Batch size.
            weight_decay (float): Weight decay for optimization.
            dropout (float, optional): Dropout rate. Defaults to 0.
            loss (str, optional): Loss function to use. Defaults to 'cos'.
            retrieval_val_pth (Path, optional): Path for validation retrieval. Defaults to None.
            retrieval_epoch_freq (int, optional): Frequency of retrieval validation. Defaults to 10.
            unfreeze_backbone_at_epoch (int, optional): Epoch to unfreeze the backbone. Defaults to 0.
            head_depth (int, optional): Depth of the head. Defaults to 1.
            store_val_out_dir (Path, optional): Directory to store validation outputs. Defaults to None.
            head_phi_depth (int, optional): Depth of the phi network in DeepSets. Defaults to 0.
        """
        super().__init__(backbone=backbone, lr=lr, weight_decay=weight_decay, precursor_emb=not head_phi_depth,
                         unfreeze_backbone_at_epoch=unfreeze_backbone_at_epoch)

        self.fp_str = fp_str
        self.fp_size = int(self.fp_str.split('_')[-1])
        self.retrieval_epoch_freq = retrieval_epoch_freq
        self.batch_size = batch_size
        self.head_depth = head_depth
        self.head_phi_depth = head_phi_depth
        self.store_val_out_dir = store_val_out_dir
        if self.store_val_out_dir:
            self.store_val_out_dir.mkdir(parents=True, exist_ok=True)

        # Define loss function
        if loss == 'cross_entropy':
            self.loss = nn.BCELoss()
        elif loss == 'cos':
            self.loss = CosSimLoss()
        elif loss == 'smooth_iou':
            self.loss = SmoothIoULoss()
        else:
            raise ValueError(f'Invalid loss function name: {self.loss_f}.')

        # Define head for the backbone
        if self.head_phi_depth > 1:
            raise NotImplementedError
        if self.head_phi_depth == 1:
            self.head = DeepSets(
                phi=nn.Sequential(
                    nn.Linear(self.backbone.d_model, self.backbone.d_model, bias=False),
                    nn.Dropout(dropout)
                ),
                rho=nn.Linear(self.backbone.d_model, self.fp_size, bias=False)
            )
        else:
            self.head = FeedForward(
                in_dim=self.backbone.d_model, out_dim=self.fp_size, hidden_dim='interpolated',
                depth=self.head_depth, act_last=False, dropout=dropout, bias=False
            )

        # Initialize fingerprint retrieval index
        self.retrieval_val_pth = retrieval_val_pth
        if self.retrieval_val_pth:
            self.val_retrieval = FingerprintInChIRetrieval(
                df_pkl_pth=self.retrieval_val_pth,
                candidate_smiles_col='isomers_smiles',
                candidate_inchi14_col='isomers_inchi14',
                top_k=[1, 5, 10, 20, 50, 100, 200],
                fp_name=self.fp_str
            )

        # Define metrics
        self.val_metrics = FingerprintMetrics(prefix='Val')

    def step(self, data, batch_idx):
        """
        Perform a single step of computation.

        Args:
            data (dict): A dictionary containing the input data.
            batch_idx (int): The index of the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the predictions and the loss.
        """
        pred = self(data['spec'], data['charge'])
        loss = self.loss(pred, data['label'])
        return pred, loss

    def validate(self, data, batch_idx, dataloader_idx):
        """
        Perform validation computations.

        Args:
            data (dict): A dictionary containing the input data.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the dataloader.

        Returns:
            torch.Tensor: The computed loss for this validation step.
        """
        pred, loss = self.step(data, batch_idx)
        real = data['label']

        metrics = self.val_metrics(pred, real)
        metrics[f'Val loss'] = loss

        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=False, batch_size=self.batch_size,
                      add_dataloader_idx=False)

        # Validate retrieval at k
        if self.__retrieval_epoch():

            if self.store_val_out_dir:
                torch.save(
                    dict(data, **{'pred': pred}),
                    self.store_val_out_dir / f'epoch{self.trainer.current_epoch}_rank{self.trainer.global_rank}_batch{batch_idx}.pt'
                )

            for i in range(len(pred)):
                self.val_retrieval.retrieve_inchi14s(query_fp=pred[i].cpu().numpy(), label_smiles=data['smiles'][i])

        return loss

    def __retrieval_epoch(self) -> bool:
        """
        Check if the current epoch is a retrieval epoch.

        Returns:
            bool: True if it's a retrieval epoch, False otherwise.
        """
        return self.retrieval_val_pth and self.current_epoch % self.retrieval_epoch_freq == 0 and self.current_epoch != 0

    def validation_step(self, data, batch_idx, dataloader_idx=0):
        """
        Perform a single validation step.

        Args:
            data (dict): A dictionary containing the input data.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader. Defaults to 0.

        Returns:
            torch.Tensor: The computed loss for this validation step.
        """
        return self.validate(data, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self):
        """
        Perform operations at the end of each validation epoch.
        This method computes and logs retrieval metrics if it's a retrieval epoch.
        """
        if self.__retrieval_epoch():
            if self.val_retrieval.n_retrievals > 0:
                metrics_avg, metrics = self.val_retrieval.compute_reset_metrics('Val', return_unaveraged=True)

            self.log_dict(metrics_avg, sync_dist=True, batch_size=self.batch_size)

            if self.store_val_out_dir:
                io.write_pickle(
                    metrics,
                    self.store_val_out_dir / f'acc_epoch{self.trainer.current_epoch}_rank{self.trainer.global_rank}.pkl'
                )


class ContrastiveHead(FineTuningHead):
    def __init__(self, backbone_pth: Path, lr, weight_decay, triplet_loss_margin: float):
        """
        Initialize the ContrastiveHead.

        Args:
            backbone_pth (Path): Path to the pre-trained backbone.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for optimization.
            triplet_loss_margin (float): Margin for the triplet loss.
        """
        super().__init__(backbone_pth, lr, weight_decay, precursor_emb=True)
        self.head = nn.Linear(self.backbone.d_model, self.backbone.d_model, bias=True)

        # Metrics for similarity correlation
        self.triplet_loss_margin = triplet_loss_margin

    def step(self, data, batch_idx):
        """
        Perform a single step of computation.

        Args:
            data (dict): A dictionary containing the input data.
            batch_idx (int): The index of the current batch.

        Returns:
            Tuple[None, torch.Tensor]: A tuple containing None and the computed loss.
        """
        # TODO: extend for more than 1 positive example? (current -cos_sim_pos + ... will not work)
        # TODO: t
        # TODO: DeepSets head (change shape comments)

        # Parse input
        spec = data['spec']  # (bs, n, d)
        pos_specs = data['pos_specs']  # (bs, n_pos, n, d)
        neg_specs = data['neg_specs']  # (bs, n_neg, n, d)
        bs, n, d = data['spec'].size()
        n_pos = data['pos_specs'].size(1)
        n_neg = data['neg_specs'].size(1)

        # Forward
        emb = self(spec, charge=None).unsqueeze(1)  # -> (bs, 1, h)

        pos_specs = pos_specs.view(-1, n, d)  # (bs, n_pos, n, d) -> (bs * n_pos, n, d)
        pos_embs = self(pos_specs, charge=None)
        pos_embs = pos_embs.view(bs, n_pos, -1)  # back to (bs, n_pos, h)

        neg_specs = neg_specs.view(-1, n, d)  # (bs, n_neg, n, d) -> (bs * n_neg, n, d)
        neg_embs = self(neg_specs, charge=None)
        neg_embs = neg_embs.view(bs, n_neg, -1)  # back to (bs, n_neg, h)

        # Contastive similarities
        cos_sim_pos = F.cosine_similarity(emb, pos_embs, dim=-1)  # (bs, n_pos)
        cos_sim_neg = F.cosine_similarity(emb, neg_embs, dim=-1)  # (bs, n_neg)

        # # Loss
        # Likelihood Contrastive
        # loss = -cos_sim_pos + torch.logsumexp(cos_sim_neg, dim=-1)
        # loss = loss.mean()
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.triplet_margin_with_distance_loss.html#torch.nn.functional.triplet_margin_with_distance_loss
        # Triplet With Margin
        loss = torch.clamp_min(self.triplet_loss_margin + (-cos_sim_pos) - (-cos_sim_neg), 0)
        loss = loss.mean()

        return None, loss

    def training_step(self, data, batch_idx):
        _, loss = self.step(data, batch_idx)
        self.log('Train loss', loss, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx, dataloader_idx=0):
        _, loss = self.step(data, batch_idx)
        self.log('Val loss', loss, sync_dist=True)
        return loss
