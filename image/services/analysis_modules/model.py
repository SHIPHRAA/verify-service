from typing import Dict, List, Tuple

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim import Adam
import numpy as np

from .config import BACKBONE


class BinaryClassificationEfficientNet(pl.LightningModule):
    """
    PyTorch Lightning module for training and evaluation using an EfficientNet model.
    Parameters
    ----------
    is_train : bool
        Whether to initialize the model for training.
    pretrained_path : str
        Path to a pre-trained model's state dictionary.
    Attributes
    ----------
    model : timm.models.Model
        EfficientNet model from the TIMM library.
    criterion : torch.nn.BCEWithLogitsLoss
        Binary cross entropy loss with logits.
    optimizer : torch.optim.Adam
        Optimizer for training.
    Methods
    -------
    forward(inputs)
        Defines the forward pass of the model.
    training_step(batch)
        Processes a training batch.
    validation_step(batch)
        Processes a validation batch.
    configure_optimizers()
        Configures the model's optimizer.
    calculate_metrics(stage, loss, preds, label)
        Calculates and logs metrics.
    predict_step(batch, batch_idx, dataloader_idx)
        Makes predictions on a given batch.
    """

    def __init__(self, is_train: bool = True, pretrained_path: str = "") -> None:
        super().__init__()
        if is_train:
            self.model = timm.create_model(BACKBONE, pretrained=is_train)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
        else:
            self.load_state_dict(torch.load(pretrained_path)["state_dict"])
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        Parameters
        ----------
        inputs : torch.Tensor
            Input data tensor.
        Returns
        -------
        torch.Tensor
            Output tensor from the model.
        """
        outputs = self.model(inputs)
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Processes a training batch.
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Training batch.
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing loss, predictions, and labels.
        """
        inputs, label = batch["image"], batch["label"]
        preds = self(inputs)
        loss = self.criterion(preds, label.float().view(-1, 1))
        data = {"loss": loss, "preds": preds, "label": label}
        self.calculate_metrics(
            "train",
            loss.cpu().detach().numpy(),
            preds.cpu().detach().numpy(),
            label.cpu().detach().numpy(),
        )
        return data

    def validation_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a validation batch.
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Validation batch.
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing loss, predictions, and labels.
        """
        inputs, label = batch["image"], batch["label"]
        preds = self(inputs)
        loss = self.criterion(preds, label.float().view(-1, 1))
        data = {"loss": loss, "preds": preds, "label": label}
        self.log("val_loss", loss)
        self.calculate_metrics(
            "val",
            loss.cpu().detach().numpy(),
            preds.cpu().detach().numpy(),
            label.cpu().detach().numpy(),
        )
        return data

    def configure_optimizers(self) -> List[Adam]:
        """
        Configures the model's optimizer.
        Returns
        -------
        List[Adam]
            List containing the optimizer.
        """
        self.optimizer = Adam(self.parameters(), lr=5e-4)
        return [self.optimizer]

    def calculate_metrics(
        self, stage: str, loss: float, preds: torch.Tensor, label: torch.Tensor
    ) -> None:
        """
        Calculates and logs metrics.
        Parameters
        ----------
        stage : str
            Stage of training ('train' or 'val').
        loss : float
            Calculated loss value.
        preds : torch.Tensor
            Predictions created by the model.
        label : torch.Tensor
            Actual labels.
        """
        try:
            roc_auc = metrics.roc_auc_score(label, preds)
        except:
            roc_auc = 0
        self.log_dict({f"{stage}_metric": roc_auc}, prog_bar=True, logger=True)

    def predict_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Defines the prediction step of the model.
        This method executes the model's predictions on a given batch.
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch containing input data for prediction. Typically a dictionary with 'image' and 'label' keys.
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing prediction results. Typically has a 'tensor' key with the prediction values.
        """
        inputs, label, path = batch["image"], batch["label"], batch["path"]
        preds = self(inputs)
        return (
            preds.sigmoid().cpu().detach().numpy().flatten(),
            label.cpu().detach().numpy().flatten(),
            np.array(path),
        )
