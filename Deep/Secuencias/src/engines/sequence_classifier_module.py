
import torch
from torch import nn
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class SequenceClassifierModule(LightningModule):
    def __init__(self, backbone: nn.Module, num_classes: int,
                 lr: float = 1e-3, weight_decay: float = 1e-2, head_dropout: float = 0.2):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "num_classes"])
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(head_dropout),
                                  nn.Linear(self.backbone.out_dim, num_classes))
        self.crit = nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc  = MulticlassAccuracy(num_classes=num_classes)

        self.val_f1    = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1   = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x.float())
        loss = self.crit(logits, y)
        preds = logits.argmax(-1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x.float())
        loss = self.crit(logits, y)
        preds = logits.argmax(-1)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1",  self.val_f1,  on_epoch=True, prog_bar=True)

    def test_step(self, batch, _):
        x, y = batch
        logits = self(x.float())
        preds = logits.argmax(-1)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        self.log("test_f1",  self.test_f1,  on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
