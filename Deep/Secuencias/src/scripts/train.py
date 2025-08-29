
import os
import sys
import argparse
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Asegura que el proyecto raíz esté en sys.path al ejecutar desde cualquier carpeta
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from datasets.radioml2018_data_module import RadioML2018DataModule
from engines.sequence_classifier_module import SequenceClassifierModule
from models.lstm_backbone import LSTMBackbone
from models.gru_backbone import GRUBackbone
from models.transformer_backbone import TransformerBackbone


def build_backbone(name: str,
                   hidden: int, rnn_layers: int, bidir: bool,
                   d_model: int, nhead: int, tf_layers: int, ff: int, dropout: float):
    name = name.lower()
    if name == "lstm":
        return LSTMBackbone(input_dim=2, hidden=hidden, layers=rnn_layers, bidir=bidir, dropout=dropout)
    if name == "gru":
        return GRUBackbone(input_dim=2, hidden=hidden, layers=rnn_layers, bidir=bidir, dropout=dropout)
    if name == "transformer":
        return TransformerBackbone(input_dim=2, d_model=d_model, nhead=nhead,
                                   layers=tf_layers, ff=ff, dropout=dropout)
    raise ValueError(f"Modelo no soportado: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="lstm", choices=["lstm", "gru", "transformer"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=230)
    ap.add_argument("--num_workers", type=int, default=31)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--head_dropout", type=float, default=0.2)
    ap.add_argument("--min_snr", type=int, default=None)
    ap.add_argument("--max_snr", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)
    # RNN
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--rnn_layers", type=int, default=2)
    ap.add_argument("--unidirectional", action="store_true")
    # Transformer
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--tf_layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    seed_everything(args.seed)

    # Data
    dm = RadioML2018DataModule(min_snr=args.min_snr, max_snr=args.max_snr,
                               max_samples=args.max_samples, batch_size=args.batch_size,
                               num_workers=args.num_workers, seed=args.seed)
    dm.prepare_data()
    dm.setup()

    # Modelo
    backbone = build_backbone(args.model_name,
                              hidden=args.hidden, rnn_layers=args.rnn_layers, bidir=not args.unidirectional,
                              d_model=args.d_model, nhead=args.nhead, tf_layers=args.tf_layers,
                              ff=args.ff, dropout=args.dropout)
    lit = SequenceClassifierModule(backbone, num_classes=24,
                                   lr=args.lr, weight_decay=args.weight_decay, head_dropout=args.head_dropout)

    # Logging + callbacks
    logger = TensorBoardLogger(save_dir=os.path.join(PROJECT_ROOT, "tb_logs"),
                               name=f"radioml2018_{args.model_name}")
    ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1,
                           filename=f"{args.model_name}-{{epoch}}-{{val_acc:.3f}}")
    es = EarlyStopping(monitor="val_acc", mode="max", patience=5)

    trainer = Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto",
                      logger=logger, callbacks=[ckpt, es], log_every_n_steps=10)

    trainer.fit(lit, datamodule=dm)
    trainer.test(lit, datamodule=dm)
    print("Best checkpoint:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
