
import os
import numpy as np
import h5py
from torch.utils.data import DataLoader
from lightning import LightningDataModule
import kagglehub

from .radioml2018_dataset import RadioML2018Dataset


class RadioML2018DataModule(LightningDataModule):
    """
    Descarga con kagglehub y construye splits estratificados por clase.
    Opcionalmente filtra por SNR y submuestrea para prototipos.
    """
    def __init__(self,
                 min_snr: int | None = None,
                 max_snr: int | None = None,
                 max_samples: int | None = 120_000,
                 train_frac: float = 0.7,
                 val_frac: float = 0.15,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 seed: int = 42):
        super().__init__()
        self.save_hyperparameters()
        self.h5_path: str | None = None
        self.num_classes = 24
        self.train_idx = self.val_idx = self.test_idx = None
        self.train_ds = self.val_ds = self.test_ds = None

    def prepare_data(self):
        path = kagglehub.dataset_download("pinxau1000/radioml2018")
        candidates = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".hdf5"):
                    candidates.append(os.path.join(root, f))
        if not candidates:
            raise FileNotFoundError("No se encontró el archivo .hdf5 tras la descarga.")
        self.h5_path = sorted(candidates)[0]

    def setup(self, stage=None):
        assert self.h5_path and os.path.isfile(self.h5_path), "Falta el HDF5."
        rng = np.random.default_rng(self.hparams.seed)

        # Cargar sólo Y (one-hot) y Z (SNR) para dividir/filtrar
        with h5py.File(self.h5_path, "r") as f:
            Y = f["Y"][:]              # (N, 24)
            Z = f["Z"][:]              # (N,)
        y = np.argmax(Y, axis=1).astype(np.int64)
        snr = Z.astype(np.int64)

        mask = np.ones_like(y, dtype=bool)
        if self.hparams.min_snr is not None:
            mask &= snr >= int(self.hparams.min_snr)
        if self.hparams.max_snr is not None:
            mask &= snr <= int(self.hparams.max_snr)
        idx_all = np.nonzero(mask)[0]

        if self.hparams.max_samples is not None and len(idx_all) > self.hparams.max_samples:
            idx_all = rng.choice(idx_all, size=int(self.hparams.max_samples), replace=False)

        # Split estratificado por clase (simple)
        train_idx, val_idx, test_idx = [], [], []
        for c in range(self.num_classes):
            idx_c = idx_all[y[idx_all] == c]
            if len(idx_c) == 0:
                continue
            rng.shuffle(idx_c)
            n = len(idx_c)
            n_tr = int(self.hparams.train_frac * n)
            n_va = int(self.hparams.val_frac * n)
            train_idx.extend(idx_c[:n_tr])
            val_idx.extend(idx_c[n_tr:n_tr+n_va])
            test_idx.extend(idx_c[n_tr+n_va:])

        self.train_idx = np.array(train_idx, dtype=np.int64)
        self.val_idx   = np.array(val_idx,   dtype=np.int64)
        self.test_idx  = np.array(test_idx,  dtype=np.int64)

        self.train_ds = RadioML2018Dataset(self.h5_path, self.train_idx, normalize=True)
        self.val_ds   = RadioML2018Dataset(self.h5_path, self.val_idx,   normalize=True)
        self.test_ds  = RadioML2018Dataset(self.h5_path, self.test_idx,  normalize=True)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
