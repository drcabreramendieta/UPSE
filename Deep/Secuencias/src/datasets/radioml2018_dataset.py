
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class RadioML2018Dataset(Dataset):
    """Lectura on-the-fly del HDF5. Normalización por muestra (I/Q)."""
    def __init__(self, h5_path: str, indices: np.ndarray, normalize: bool = True):
        self.h5_path = h5_path
        self.indices = indices.astype(np.int64)
        self.normalize = normalize
        self._h5 = None  # se abre perezosamente por proceso/worker

    def __len__(self) -> int:
        return len(self.indices)

    def _require(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, i):
        f = self._require()
        j = int(self.indices[i])
        x = f["X"][j].astype(np.float32)          # (1024, 2) I/Q
        y = int(np.argmax(f["Y"][j]))             # one-hot -> índice [0..23]
        if self.normalize:
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True) + 1e-8
            x = (x - mu) / sd
        return torch.from_numpy(x), y
