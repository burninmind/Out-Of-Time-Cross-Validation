"""
A python implementation of the out-of-time cross-validation method presented in:
@article{maldonado2022out,
  title={Out-of-time cross-validation strategies for classification in the presence of dataset shift},
  author={Maldonado, Sebasti{\'a}n and L{\'o}pez, Julio and Iturriaga, Andr{\'e}s},
  journal={Applied Intelligence},
  volume={52},
  number={5},
  pages={5770--5783},
  year={2022},
  publisher={Springer}
}
"""
import numpy as np

class OutOfTimeSplit:

    def __init__(self, n_splits=5, method='msa', forgetting=False):
        self.n_splits = n_splits
        if method not in ('msa', 'ssa'):
            raise ValueError("method must be 'msa' or 'ssa'")
        self.method = method
        self.forgetting = forgetting

    def split(self, X, y=None, periods=20):
        if type(periods) is int:
            periods = np.arange(len(X))//int(len(X)/periods)
        periods = np.asarray(periods)
        unique_periods = np.unique(periods)
        idxs = np.arange(len(periods))
        if self.forgetting:
            origins = np.arange(self.n_splits)
        else:
            origins = np.zeros(self.n_splits, dtype=int)
        split_points = np.arange(len(unique_periods))[-self.n_splits:]
        for origin, split_point in zip(origins, split_points):
            train_periods = unique_periods[origin:split_point]
            if self.method == 'msa':
                val_periods = unique_periods[split_point:]
            else:
                val_periods = unique_periods[split_point]

            train_idx = idxs[np.isin(periods,train_periods)]
            val_idx = idxs[np.isin(periods,val_periods)]
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, periods=None):
        return self.n_splits

