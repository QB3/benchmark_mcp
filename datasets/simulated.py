from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_samples, n_features": [
            (100, 200),
        ],
        "scale": [True, False],
    }

    def __init__(self, n_samples=10, n_features=50, scale=False):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.scale = scale
        self.random_state = 0

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)
        if self.scale:
            X /= np.linalg.norm(X, axis=0) / np.sqrt(len(y))
        data = dict(X=X, y=y)

        return self.n_features, data
