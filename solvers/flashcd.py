from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from flashcd import MCP
    import numpy as np


class Solver(BaseSolver):
    name = "flashcd"

    parameters = {
        'ws': [True, False],
        'ws_strategy' : ["subdiff", "fix_point"],
        'use_acc' : [True, False]}

    def set_objective(self, X, y, lmbd, gamma):
        self.X, self.y = np.asfortranarray(X), y
        self.lmbd, self.gamma = lmbd, gamma

        if self.ws:
            p0 = 10
            prune = True
        else:
            # very dirty
            try:
                p0 = self.X[0].shape[1]
            except:
                p0 = self.X.shape[1]
            prune = False
        self.clf = MCP(
            alpha=lmbd, gamma=gamma, fit_intercept=False, prune=prune, p0=p0,
            use_acc=self.use_acc)

        # Make sure we cache the numba compilation.
        self.run(1)
        self.clf.tol = 1e-12

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        try:
            X_data = self.X[0]
        except:
            X_data = self.X
        self.clf.fit(X_data, self.y)  # dirty hack

    def get_result(self):
        return self.clf.coef_
