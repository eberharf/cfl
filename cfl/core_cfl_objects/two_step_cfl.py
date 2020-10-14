from sklearn.model_selection import train_test_split
from cfl.util.data_processing import standardize_train_test

from cfl.core_cfl_objects.cfl_core import CFL_Core

class Two_Step_CFL_Core(CFL_Core): #pylint says there's an issue here but there isn't

    def __init__(self, CDE_model, cluster_model):
        self.CDE_model = CDE_model
        self.cluster_model = cluster_model

    
    def train(self, X, Y, standardize=False, save_path=None):
        
        # train-test split
        # standardize if specified
        self.Xtr, self.Xts, self.Ytr, self.Yts, self.Itr, self.Its= train_test_split(X, Y, range(X.shape[0]), shuffle=True, train_size=0.85)
        if standardize:
            self.Xtr, self.Xts, self.Ytr, self.Yts = standardize_train_test([self.Xtr, self.Xts, self.Ytr, self.Yts])

        # train CDE
        train_losses, test_losses = self.CDE_model.train(self.Xtr, self.Ytr, self.Xts, self.Yts, save_path)

        # predict P(Y|X)
        pyx_tr = self.CDE_model.predict(self.Xtr)

        # partition X and Y with P(Y|X)
        xlbls, ylbls = self.cluster_model.train(pyx_tr, self.Ytr)
        return xlbls, ylbls, train_losses, test_losses
        
    def tune(self, X, Y):
        # TODO: do this later
        ...


    def predict(self, X, Y):
        # TODO: make sure model's already trained etc
        
        # predict P(Y|X)
        pyx = self.CDE_model.predict(X, Y)

        # partition X and Y with P(Y|X)
        xlbls, ylbls = self.cluster_model.predict(pyx, Y)

        return xlbls, ylbls
