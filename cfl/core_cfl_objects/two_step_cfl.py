

from cfl.core_cfl_objects.cfl_core import CFL_Core

class Two_Step_CFL_Core(CFL_Core): #pylint says there's an issue here but there isn't

    def __init__(self, CDE_model, cluster_model):
        self.CDE_model = CDE_model
        self.cluster_model = cluster_model

    
    def train(self, dataset, standardize=False):

        # train CDE
        train_losses, test_losses = self.CDE_model.train(dataset, standardize=standardize)

        # predict P(Y|X)
        pyx = self.CDE_model.predict(dataset)

        # partition X and Y with P(Y|X)
        xlbls, ylbls = self.cluster_model.train(dataset)

        return xlbls, ylbls, train_losses, test_losses
        
    def tune(self, dataset):
        # TODO: do this later
        ...


    def predict(self, dataset):
        # TODO: make sure model's already trained etc
        
        # predict P(Y|X)
        pyx = self.CDE_model.predict(dataset)

        # partition X and Y with P(Y|X)
        xlbls, ylbls = self.cluster_model.predict(dataset)

        return xlbls, ylbls
