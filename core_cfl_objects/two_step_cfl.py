from core_cfl_objects.cfl_core import CFL_Core
from sklearn.model_selection import train_test_split

class Two_Step_CFL_Core(CFL_Core): #pylint says there's an issue here but there isn't

    def __init__(self, CDE_model, cluster_model):
        self.CDE_model = CDE_model
        self.cluster_model = cluster_model

    
    def train(self, X, Y):
        
        # train-test split
        Xtr, Xts, Ytr, Yts = train_test_split(X, Y, shuffle=True, test_size=0.25)

        # train CDE
        self.CDE_model.train(Xtr, Ytr, Xts, Yts)

        # predict P(Y|X)
        pyx = self.CDE_model.predict(X, Y)

        # partition X and Y with P(Y|X)
        xlbls, ylbls = self.cluster_model.train(pyx)
        return xlbls, ylbls
        
    def tune(self, X, Y):
        # TODO: do this later
        ...


    def predict(self, X, Y):
        # TODO: make sure model's already trained etc
        
        # predict P(Y|X)
        pyx = self.CDE_model.predict(X, Y)

        # partition X and Y with P(Y|X)
        xlbls, ylbls = self.cluster_model.predict(pyx)

        return xlbls, ylbls
