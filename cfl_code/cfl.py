from core_cfl_objects.two_step_cfl import Two_Step_CFL_Core
import cluster_methods
from density_estimation_methods.cde import CDE  
# TODO: import these with init file

class CFL(): 
    

    def __init__(self, tags, data_info, config):     
        '''constructor for CFL from string

        Inputs: 
        - tags : tuple containing strings specifying the methods to use. Can take either the format ("CRE") or ("CDE", "clusterer")
        - config : a configuration file for the model. TODO: not specified yet 
        '''    
    
        #constructor for CRE (conditional ratio estimator)-type CFL
        if len(tags)==1: 
            params = self.config_parser(config)
            CRE = self.lookup_CRE(tags[0])(params)
            #self.core = CRE_CFL_Core(CRE)

        #constructor for CDE (conditional density estimator) then clustering (two-step) CFL  
        elif len(tags)==2:
            CDE_tag, clusterer_tag = tags
            CDE_params, clusterer_params = self.config_parser(config)
            CDE = self.lookup_CDE(tags[0])(data_info, CDE_params, verbose=True)
            clusterer = self.lookup_clusterer(tags[1])(clusterer_params)
            self.core = Two_Step_CFL_Core(CDE, clusterer)
        else:
            raise ValueError("tags argument must be a tuple of either the form (CRE_tag, ) or (CDE_tag, clusterer_tag) ")

    # CFL_core method to wrap
    def train(self, X, Y):
        return self.core.train(X, Y)

    def tune(self, X, Y):
        self.core.tune(X, Y)

    def predict(self, X, Y):
        return self.core.predict(X, Y)
        

    def config_parser(self, config):
        # separates config file into the parameters needed to initialize object 
        # TODO: implementation
        return 0,0

    
    def lookup_CRE(self, CRE_tag):
        # sc_names = [sc.__name__ for sc in CRE.__subclasses__()]
        # for sc in CRE.__subclasses__():
        #     if CRE_tag==sc.__name__:
        #         return sc
        # raise LookupError("CRE_tag must exactly match one of the following names:", sc_names)
        
        # TODO: turn this on when we have a CRE implementation
        class Tmp(): pass
        return (Tmp)

    def lookup_CDE(self, CDE_tag):            
        sc_names = [sc.__name__ for sc in CDE.__subclasses__()]
        for sc in CDE.__subclasses__():
            if CDE_tag==sc.__name__:
                return sc
        raise LookupError("CDE_tag must exactly match one of the following names:", sc_names)
    
    def lookup_clusterer(self, clusterer_tag):
        sc_names = [sc.__name__ for sc in Clusterer.__subclasses__()]
        for sc in Clusterer.__subclasses__():
            if clusterer_tag==sc.__name__:
                return sc
        raise LookupError("clusterer_tag must exactly match one of the following names:", sc_names)

        