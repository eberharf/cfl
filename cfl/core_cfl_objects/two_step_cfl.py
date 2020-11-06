

from cfl.core_cfl_objects.cfl_core_interface import CFL_Core

class Two_Step_CFL_Core(CFL_Core): #pylint says there's an issue here but there isn't
    ''' This class encapsulates an entire CFL pipeline by acting as a wrapper
        around a CDE object and a Cluster object. It handles training and prediction
        of the entire CFL pipeline in one step. 

        Attributes:
            CDE_model : an instance of a class that follows the interface 
                        specified in cfl/density_estimation_methods/cde.py
            cluster_model : an instance of a class that follows the interface 
                            specified in cfl/cluster_methods/clusterer.py

        Methods:
            train : trains CDE and passes CDE predictions to a cluster object to
                    create an observational partition on a given Dataset.
            tune : tune the CDE and cluster models over ranges of inputs. Not yet implemented. 
            predict : given trained CDE and cluster objects, predicts the 
                      observational partition on the given Dataset.
    '''

    def __init__(self, CDE_model, cluster_model):
        ''' Assign class attributes. 
        
            Arguments: 
                CDE_model : an instance of a class that follows the interface 
                        specified in cfl/density_estimation_methods/cde.py
                cluster_model : an instance of a class that follows the interface 
                            specified in cfl/cluster_methods/clusterer.py
            Returns: None
        '''

        self.CDE_model = CDE_model
        self.cluster_model = cluster_model
    
    def train(self, dataset, standardize=False):
        ''' Train a CDE, predict conditional probabilities, and train a cluster
            model to form observational partition on a given Dataset. 

            Arguments: 
                dataset : a Dataset object containing X and Y to train on (Dataset)
                standardize : whether or not to z-score X and Y (bool)

            Returns:
                xlbls : X macrovariable class assignments for this Dataset (np.array)
                ylbls : Y macrovariable class assignments for this Dataset (np.array)
                train_losses : loss on training set from CDE training (np.array)
                test_losses : loss on test set from CDE training (np.array)
        '''

        # train CDE
        train_losses, test_losses = self.CDE_model.train(dataset, standardize=standardize, best=True)

        # predict P(Y|X)
        pyx = self.CDE_model.predict(dataset)

        # partition X and Y with P(Y|X)
        xlbls, ylbls = self.cluster_model.train(dataset)

        return xlbls, ylbls, train_losses, test_losses
        
    def tune(self, dataset):
        # TODO: do this later
        ...


    def predict(self, dataset):
        ''' Given a trained CFL pipeline, predict X and Y macrovariable
        classes for a given Dataset. 

        Arguments:
            dataset : a Dataset object containing X and Y to predict on (Dataset)
        Returns:                
            xlbls : X macrovariable class assignments for this Dataset (np.array)
            ylbls : Y macrovariable class assignments for this Dataset (np.array)
        '''
        
        # TODO: make sure model's already trained etc
        
        # predict P(Y|X)
        pyx = self.CDE_model.predict(dataset)

        # partition X and Y with P(Y|X)
        xlbls, ylbls = self.cluster_model.predict(dataset)

        return xlbls, ylbls
