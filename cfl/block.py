

class Block():
    '''
    A Block is an object that can be trained and that can:
        1) be trained on a Dataset
        2) predict some target for a Dataset.
    Blocks are intended to be components of a graph workflow in an Experiment. 
    For example, if the graph Block_A->Block_B is constructed in an Experiment, 
    the output of Block_A.predict will provide input to Block_B.predict. 

    Attributes:
        trained : 
        name : 
        model :
    
    Methods:
        train : 
        predict : 
        get_name : 
        is_trained : 
        get_model_by_name : 

    '''

    
    def __init__(self, name):
        '''
        Instantiate the specified model. 

        Arguments:
            name : name of the model to instantiate (str)
        
        Returns: None
        '''
        self.trained = False
        self.name = name
        self.model = self.get_model_by_name(name)
    
    
    def train(self, dataset, prev_results=None):
        '''
        Train model attribute.

        Arguments:
            dataset : dataset to train model with (Dataset)
            prev_results : any results needed from previous Block training (dict)
        '''
        results = self.model.train(dataset, prev_results)
        self.trained = True
        return results
    
    def predict(self, dataset, prev_results=None):
        '''
        Make prediction for the specified dataset with the model attribute.

        Arguments:
            dataset : dataset for model to predict on (Dataset)
            prev_results : any results needed from previous Block prediction (dict)
        '''
        if self.trained:
            return self.model.predict(dataset, prev_results)
        else:
            raise Exception('Block needs to be trained before prediction.')

    def get_name(self):
        '''
        '''
        return self.name

    def is_trained(self):
        '''
        '''
        return self.trained
    
    def get_model_by_name(self, name):
        '''
        '''
        return None