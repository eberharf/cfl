
from cfl.block import Block
import cfl.cond_density_estimation as c


class CondDensityEstimator(Block):

    def __init__(self, data_info, block_params):
        # parameter checks and self.params assignment done here
        super().__init__(data_info=data_info, block_params=block_params)

        # attributes:
        self.name = 'CondDensityEstimator'
        self.model = self._create_model()

    def _create_model(self):
        if isinstance(self.block_params['model'], str):
            # create model
            # TODO: this is hacky
            model = eval('c.' + self.block_params['model'])(self.data_info, 
                self.block_params['model_params'])
        else:
            model = self.block_params['model']
        return model

    def get_block_params(self):
        ''' 
        TODO
        '''
        return self.block_params()

    def _get_default_block_params(self):
        """ Private method that specifies default CDE parameters.

            Arguments: None
            Returns: 
                dict: dictionary of parameter names (keys) and values (values)

        """
        return  {'model'   : 'CondExpMod',
                 'model_params' : {},
                 'verbose' : 1 }

    def train(self, dataset, prev_results):
        """
        TODO
        """
        return self.model.train(dataset, prev_results)

    def predict(self, dataset, prev_results):
        """  
        TODO
        """
        return self.model.predict(dataset, prev_results)

    ############ SAVE/LOAD FUNCTIONS (required by block.py) ###################

    def save_block(self, file_path):
        ''' 
        TODO
        '''
        self.model.save_model(file_path)

    def load_block(self, file_path):
        '''
        TODO
        '''
        self.model.load_model(file_path)
