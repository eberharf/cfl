
from cfl.block import Block
import cfl.cond_density_estimation as c


class CondDensityEstimator(Block):

    def __init__(self, data_info, params):
        # parameter checks and self.params assignment done here
        super().__init__(data_info=data_info, params=params)

        # attributes:
        self.name = 'CondDensityEstimator'
        self.model = self._create_model()

    def _create_model(self):

        if self.params == {}:
            self.params = self._get_default_params()

        if 'model' not in self.params.keys():
            raise KeyError(
                'if any parameters are specified, `model` must be specified as well.')

        if isinstance(self.params['model'], str):
            # pull dict entries to pass into clusterer object
            excluded_keys = ['model']
            model_keys = list(set(self.params.keys()) - set(excluded_keys))
            model_params = {key: self.params[key] for key in model_keys}

            # create model
            # TODO: this is hacky
            model = eval('c.' + self.params['model']
                         )(self.data_info, model_params)
        else:
            model = self.params['model']
        return model

    def get_params(self):
        ''' 
        TODO
        '''
        return self.model.get_params()

    def _get_default_params(self):
        """ Private method that specifies default CDE parameters.

            Arguments: None
            Returns: 
                dict: dictionary of parameter names (keys) and values (values)

        """
        return {'model': 'CondExpMod'}

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
        self.model.save_block(file_path)

    def load_block(self, file_path):
        '''
        TODO
        '''
        self.model.load_block(file_path)
