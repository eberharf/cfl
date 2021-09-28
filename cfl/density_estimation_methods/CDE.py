
from cfl.block import Block
import cfl.density_estimation_methods as c

class CDE(Block):
    
    def __init__(self, data_info, params):
        # parameter checks and self.params assignment done here 
        super().__init__(data_info=data_info, params=params) 
        
        #attributes:
        self.name = 'CDE'
        self.model = self._create_model()

    def _create_model(self):

        if self.params == {}:
            self.params = self._get_default_params()
        
        if 'model' not in self.params.keys():
            raise KeyError('if any parameters are specified, `model` must be specified as well.') 

        if isinstance(self.params['model'], str):
            # pull dict entries to pass into clusterer object
            excluded_keys = ['model']
            model_keys = list(set(self.params.keys()) - set(excluded_keys))
            model_params = {key: self.params[key] for key in model_keys}

            # create model
            # TODO: this is hacky
            model = eval('c.' + self.params['model'])(self.data_info, model_params)
        else:
            model = self.params['model']
        return model


    def get_params(self):
        ''' 
        TODO
        '''
        return self.model.get_params()

    def _get_default_params(self):
        """ Private method that specifies default clustering method parameters.
            Note: clustering method currently defaults to DBSCAN. While DBSCAN
            is a valid starting method, the choice of clustering method is
            highly dependent on your dataset. Please do not rely on the defaults
            without considering your use case.

            Arguments: None
            Returns: 
                dict: dictionary of parameter names (keys) and values (values)

        """
        return {'model' : 'CondExpMod',
                'batch_size'  : 32,
                'n_epochs'    : 20,
                'optimizer'   : 'adam',
                'opt_config'  : {},
                'verbose'     : 1,
                'dense_units' : [50, self.data_info['Y_dims'][1]],
                'activations' : ['relu', 'linear'],
                'dropouts'    : [0, 0],
                'weights_path': None,
                'loss'        : 'mean_squared_error',
                'show_plot'   : True,
                'standardize' : False,
                'best'        : True,
                'tb_path'     : None,
            }


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