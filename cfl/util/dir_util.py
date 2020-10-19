
import os

def get_next_dirname(path):
    ''' gets the next subdirectory name in numerical order. i.e. if  'path' 
    contains 'run0000' and 'run0001', this will return 'run0002'. 
    Arguments: 
        path: path of directory in which to find next subdirectory name (string)
    Returns:
        next subdirectory name. 
    '''
    i = 0
    while os.path.exists(os.path.join(path, 'run{}'.format(str(i).zfill(4)))):
        i += 1  
    return 'run{}'.format(str(i).zfill(4))


