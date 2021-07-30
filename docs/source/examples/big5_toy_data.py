'''this script generates a synthetic data set that 
you can use for trying out CFL.

Example Usage: 
    import big5_toy_data as btd
    X, Y = btd.big5_for_cfl()
    << and then run CFL >>
    
'''

import numpy as np
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

RANDOM_SEED = 42
N_SAMPLES = 2000


def big5_for_cfl(): 
    '''run this function to generate Big 5 synthetic data for cfl'''

    # generate X data 
    big5 = download_cause_data()
    big5_consistent = flip_even_values(big5)

    X = big5_consistent.to_numpy()

    # generate Y data 
    Y = generate_effect_data(big5_consistent)

    return X, Y



# code for downloading file from internet modified from 
# https://stereopickle.medium.com/how-to-download-unzip-zip-files-in-python-5f326bb1a829
def download_cause_data(): 
    # initiating a dataframe
    big5_df = pd.DataFrame()

    # download big 5 personality data 
    data_url = 'http://openpsychometrics.org/_rawdata/BIG5.zip'
    resp = urlopen(data_url)

    # read zipfile
    zipfile = ZipFile(BytesIO(resp.read()))

    # get the csv file name
    fname = zipfile.namelist()[1]

    # convert to pandas dateframe
    big5_df = pd.read_csv(zipfile.open(fname), 
                    sep='\t',  #tab-delimited
                    lineterminator='\n', #new line at the end of row 
                    usecols=range(7, 57), #only load the questionnaire results, drop demographic info 
                    nrows=N_SAMPLES) # load first N_SAMPLES rows  

    # close zipfile we don't need
    zipfile.close()

    return big5_df

def flip_even_values(big5_df): 
    ''' On the Big 5 personality questionnaire, every even numbered question
should be reversed (ie a 5 score on E2 indicates a 1 on extraversion). This
function flips those values so that for every question, a 1 indicates minimum
value of that trait and 5 indicates the max 

Parameters: 
    big5_df (pandas df): data frame with big 5 personality traits. Rows indicate
    subjects, columns indicate questions 

Returns: 
    (pd data frame): the same data frame but with even values flipped 
'''
    def subtract_from_6(n):
        '''flips scale from 5-1 to 1-5''' 
        return 6 - n 

    for col in big5_df.columns: 
        no_whitespace = col.strip() # remove whitespace 
        is_even = (int(no_whitespace[1:]) % 2 == 0) #get just the question number and check if even 
        if is_even: 
            # flip all even columns 
            big5_df[col]=big5_df[col].map(subtract_from_6)
    return big5_df

def generate_effect_data(big5_df): 
    '''
    Generates a target distribution for a toy CFL set-up based on the below
    equation: 
    This target variable should be thought of as a 'hypothesized summary
    statistic of general well-being'. In running CFL on this data, we are 
    trying to figure out whether the big 5 personality traits (as measured
    by this questionnaire) are causally relevant for this summary statistic
    (hint: they are, because this statistic is fake and is a function of those 
    traits). 
    
    equation: T = 0.5*openness + 0.5* extraversion + 0.35*conscientiousness +
    0.35*agreeableness  + N(0, 0.2)

    Parameters: 
        big5_df (pandas df): a data frame with big 5 personality traits. The `X`/
            causal data set 

    Returns: 
        (np array): an array of `Y`/effect data. The target is a continuous
            (single) variable, which is a linear combination of different personality traits 
    '''

    # find the mean across all questions of each trait for each sample 
    big5_means = np.zeros((N_SAMPLES, 5)) # 5 traits 
    for i, trait in enumerate(['E', 'N', 'A', 'C', 'O']): 
        cols = [col for col in big5_df.columns.values if trait in col] # get columns that correspond to each trait 
        big5_means[:, i] = big5_df[cols].mean(axis=1)

    big5_weights_matrix = np.array([0.5, 0, 0.35, 0.35, 0.5]) # traits in order E N A C O 
    
    # generate noise term (normal distribution with mean 0 and var 0.2) for each sample
    rng = np.random.default_rng(RANDOM_SEED) #reproducible randomness (same dataset)
    noise = rng.normal(loc=0, scale=0.2, size=(N_SAMPLES, 1))

    target = big5_means + noise 
    return target 


