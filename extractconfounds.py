import pandas as pd

def extract_confounds(confound_tsv, confounds, dt=True):
    '''
    Arguments:
        confound_tsv                    Full path to confounds.tsv
        confounds                       A list of confounder variables to extract
        dt                              Compute temporal derivatives [default = True]

    Outputs:
        confound_mat
    '''

    if dt:
        dt_names = ['{}_derivative1'.format(c) for c in confounds]
        confounds = confounds + dt_names

    # Load in data using Pandas then extract relevant columns
    confound_df = pd.read_csv(confound_tsv, delimiter='\t')
    confound_df = confound_df[confounds]

    # Convert into a matrix of values (timepoints)x(variable)
    confound_mat = confound_df.values

    # Return confound matrix
    return confound_mat
