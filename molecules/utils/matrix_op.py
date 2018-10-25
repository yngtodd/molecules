import numpy as np 

def triu_to_full(cm0): 
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0 
    cm_full.T[iu1] = cm0 
    np.fill_diagonal(cm_full, 1)
    
    return cm_full
