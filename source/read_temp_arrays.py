import numpy as np

def read_arrays(filename):
    with open(filename, 'r') as f:
        data = f.read().strip()

    parts = data.split('\n\n')
    arrs = []
    for part in parts:
        if '[' in part and ']' in part:
            arr = np.array([float(x) for x in part.strip('[]').split()])
        else:
            arr = np.array([[float(num) for num in line.split()] 
                            for line in part.splitlines()])
        arrs.append(arr)
    return arrs

def calc_chisq(residual, covmat):
    return np.matmul(np.matmul(residual, covmat), residual)/2