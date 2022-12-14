import numpy as np

#Functions for generating the shifter problem
def one_hot_number(number, total_digits = 3):
    one_array = np.zeros(total_digits)
    one_array[int(number)] = 1
    return one_array

def create_shifter(n, orig_shifter_length):
    labels = np.random.randint(3, size = n)

    samples = np.zeros((n, orig_shifter_length * 2+3))
    X_val = np.zeros((n, orig_shifter_length * 2))
    for i in np.arange(n):
        one_sample = np.random.randint(2, size=8)
        shifted_sample = np.roll(one_sample, labels[i]-1)
        X_val[i] =  np.concatenate((one_sample, shifted_sample))
        samples[i] = np.concatenate((X_val[i], one_hot_number(labels[i])))
        samples = samples.astype(int)
    return samples