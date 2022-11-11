import numpy as np




def bootstrap(outputs, labels, func, sampling_times=500, c=0.95):

    outputs = np.array(outputs)
    labels = np.array(labels)
    samples_num = len(outputs)
    list_sampling_result = []
    for i in range(sampling_times):
        index_arr = np.random.randint(0, samples_num, samples_num)
        sample_result = func(outputs[index_arr], labels[index_arr])
        list_sampling_result.append(sample_result)

    a = 1 - c
    k1 = int(sampling_times * a / 2)
    k2 = int(sampling_times * (1 - a / 2))
    list_sampling_result = sorted(list_sampling_result)

    lower = list_sampling_result[k1]
    higher = list_sampling_result[k2]

    return lower, higher


