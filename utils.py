import numpy as np
import torch


# raw_predict_matching  bx5x18  np.array
def to_predict_matching(raw_predict_matching):
    shape = raw_predict_matching.shape
    zeros = np.zeros(shape)
    dim2 = np.arange(shape[1])
    dim3 = np.argmax(raw_predict_matching, 2)
    for i in np.arange(shape[0]):
        zeros[i, dim2, dim3[i]] = 1
    m = zeros
    return m


def acc_renzao(predict_matching, q_size, da_size):
    shape = predict_matching.shape
    predict_matching = np.sum(predict_matching, 1)
    predict_matching = predict_matching >= 1
    predict_matching = predict_matching.astype(float).reshape(-1)

    label = np.zeros(da_size)
    label[0:q_size] = 1
    label = torch.tensor(label)
    label = label.repeat(shape[0])
    label = label.numpy()

    acc = predict_matching == label
    acc = acc.astype(float)
    acc = np.mean(acc)
    return acc


def eval_mapping(groundtruth, predict_list):
    acc = []
    MRR = []

    for sgn in groundtruth:
        # Calculate precision
        list_acc = [0] * 10
        for i in range(1, 11):
            if groundtruth[sgn] in predict_list[sgn][:i]:
                list_acc[i - 1] = 1

        acc.append(list_acc)

        if groundtruth[sgn] in predict_list[sgn]:
            MRR.append(1 / (predict_list[sgn].index(groundtruth[sgn]) + 1))
        else:
            MRR.append(0)

    acc = np.mean(np.array(acc), axis=0)
    MRR = np.mean(np.array(MRR))
    return np.concatenate([acc, np.array([MRR])])
