import time 

import torch 
import numpy as np 
import jellyfish
import sklearn 
from sklearn.metrics.pairwise import cosine_similarity


def cal_seq2seq_string_distance(dataset):
    N = len(dataset)
    seqs = dataset['text']
    for x in seqs:
        dhx, dlx = 0, 0
        for y in seqs:
            l = max(len(x), len(y))
            dhx += jellyfish.hamming_distance(x, y) / l 
            dlx += jellyfish.levenshtein_distance(x, y) / l 
        rh.append(dhx / N)
        rl.append(dlx / N)
    return np.array(rh), np.array(rl)


def _get_topk_avg(mat, topk_list):
    '''
        Get the average diversity, where the diversity of one sequence is 
        estimated as the average top-k distance with other sequences.
    '''
    ret = []
    if isinstance(mat, np.ndarray):
        for top_k in topk_list:
            partitioned_mat = np.partition(mat, kth=top_k-1, axis=-1)
            div_score = np.sum(partitioned_mat[:, :top_k], axis=-1) / top_k 
            ret.append(div_score)
    elif isinstance(mat, torch.Tensor):
        for top_k in topk_list:
            div_score = torch.sum(torch.topk(mat, top_k, dim=-1, largest=False).values, dim=-1) / top_k 
            div_score = div_score.cpu().numpy()
            ret.append(div_score)
    else:
        raise NotImplementedError
    return ret # list of np.ndarray()

def cal_seq2seq_embedding_distance(args, dataset, topk_list=[1, 10, 100,]):

    topk_list.append(len(dataset))

    feat_embed = dataset[args.feature_type]
    mat = cosine_similarity(feat_embed)

    ret = _get_topk_avg(mat, topk_list)
    return ret 


