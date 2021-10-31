'''
Credit: https://github.com/clamp-gen/common-evaluation/blob/main/clamp_common_eval/oracles/CTDD.py
'''

import math
import re
from tqdm import tqdm
import numpy as np

def Count(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code

# modified version to avoid some overhead
group1 = {
    'hydrophobicity_PRAM900101': 'RKEDQN',
    'hydrophobicity_ARGP820101': 'QSTNGDE',
    'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
    'hydrophobicity_PONP930101': 'KPDESNQT',
    'hydrophobicity_CASG920101': 'KDEQPSRNTG',
    'hydrophobicity_ENGD860101': 'RDKENQHYP',
    'hydrophobicity_FASG890101': 'KERSQD',
    'normwaalsvolume': 'GASTPDC',
    'polarity': 'LIFWCMVY',
    'polarizability': 'GASDT',
    'charge': 'KR',
    'secondarystruct': 'EALMQKRH',
    'solventaccess': 'ALFCGIVW'
}
group2 = {
    'hydrophobicity_PRAM900101': 'GASTPHY',
    'hydrophobicity_ARGP820101': 'RAHCKMV',
    'hydrophobicity_ZIMJ680101': 'HMCKV',
    'hydrophobicity_PONP930101': 'GRHA',
    'hydrophobicity_CASG920101': 'AHYMLV',
    'hydrophobicity_ENGD860101': 'SGTAW',
    'hydrophobicity_FASG890101': 'NTPG',
    'normwaalsvolume': 'NVEQIL',
    'polarity': 'PATGS',
    'polarizability': 'CPNVEQIL',
    'charge': 'ANCQGHILMFPSTWYV',
    'secondarystruct': 'VIYCWFT',
    'solventaccess': 'RKQEND'
}
group3 = {
    'hydrophobicity_PRAM900101': 'CLVIMFW',
    'hydrophobicity_ARGP820101': 'LYPFIW',
    'hydrophobicity_ZIMJ680101': 'LPFYI',
    'hydrophobicity_PONP930101': 'YMFWLCVI',
    'hydrophobicity_CASG920101': 'FIWC',
    'hydrophobicity_ENGD860101': 'CVLIMF',
    'hydrophobicity_FASG890101': 'AYHWVMFLIC',
    'normwaalsvolume': 'MHKFRYW',
    'polarity': 'HQRKNED',
    'polarizability': 'KMHFRYW',
    'charge': 'DE',
    'secondarystruct': 'GNPSD',
    'solventaccess': 'MSPTHY'
}

_property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
    'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

def CTDD_fast(sequence, **kw):
    code = []
    for p in _property:
        code = code + Count(group1[p], sequence) + Count(group2[p], sequence) + Count(group3[p], sequence)
    return code

def CTDD_array(sequences, **kw):
    ''' CTDD feature extraction function
        Parameters:
            sequences : list of str, a list of str represent the amino acids
        Return:
            list, a list of numpy.Array
    '''
    encodings = []
    #for seq in tqdm(sequences, desc='CTDD feature extraction'):
    for seq in sequences:
        code = []
        for p in _property:
            code = code + Count(group1[p], seq) + Count(group2[p], seq) + Count(group3[p], seq)
        encodings.append(code)
    return np.array(encodings)