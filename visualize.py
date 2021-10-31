import numpy as np
import pandas as pd 


import matplotlib 
matplotlib.use("agg")
from matplotlib import pyplot as plt 
import seaborn as sns 
from sklearn.manifold import TSNE 

def plot_dict2hist(d, fig_title='N-gram Counting'):
    '''
    Params:
        d: dict of (n-gram string, counts)
    '''
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

    fig = plt.figure(figsize=(20.0, 8.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(fig_title)
    ax.scatter([_ for _ in range(len(d))], [v for k,v in d.items()])
    for i, (k, v) in enumerate(d.items()):
        ax.annotate(k, (i, v))
    plt.savefig('./dict2hist.png')
    plt.close()


def plot_TSNE(args, dataset, fig_title='TSNE for Generated Sequences'):
    feat_embed = dataset['feat_'+args.feature_type]
    if len(feat_embed) > 10000:
        print("[Warning]: tsne could be super slow when processsing large data, we downsampled this data")
        idx = np.random.choice(len(feat_embed), 10000, replace=False)
    
    tsne = TSNE(n_components=2, random_state=args.seed, init='pca')
    tsne_embed = tsne.fit_transform(feat_embed)
    labels = dataset['prob']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(fig_title)
    ax.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=labels)
    plt.savefig('tsne.png')
    plt.close()
