# Oracles
Implement a few oracles, including KNN, RandomForest, LSTM and Transformer.

## Files
* metric.py, implement some evaluation metrics, which are shared across different classifiers.
* utils.py, define some common variables and methods.
* nn_model.py, consisting of some neural classifiers.
    ** LSTM, a bi-directional lstm is used for classification.
    ** Transformer, a Transformer encoder is used for classifiers.
* nn_main.py, running file for neural models.
* kNN.py, implement a kNN algorithm for sequence classification.
* randomForest.py, implement a RandomForest algorithm for sequence classification.

## Environments
The codebase is developed under following requirements:
- Pytorch=1.8.1
- Sklearn=0.24.2
- Numpy=1.20.2
- Pandas=1.2.4


## Usages

utils.py is shared across all methods, please set the data path and model saving path correspongdingly before use.
All models run in a command line fashion. Configurations are given below:

### kNN
> --mode, running mode, 0:train, 1:test, 2:query.
> 
> --feat, indicating the input is raw sequence or sequence features. 0:sequence, 1:features.
> 
> --n_neighbors, number of neighbors used for classification.
>
> --n_jobs, number of threads available.
> 
> --weights, a parameter used in kNN classifier. 
>
> --n_gram_max, used if the input is raw sequences.
>
> --n_gram_min, used if the input is raw sequences. 

### RandomForest
> --mode, running mode, 0:train, 1:test, 2:query.
> 
> --feat, indicating the input is raw sequence or sequence features. 0:sequence, 1:features.
>
> --n_trees, number of trees used in a forest.
> 
> --depth, tree depth.
> 
> --n_jobs, number of threads available.
> 
> --n_gram_max, used if the input is raw sequences.
>
> --n_gram_min, used if the input is raw sequences. 
> 
> --pos_weight, a parameter used in randomforest classifier. 

Note: if the inputs for kNN and RandomForest are sequence feature (e.g., extracted by pre-trained LM), the format should be label,n1,n2,...

### LSTM \& Transformer
> --epochs, number of epochs for training.
> 
> --nhead, number of attention heads if use Transformer.
> 
> --batch_size, size of a batch.
> 
> --embedding_size, embedding dimension for each token.
> 
> --hidden_size, number of hidden units.
> 
> --output_size, default is 1, for binary classification.
> 
> --num_layers, network depth.
> 
> --dropout, dropout rate.
> 
> --lr, initial learning rate.
> 
> --weight, sampling weight to control the ratio of pos/neg samples.
> 
> --clip, gradients are in [-clip, clip].
> --model, select model in [lstm, tfr].