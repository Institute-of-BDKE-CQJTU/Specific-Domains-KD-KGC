# Specific-Domains-KD-KGC
## Data
The data can be downloaded in [Google Drive](https://drive.google.com/file/d/1pAgYnabv57IkOecdnnRWBeMNZOb4Fgo1/view?usp=sharing)
## Code
1. put your arguments in the json files <code> ./config/*.json </code>, e.g. <code> WN18.json </code>
2. execute command, <code> python3 run.py </code>
#### Code files
Python files:
* <code> configure.py</code>: including all hyper parameter, reading arguments from <code> ./config/*.json </code>;
* <code> data.py</code>: dataloader, a KG class containing all data in a dataset;
* <code> lineare.py</code>: the implementation of the `Student` model;
* <code>lineare_rule_mtach.py</code>: the implementation of the `Teacher` model;
* <code>teacher.py</code>: train the Teacher Net;
* <code> student.py</code>: train the Student Net;
* <code>utils.py</code>: global tools 
* <code>extract_rule.py</code> : extract rules from original Knowledge Graph
#### Dependencies
* Python 3.8
* PyTorch 1.7
* Numpy

## Datasets
Datasets: WN18, Med-KGC
 - *entities.dict*: a dictionary map entities to unique ids;
 - *relations.dict*: a dictionary map relations to unique ids;
 - *train.txt*: the KGE model is trained to fit this data set;
 - *valid.txt*: create a blank file if no validation data is available;
 - *test.txt*: the KGE model is evaluated on this data set.
 - <code>***__rules.txt</code>: extracted rules.

## Parameters(./config/*.json)
 - *dim*: the dimmention (size) of embeddings,
 - *norm_p*: L1-Norm or L2-Norm,
 - *alpha*: the temperature of Self-Adversarial Negative Sampling [[1](#refer-1)], Eq(3) in our paper,
 - *beta*: a hyper-parameter in softplus(pytorch)
 - *gamma*: a hyper-parameter in loss the function, used to separate the positive sample from the negative sample,
 - lambda: a hyper-parameter in total loss to control the proportion of loss_1 and loss_2
 - *learning_rate*: initial learning rate, decaying during training.
 - *decay_rate*: learning rate decay rate,
 - *batch_size*: 
 - *neg_size*: the number of negative samples for each positive sample in an optimization step,
 - *regularization*: the regularization coefficient,
 - *drop_rate*: some dimensions of embeddings are randomly dropped with probability 'drop_rate',
 - *test_batch_size*:,
 - *data_path*: root of dataset,
 - *save_path*: root of model parameters' checkpoint file,
 - *max_step*: total training steps,
 - *valid_step*: valid the model every 'valid_step' training steps,
 - *log_step*: logging the average loss value every 'log_step' training steps,
 - *test_log_step*: ,
 - *optimizer*: SGD, Adam ...,
 - *init_checkpoint*: whether load model from checkpoint file or not,
 - *use_old_optimizer*: if init_checkpoint is True, load the stored optimizer or use a new optimizer,
 - *sampling_rate*: assign a weight for each triple, like word2vec,
 - *sampling_bias*: assign a weight for each triple, like word2vec,
 - *device*: 'cuda:0', cpu...,
 - *multiGPU*: use multiple GPUs?

