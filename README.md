# word2vecpp

A C++ word2vec implementation, following the training style of https://github.com/tmikolov/word2vec.git  
This software has been tested on **Pop_OS 20.10** (an Ubuntu patch) with **g++ (Ubuntu 10.2.0-13ubuntu1) 10.2.0**

## Installation

To install and run the software:

1. git clone https://github.com/belerico/word2vecpp.git
2. make
3. ./word2vec \<parameters\>

where the possible parameters are listed below:

```
Options:
Parameters for training:
        -train-file-path <file>
                Use text data from <file> to train the model
        -out-vectors-path <file>
                Use <file> to save the resulting word vectors; default is './vectors.txt'
        -out-vocab-path <file>
                The vocabulary will be saved to <file>; default is './vocab.txt'
        -in-vocab-path <file>
                The vocabulary will loaded from <file>; default is ''
        -emb-dim <int>
                Set size of word vectors; default is 100
        -window-size <int>
                Set max skip length between words; default is 5
        -min-count <int>
                This will discard words that appear less than <int> times; default is 5
        -negative <int>
                Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
        -unigram-table-size <float>
                Set size of the unigram table; default is 1e8
        -unigram-pow <float>
                Set the power of the unigram distribution for the negative sampling; default is 0.75
        -sample <float>
                Set threshold for occurrence of words. Those that appear with higher frequency in the training data
                will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
        -max-sentence-length <int>
                Max number of words in a sentence to be processed; default is 1000
        -num-workers <int>
                Use <int> threads (default 4)
        -epochs <int>
                Run more training iterations (default 5)
        -lr <float>
                Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
        -cbow <int>
                Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)
        -log-freq <int>
                Log training information every <int> processed words; default is 10000

Examples:
./word2vec -train-file-path data.txt -emb-dim 100 -window-size 5 -sample 1e-3 -negative 5 -cbow 1 -epochs 3
```

As an example, one can train a word2vec model on the [text8](http://mattmahoney.net/dc/textdata.html) dataset running the following commands:

1. chmod +x train_text8.sh
2. ./train_text8.sh

This will download and unzip the text8 dataset in the current directory and train a word2vec CBOW model