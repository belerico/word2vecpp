# word2vecpp

A C++ word2vec implementation, following the training style of https://github.com/tmikolov/word2vec.git  
This software has been tested on **Pop_OS 20.10** (an Ubuntu patch) with **g++ (Ubuntu 10.2.0-13ubuntu1) 10.2.0**

## Installation & Run

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

## Results

All the test reported below are run thanks to https://github.com/mfaruqui/eval-word-vectors script.  
Model hyperparameters for both tests:

- window size: 5
- min count: 5
- negative size: 10
- embedding dimension: 100
- others by default

### Results CBOW

| Dataset             | Num Pairs | Not found | Rho (Mine) | Rho (Gensim) | Rho (Mikolov) |
|---------------------|-----------|-----------|------------|--------------|---------------|
| EN-MC-30.txt        | 30        | 0         | 0.5450     | 0.5277       | 0.6047        |
| EN-MTurk-771.txt    | 771       | 2         | 0.5451     | 0.5516       | 0.5468        |
| EN-SimVerb-3500.txt | 3500      | 255       | 0.1405     | 0.1428       | 0.1432        |
| EN-WS-353-ALL.txt   | 353       | 2         | 0.6611     | 0.6632       | 0.6736        |
| EN-WS-353-REL.txt   | 252       | 1         | 0.6283     | 0.6293       | 0.6400        |
| EN-YP-130.txt       | 130       | 12        | 0.2501     | 0.2746       | 0.2652        |
| EN-VERB-143.txt     | 144       | 0         | 0.3427     | 0.3607       | 0.3609        |
| EN-MEN-TR-3k.txt    | 3000      | 13        | 0.5869     | 0.5884       | 0.5963        |
| EN-RW-STANFORD.txt  | 2034      | 1083      | 0.3373     | 0.3277       | 0.3282        |
| EN-MTurk-287.txt    | 287       | 3         | 0.6363     | 0.6373       | 0.6448        |
| EN-RG-65.txt        | 65        | 0         | 0.5642     | 0.5173       | 0.5613        |
| EN-WS-353-SIM.txt   | 203       | 1         | 0.6991     | 0.6971       | 0.7153        |
| EN-SIMLEX-999.txt   | 999       | 7         | 0.2626     | 0.2659       | 0.2729        |                 

### Results Skip-Gram

| Dataset             | Num Pairs | Not found | Rho (Mine) | Rho (Gensim) | Rho (Mikolov) |
|---------------------|-----------|-----------|------------|--------------|---------------|
| EN-MC-30.txt        | 30        | 0         | 0.6674     | 0.5733       | 0.6650        |
| EN-MTurk-771.txt    | 771       | 2         | 0.5655     | 0.5641       | 0.5603        |
| EN-SimVerb-3500.txt | 3500      | 255       | 0.1702     | 0.1747       | 0.1731        |
| EN-WS-353-ALL.txt   | 353       | 2         | 0.6755     | 0.6649       | 0.6798        |
| EN-WS-353-REL.txt   | 252       | 1         | 0.6483     | 0.6341       | 0.6496        |
| EN-YP-130.txt       | 130       | 12        | 0.3325     | 0.3497       | 0.3220        |
| EN-VERB-143.txt     | 144       | 0         | 0.3738     | 0.3467       | 0.3828        |
| EN-MEN-TR-3k.txt    | 3000      | 13        | 0.6042     | 0.6097       | 0.6064        |
| EN-RW-STANFORD.txt  | 2034      | 1083      | 0.3833     | 0.3774       | 0.3861        |
| EN-MTurk-287.txt    | 287       | 3         | 0.6530     | 0.6359       | 0.6446        |
| EN-RG-65.txt        | 65        | 0         | 0.6215     | 0.5823       | 0.6324        |
| EN-WS-353-SIM.txt   | 203       | 1         | 0.7316     | 0.7085       | 0.7260        |
| EN-SIMLEX-999.txt   | 999       | 7         | 0.2985     | 0.3150       | 0.2983        |

## Future works

- Implement a function to instantly retrieve a word vector
- Implement distance functions between vectors, such as cosine or L2
- Implement a function to get nearest neighbours given a word
