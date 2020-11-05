#include "vocab.h"
#include "w2v.h"
#include <iostream>
#include <string>

using namespace std;
using namespace vocab;
using namespace w2v;

int ArgPos(string str, int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (!str.compare(argv[i]))
        {
            if (i == argc - 1)
            {
                printf("Argument missing for %s\n", str.c_str());
                exit(1);
            }
            return i;
        }
    return -1;
}

int main(int argc, char **argv)
{
    string train_file_path, out_vectors_path = "./vectors.txt",
                            out_vocab_path = "./vocab.txt", in_vocab_path = "";
    int i, emb_dim = 100, window_size = 5, ns_size = 5,
           max_sentence_length = 1000, epochs = 5, num_workers = 4,
           min_count = 5;
    float unigram_pow = 0.75, sample = 1e-3, init_lr = 0.025;
    bool cbow = 1, shrink_window_size = 1;
    long long unigram_table_size = 1e8;
    long log_freq = 10000;
    if (argc == 1)
    {
        printf("Options:\n");
        printf("Parameters for training:\n");

        printf("\t-train-file-path <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");

        printf("\t-out-vectors-path <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors; default is "
               "'./vectors.txt'\n");

        printf("\t-out-vocab-path <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>; default is "
               "'./vocab.txt'\n");

        printf("\t-in-vocab-path <file>\n");
        printf("\t\tThe vocabulary will loaded from <file>; default is "
               "''\n");

        printf("\t-emb-dim <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");

        printf("\t-window-size <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");

        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; "
               "default is 5\n");

        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values "
               "are 3 - 10 (0 = not used)\n");

        printf("\t-unigram-table-size <float>\n");
        printf("\t\tSet size of the unigram table; default is 1e8\n");

        printf("\t-unigram-pow <float>\n");
        printf("\t\tSet the power of the unigram distribution for the negative "
               "sampling; default is 0.75\n");

        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear "
               "with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful "
               "range is (0, 1e-5)\n");

        printf("\t-max-sentence-length <int>\n");
        printf("\t\tMax number of words in a sentence to be processed; default "
               "is 1000\n");

        printf("\t-num-workers <int>\n");
        printf("\t\tUse <int> threads (default 4)\n");

        printf("\t-epochs <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");

        printf("\t-lr <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for "
               "skip-gram and 0.05 for CBOW\n");

        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 1 (use 0 "
               "for skip-gram model)\n");

        printf("\t-log-freq <int>\n");
        printf("\t\tLog training information every <int> processed words; "
               "default is 10000\n");

        printf("\nExamples:\n");
        printf("./word2vec -train-file-path data.txt -emb-dim 100 "
               "-window-size 5 "
               "-sample 1e-3 -negative 5 -cbow 1 -epochs 3\n\n");
        return 0;
    }

    if ((i = ArgPos("-emb-dim", argc, argv)) > 0)
        emb_dim = atoi(argv[i + 1]);
    if ((i = ArgPos("-train-file-path", argc, argv)) > 0)
        train_file_path = argv[i + 1];
    if ((i = ArgPos("-out-vocab-path", argc, argv)) > 0)
        out_vocab_path = argv[i + 1];
    if ((i = ArgPos("-in-vocab-path", argc, argv)) > 0)
        in_vocab_path = argv[i + 1];
    if ((i = ArgPos("-cbow", argc, argv)) > 0)
        cbow = atoi(argv[i + 1]);
    if (cbow)
        init_lr = 0.05;
    if ((i = ArgPos("-negative", argc, argv)) > 0)
        ns_size = atoi(argv[i + 1]);
    if ((i = ArgPos("-lr", argc, argv)) > 0)
        init_lr = atof(argv[i + 1]);
    if ((i = ArgPos("-out-vectors-path", argc, argv)) > 0)
        out_vectors_path = argv[i + 1];
    if ((i = ArgPos("-window-size", argc, argv)) > 0)
        window_size = atoi(argv[i + 1]);
    if ((i = ArgPos("-sample", argc, argv)) > 0)
        sample = atof(argv[i + 1]);
    if ((i = ArgPos("-num-workers", argc, argv)) > 0)
        num_workers = atoi(argv[i + 1]);
    if ((i = ArgPos("-epochs", argc, argv)) > 0)
        epochs = atoi(argv[i + 1]);
    if ((i = ArgPos("-min-count", argc, argv)) > 0)
        min_count = atoi(argv[i + 1]);
    if ((i = ArgPos("-max-sentence-length", argc, argv)) > 0)
        max_sentence_length = atoi(argv[i + 1]);
    if ((i = ArgPos("-unigram-pow", argc, argv)) > 0)
        unigram_pow = atof(argv[i + 1]);
    if ((i = ArgPos("-unigram-table-size", argc, argv)) > 0)
        unigram_table_size = atoi(argv[i + 1]);
    if ((i = ArgPos("-log-freq", argc, argv)) > 0)
        log_freq = atoi(argv[i + 1]);

    cout << "\nRecap parameters\n";
    cout << "Embedding dimension: " << emb_dim << '\n';
    cout << "Window size: " << window_size << '\n';
    cout << "Min count: " << min_count << '\n';
    cout << "Number of negative samples: " << ns_size << '\n';
    cout << "Sentence length: " << max_sentence_length << '\n';
    cout << "Sample: " << sample << '\n';
    cout << "Lr: " << init_lr << '\n';
    cout << "Epochs: " << epochs << '\n';
    cout << "Log info every " << log_freq << " words" << '\n';
    cout << "Training with the " << (cbow ? "CBOW" : "SKIP-GRAM")
         << " algorithm\n"
         << '\n';

    Word2Vec w2v = Word2Vec(train_file_path, out_vectors_path, out_vocab_path,
                            in_vocab_path, emb_dim, min_count, window_size,
                            ns_size, max_sentence_length, epochs, num_workers,
                            unigram_pow, sample, init_lr, cbow,
                            shrink_window_size, unigram_table_size, log_freq);
    w2v.train();
    w2v.save_vectors();
}