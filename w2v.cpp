#include "w2v.h"
#include "utils.h"
#include "vocab.h"
#include <atomic>
#include <chrono>
#include <fstream> // std::ifstream
#include <iostream>
#include <math.h>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace utils;
using namespace vocab;

#define MAX_EXP 6
#define EXP_TABLE_SIZE 1000

namespace w2v
{
Word2Vec::Word2Vec(string train_file_path, string out_vectors_path,
                   string out_vocab_path, string in_vocab_path, int emb_dim,
                   int min_count, int window_size, int ns_size,
                   int max_sentence_length, int epochs, int num_workers,
                   float unigram_pow, float sample, float init_lr, bool cbow,
                   bool shrink_window_size, long long unigram_table_size,
                   long log_freq)
    : train_file_path(train_file_path), out_vectors_path(out_vectors_path),
      emb_dim(emb_dim), window_size(window_size), ns_size(ns_size),
      max_sentence_length(max_sentence_length), epochs(epochs),
      num_workers(num_workers), unigram_pow(unigram_pow), sample(sample),
      init_lr(init_lr), cbow(cbow), shrink_window_size(shrink_window_size),
      precomputed_l2(false), unigram_table_size(unigram_table_size),
      log_freq(log_freq), unigram_table(new long long[unigram_table_size]),
      exp_table(new float[EXP_TABLE_SIZE])
{
    if (!in_vocab_path.compare(""))
    {
        this->vocab = Vocab(train_file_path, min_count, sample, true);
        this->vocab.save_vocab(out_vocab_path);
    }
    else
        this->vocab = Vocab::read_vocab(in_vocab_path, sample, true);
    if (!cbow && init_lr == 0.05)
        this->init_lr = 0.025;
    this->init_net();
    this->build_exp_table();
    this->build_unigram_table();
};

/* void read_word(ifstream &is, string &word)
{
    word.clear();
    word = "";
    if (!feof(fi))
        is >> word;
}; */

std::vector<float> Word2Vec::get_vector_syn0(string word)
{
    /* size_t word_idx = this->vocab.word2id(word) * this->emb_dim;
    std::vector<float> v(this->emb_dim);
    for (int i = 0; i < this->emb_dim; ++i)
        v[i] = this->syn0[word_idx + i];
    return v; */
    long word_idx = this->vocab.word2id(word);
    if (word_idx == -1)
        return vector<float>(0);
    float *p = &this->syn0[word_idx * this->emb_dim];
    vector<float> v{p, p + this->emb_dim};
    return v;
};

std::vector<float> Word2Vec::get_vector_syn1(string word)
{
    /* size_t word_idx = this->vocab.word2id(word) * this->emb_dim;
    std::vector<float> v(this->emb_dim);
    for (int i = 0; i < this->emb_dim; ++i)
        v[i] = this->syn1[word_idx + i];
    return v; */
    long word_idx = this->vocab.word2id(word);
    if (word_idx == -1)
        return vector<float>(0);
    float *p = &this->syn1[word_idx * this->emb_dim];
    vector<float> v{p, p + this->emb_dim};
    return v;
};

vector<vector<float>> Word2Vec::get_syn0()
{
    vector<vector<float>> v(this->vocab.size());
    for (size_t i = 0; i < this->vocab.size(); ++i)
    {
        v[i] = vector<float>(this->emb_dim);
        for (int j = 0; j < this->emb_dim; ++j)
            v[i][j] = this->syn0[i * this->emb_dim + j];
    }
    return v;
};

vector<float> Word2Vec::get_syn0_flat()
{
    vector<float> v(this->vocab.size() * this->emb_dim);
    for (size_t i = 0; i < this->vocab.size(); ++i)
    {
        for (int j = 0; j < this->emb_dim; ++j)
            v[i * this->emb_dim + j] = this->syn0[i * this->emb_dim + j];
    }
    return v;
};

vector<vector<float>> Word2Vec::get_syn1()
{
    vector<vector<float>> v(this->vocab.size());
    for (size_t i = 0; i < this->vocab.size(); ++i)
    {
        v[i] = vector<float>(this->emb_dim);
        for (int j = 0; j < this->emb_dim; ++j)
            v[i][j] = this->syn1[i * this->emb_dim + j];
    }
    return v;
};

vector<float> Word2Vec::get_syn1_flat()
{
    vector<float> v(this->vocab.size() * this->emb_dim);
    for (size_t i = 0; i < this->vocab.size(); ++i)
    {
        for (int j = 0; j < this->emb_dim; ++j)
            v[i * this->emb_dim + j] = this->syn1[i * this->emb_dim + j];
    }
    return v;
};

const Vocab Word2Vec::get_vocab() const { return this->vocab; };

void read_word(FILE *fi, string &word)
{
    char ch;
    word.clear();

    while (!feof(fi))
    {
        ch = fgetc(fi);
        if (ch == 13)
            continue;
        if (isspace(ch))
        {
            if (word.length() > 0)
            {
                // Put the newline back before returning so that we find it next
                // time.
                if (ch == '\n')
                    ungetc(ch, fi);
                break;
            }
            if (ch == '\n')
            {
                word = "";
                return;
            }
            else
                continue;
        }
        word.push_back(ch);
    }
};

void Word2Vec::save_vectors()
{
    cout << "\nSaving vectors to " << this->out_vectors_path << '\n';
    auto t1 = std::chrono::high_resolution_clock::now();

    ofstream os(this->out_vectors_path, ofstream::out);
    os << this->vocab.size() << ' ' << this->emb_dim << '\n';

    for (size_t i = 0; i < this->vocab.size(); ++i)
    {
        os << this->vocab[i].word << ' ';
        for (int j = 0; j < this->emb_dim; ++j)
            os << this->syn0[i * this->emb_dim + j] << ' ';
        os << '\n';
    }
    os.close();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "Elpased: " << duration.count() << "s\n";
};

void Word2Vec::train()
{
    cout << "\nTraining\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto t1 = start;
    atomic<long long> global_wc(0);
    thread workers[this->num_workers];

    if (this->cbow)
        for (int i = 0; i < this->num_workers; ++i)
            workers[i] = thread(&Word2Vec::train_thread_cbow, this, i,
                                ref(global_wc), start);
    else
        for (int i = 0; i < this->num_workers; ++i)
            workers[i] = thread(&Word2Vec::train_thread_sg, this, i,
                                ref(global_wc), start);

    for (int i = 0; i < this->num_workers; ++i)
        if (workers[i].joinable())
            workers[i].join();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "\nElpased: " << duration.count() << "s\n";
};

void Word2Vec::get_most_similar_thread(long word_idx, unsigned long start,
                                       unsigned long end, float *cos)
{
    float dot = 0;
    for (unsigned long i = start; i < end; ++i)
    {
        dot = 0;
        for (int j = 0; j < this->emb_dim; ++j)
            dot += this->syn0[word_idx * this->emb_dim + j] *
                   this->syn0[i * this->emb_dim + j];
        cos[i] = dot / (this->norm[i] * this->norm[word_idx]);
    }
    pthread_exit(NULL);
};

vector<string> Word2Vec::get_most_similar(string word, int topn,
                                          int num_workers)
{
    cout << "\nGetting most similar words on " << num_workers << " threads\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    thread workers[num_workers];
    unsigned long chunk_size = this->vocab.size() / num_workers, start = 0,
                  end = chunk_size;
    unsigned int bonus = this->vocab.size() - chunk_size * num_workers;
    if (bonus > 0)
    {
        end = chunk_size + 1;
        --bonus;
    }

    if (!this->precomputed_l2)
        this->precompute_l2_norm(num_workers);

    long word_idx = this->vocab.word2id(word);
    vector<string> sim_words{};
    if (word_idx == -1)
        return sim_words;

    float *cos = new float[this->vocab.size()];
    for (int i = 0; i < num_workers; ++i)
    {
        // cout << "Thread " << i << " working from " << start << " to " << end
        //      << '\n';
        workers[i] = thread(&Word2Vec::get_most_similar_thread, this, word_idx,
                            start, end, ref(cos));
        start = end;
        if (bonus > 0)
        {
            end += chunk_size + 1;
            --bonus;
        }
        else
            end += chunk_size;
    }

    for (int i = 0; i < num_workers; ++i)
        if (workers[i].joinable())
            workers[i].join();

    size_t *sorted_idxs = Utils::argsort(cos, this->vocab.size());
    free(cos);

    sim_words.resize(topn);
    for (int i = 0; i < topn; ++i)
        sim_words[i] = this->vocab.id2word(sorted_idxs[i]);
    free(sorted_idxs);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "Elpased: " << duration.count() << "s\n";

    return sim_words;
};

void Word2Vec::precompute_l2_norm_thread(unsigned long start, unsigned long end)
{
    float dot = 0;
    for (unsigned long i = start; i < end; ++i)
    {
        dot = 0;
        for (int j = 0; j < this->emb_dim; ++j)
            dot += this->syn0[i * this->emb_dim + j] *
                   this->syn0[i * this->emb_dim + j];
        this->norm[i] = sqrt(dot);
    }
    pthread_exit(NULL);
};

void Word2Vec::precompute_l2_norm(int num_workers)
{
    // - Distance matrix is symmetric, to reduce space we can use a traingular
    //   matrix. Better to flat it down
    // - The number of distances to be computed is, if 'n' is the number of
    //   vectors, (n+1)*n/2
    // - The distance between vector 'i' and 'j' can be accessed by
    //   i*(i-1)/2 + j - 1

    cout << "\nPrecomputing L2 distance matrix on " << num_workers
         << " threads\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    thread workers[num_workers];
    unsigned long chunk_size = this->vocab.size() / num_workers, start = 0,
                  end = chunk_size;
    unsigned int bonus = this->vocab.size() - chunk_size * num_workers;
    if (bonus > 0)
    {
        end = chunk_size + 1;
        --bonus;
    }
    this->norm = new float[this->vocab.size()];

    for (int i = 0; i < num_workers; ++i)
    {
        // cout << "Thread " << i << " working from " << start << " to " << end
        //      << '\n';
        workers[i] =
            thread(&Word2Vec::precompute_l2_norm_thread, this, start, end);
        start = end;
        if (bonus > 0)
        {
            end += chunk_size + 1;
            --bonus;
        }
        else
            end += chunk_size;
    }

    for (int i = 0; i < num_workers; ++i)
        if (workers[i].joinable())
            workers[i].join();

    this->precomputed_l2 = true;

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "Elpased: " << duration.count() << "s\n";
};

void Word2Vec::train_thread_sg(int id, atomic<long long> &global_wc,
                               chrono::system_clock::time_point start)
{
    int i, j, k, shrink_value = 0, sentence_length = 0, sentence_position = 0,
                 epoch = this->epochs, label;
    long long word, context_word,
        wc = 0, prev_wc = 0, sentence[this->max_sentence_length + 1], file_size,
        target, train_words = this->vocab.get_train_words(), target_idx,
        context_idx;
    unsigned long long next_random = (long long)id;
    float f, g, init_lr = this->init_lr;
    string w;

    w.reserve(100);

    float *neu1e = new float[this->emb_dim];

    FILE *fi = fopen(this->train_file_path.c_str(), "rb");
    fseek(fi, 0, SEEK_END);
    file_size = ftell(fi);
    fseek(fi, file_size / (long long)this->num_workers * (long long)id,
          SEEK_SET);

    while (true)
    {
        if (wc - prev_wc > this->log_freq)
        {
            global_wc += wc - prev_wc;
            prev_wc = wc;

            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = now - start;
            printf("%cLr: %f  Progress: %.2f%%  Words/sec: %.2f  ", 13,
                   this->init_lr,
                   global_wc / (float)(this->epochs * train_words + 1) * 100,
                   global_wc / (elapsed.count() + 1));
            fflush(stdout);

            // Update lr
            this->init_lr =
                init_lr *
                (1 - global_wc / (float)(this->epochs * train_words + 1));
            if (this->init_lr < init_lr * 0.0001)
                this->init_lr = init_lr * 0.0001;
        }

        if (sentence_length == 0)
        {
            while (true)
            {
                read_word(fi, w);
                if (feof(fi) || w.empty())
                    break;
                else
                    word = this->vocab.word2id(w);

                if (word == -1)
                    continue;
                wc++;

                // Subsampling frequent words
                if (this->sample > 0)
                {
                    next_random =
                        next_random * (unsigned long long)25214903917 + 11;
                    if (this->vocab[word].keep_prob <
                        (next_random & 0xFFFF) / (float)65536)
                        continue;
                }

                sentence[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= this->max_sentence_length)
                    break;
            }
            sentence_position = 0;
        }

        if (feof(fi) || (wc > train_words / this->num_workers))
        {
            global_wc += wc - prev_wc;
            epoch--;
            if (epoch == 0)
                break;
            wc = 0;
            prev_wc = 0;
            sentence_length = 0;
            fseek(fi, file_size / (long long)this->num_workers * (long long)id,
                  SEEK_SET);
            continue;
        }

        word = sentence[sentence_position];

        next_random = next_random * (unsigned long long)25214903917 + 11;

        // 'shrink_value' becomes a random integer between 0 and 'window' - 1.
        if (this->shrink_window_size)
            shrink_value = next_random % this->window_size;

        // Actual train loop
        for (i = shrink_value; i < this->window_size * 2 + 1 - shrink_value;
             ++i)
            if (i != this->window_size)
            {
                // Convert the window offset 'i' into an index 'j' into the
                // sentence array.
                j = sentence_position - this->window_size + i;
                if (j < 0)
                    continue;
                else if (j >= sentence_length)
                    continue;
                context_word = sentence[j];
                context_idx = context_word * emb_dim;

                for (j = 0; j < this->emb_dim; ++j)
                    neu1e[j] = 0;

                if (this->ns_size > 0)
                    for (k = 0; k < this->ns_size + 1; ++k)
                    {
                        if (k == 0)
                        {
                            target = word;
                            label = 1;
                        }
                        else
                        {
                            // Pick a random word
                            next_random =
                                next_random * (unsigned long long)25214903917 +
                                11;
                            target =
                                this->unigram_table[(next_random >> 16) %
                                                    this->unigram_table_size];
                            if (target == word)
                                continue;
                            label = 0;
                        }
                        target_idx = target * emb_dim;

                        // Calculate the dot-product between the input words
                        // weights (in syn0) and the output word's weights
                        // (in syn1)
                        f = 0;
                        for (j = 0; j < this->emb_dim; ++j)
                            f += this->syn0[context_idx + j] *
                                 this->syn1[target_idx + j];

                        // Compute sigmoid(f) through exp_table
                        // Compute error at the output, stored in 'g'
                        if (f > MAX_EXP)
                            g = (label - 1) * this->init_lr;
                        else if (f < -MAX_EXP)
                            g = (label - 0) * this->init_lr;
                        else
                            g = (label -
                                 this->exp_table[(
                                     int)((f + MAX_EXP) *
                                          (EXP_TABLE_SIZE / MAX_EXP / 2))]) *
                                this->init_lr;

                        // Accumulate gradients for the ouput layer over the
                        // negative samples and the positive one
                        for (j = 0; j < this->emb_dim; ++j)
                            neu1e[j] += g * this->syn1[target_idx + j];

                        // Update the output layer weights
                        for (j = 0; j < this->emb_dim; ++j)
                            this->syn1[target_idx + j] =
                                this->syn1[target_idx + j] +
                                g * this->syn0[context_idx + j];
                    }

                // Update the hidden layer weights
                for (j = 0; j < this->emb_dim; ++j)
                    this->syn0[context_idx + j] =
                        this->syn0[context_idx + j] + neu1e[j];
            }

        sentence_position++;
        if (sentence_position >= sentence_length)
        {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1e);
    pthread_exit(NULL);
}

void Word2Vec::train_thread_cbow(int id, atomic<long long> &global_wc,
                                 chrono::system_clock::time_point start)
{
    int i, j, k, shrink_value = 0, sentence_length = 0, sentence_position = 0,
                 epoch = this->epochs, label, cw;
    long long word, context_word,
        wc = 0, prev_wc = 0, sentence[this->max_sentence_length + 1], file_size,
        target, train_words = this->vocab.get_train_words(), target_idx,
        context_idx;
    unsigned long long next_random = (long long)id;
    float f, g, init_lr = this->init_lr;
    string w;

    w.reserve(100);

    float *neu1 = new float[this->emb_dim];
    float *neu1e = new float[this->emb_dim];

    FILE *fi = fopen(this->train_file_path.c_str(), "rb");
    fseek(fi, 0, SEEK_END);
    file_size = ftell(fi);
    fseek(fi, file_size / (long long)this->num_workers * (long long)id,
          SEEK_SET);

    while (true)
    {
        if (wc - prev_wc > this->log_freq)
        {
            global_wc += wc - prev_wc;
            prev_wc = wc;

            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = now - start;
            printf("%cLr: %f  Progress: %.2f%%  Words/sec: %.2f  ", 13,
                   this->init_lr,
                   global_wc / (float)(this->epochs * train_words + 1) * 100,
                   global_wc / (elapsed.count() + 1));
            fflush(stdout);

            // Update lr
            this->init_lr =
                init_lr *
                (1 - global_wc / (float)(this->epochs * train_words + 1));
            if (this->init_lr < init_lr * 0.0001)
                this->init_lr = init_lr * 0.0001;
        }

        if (sentence_length == 0)
        {
            while (true)
            {
                read_word(fi, w);
                if (feof(fi) || w.empty())
                    break;
                else
                    word = this->vocab.word2id(w);
                if (word == -1)
                    continue;
                wc++;

                // Subsampling frequent words
                if (this->sample > 0)
                {
                    next_random =
                        next_random * (unsigned long long)25214903917 + 11;
                    if (this->vocab[word].keep_prob <
                        (next_random & 0xFFFF) / (float)65536)
                        continue;
                }

                sentence[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= this->max_sentence_length)
                    break;
            }
            sentence_position = 0;
        }

        if (feof(fi) || (wc > train_words / this->num_workers))
        {
            global_wc += wc - prev_wc;
            epoch--;
            if (epoch == 0)
                break;
            wc = 0;
            prev_wc = 0;
            sentence_length = 0;
            fseek(fi, file_size / (long long)this->num_workers * (long long)id,
                  SEEK_SET);
            continue;
        }

        word = sentence[sentence_position];

        cw = 0;
        for (j = 0; j < this->emb_dim; ++j)
            neu1[j] = 0;
        for (j = 0; j < this->emb_dim; ++j)
            neu1e[j] = 0;

        next_random = next_random * (unsigned long long)25214903917 + 11;

        // 'shrink_value' becomes a random integer between 0 and 'window' - 1.
        if (this->shrink_window_size)
            shrink_value = next_random % this->window_size;

        // Actual train loop
        for (i = shrink_value; i < this->window_size * 2 + 1 - shrink_value;
             ++i)
            if (i != this->window_size)
            {
                // Convert the window offset 'i' into an index 'j' into the
                // sentence array.
                j = sentence_position - this->window_size + i;

                if (j < 0)
                    continue;
                else if (j >= sentence_length)
                    continue;

                context_word = sentence[j];
                context_idx = context_word * emb_dim;

                for (j = 0; j < this->emb_dim; ++j)
                    neu1[j] += this->syn0[context_idx + j];

                cw++;
            }

        if (cw)
        {
            // average of the context word vectors
            for (j = 0; j < this->emb_dim; ++j)
                neu1[j] /= cw;

            if (this->ns_size > 0)
                for (k = 0; k < this->ns_size + 1; ++k)
                {
                    if (k == 0)
                    {
                        target = word;
                        label = 1;
                    }
                    else
                    {
                        // Pick a random word to use as a 'negative sample';
                        next_random =
                            next_random * (unsigned long long)25214903917 + 11;
                        target = this->unigram_table[(next_random >> 16) %
                                                     this->unigram_table_size];
                        if (target == word)
                            continue;
                        label = 0;
                    }
                    target_idx = target * emb_dim;

                    // Dot product between average of the context word vectors
                    // (neu1) and target word vector (syn1[target])
                    f = 0;
                    for (j = 0; j < this->emb_dim; ++j)
                        f += neu1[j] * this->syn1[target_idx + j];

                    // Compute sigmoid(f) through exp_table
                    // Compute error at the output, stored in 'g'
                    if (f > MAX_EXP)
                        g = (label - 1) * this->init_lr;
                    else if (f < -MAX_EXP)
                        g = (label - 0) * this->init_lr;
                    else
                        g = (label - this->exp_table[(int)((f + MAX_EXP) *
                                                           (EXP_TABLE_SIZE /
                                                            MAX_EXP / 2))]) *
                            this->init_lr;

                    // Multiply the error by the output layer weights.
                    // Accumulate these gradients over all of the negative
                    // samples.
                    for (j = 0; j < this->emb_dim; ++j)
                        neu1e[j] += g * this->syn1[target_idx + j];

                    // Update the output layer weights
                    for (j = 0; j < this->emb_dim; ++j)
                        this->syn1[target_idx + j] =
                            this->syn1[target_idx + j] + g * neu1[j];
                }

            for (i = shrink_value; i < this->window_size * 2 + 1 - shrink_value;
                 ++i)
                if (i != this->window_size)
                {
                    // Convert the window offset 'i' into an index 'j' into
                    // the sentence array.
                    j = sentence_position - this->window_size + i;

                    if (j < 0)
                        continue;
                    else if (j >= sentence_length)
                        continue;

                    context_word = sentence[j];
                    context_idx = context_word * emb_dim;

                    // Update context vectors
                    for (j = 0; j < this->emb_dim; ++j)
                        this->syn0[context_idx + j] =
                            this->syn0[context_idx + j] + neu1e[j];
                }
        }
        sentence_position++;
        if (sentence_position >= sentence_length)
        {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void Word2Vec::init_net()
{
    cout << "\nInitializing neural net\n";
    auto t1 = std::chrono::high_resolution_clock::now();

    unsigned long long next_random = 1;
    this->syn0 = new float[this->vocab.size() * this->emb_dim];
    this->syn1 = new float[this->vocab.size() * this->emb_dim];

    for (size_t i = 0; i < this->vocab.size(); ++i)
    {
        for (int j = 0; j < this->emb_dim; ++j)
        {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            // Random uniform in the range [-0.5; 0.5] / emb_dim
            this->syn0[i * this->emb_dim + j] =
                (((next_random & 0xFFFF) / (float)65536) - 0.5) / this->emb_dim;
        }

        for (int j = 0; j < this->emb_dim; ++j)
            this->syn1[i * this->emb_dim + j] = 0;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "Elpased: " << duration.count() << "s\n";
};

void Word2Vec::build_exp_table()
{
    cout << "\nBuilding precomputed sigmoid table\n";
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < EXP_TABLE_SIZE; ++i)
    {

        // Compute e^x in the range -6.0 to +6.0.
        this->exp_table[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) *
                                 MAX_EXP); // Precompute the exp() table

        // exp(x) / (exp(x) + 1) = 1 / (1 + exp(-x))
        this->exp_table[i] /= (this->exp_table[i] + 1);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "Elpased: " << duration.count() << "s\n";
};

void Word2Vec::build_unigram_table()
{
    cout << "\nBuilding precomputed unigram table\n";
    auto t1 = std::chrono::high_resolution_clock::now();

    int i;
    double train_words_pow = 0, d1;
    long vocab_size = vocab.size();

    for (i = 0; i < vocab_size; ++i)
        train_words_pow += pow(this->vocab[i].count, this->unigram_pow);

    // 'i' is the vocabulary index of the current word. Word 'i' will appear
    // multiple times in the table, based on its frequency in the training data
    i = 0;

    // Calculate the probability that we choose word 'i'
    d1 = pow(vocab[i].count, this->unigram_pow) / train_words_pow;

    for (int j = 0; j < this->unigram_table_size; ++j)
    {
        this->unigram_table[j] = i;

        // If the fraction of the table we have filled is greater than the
        // probability of choosing this word, move to the next word.
        if (j / (double)this->unigram_table_size > d1)
        {
            ++i;

            // Calculate the probability for the new word, and accumulate it
            // with the probabilities of all previous words, so that we can
            // compare d1 to the percentage of the table that we have filled.
            d1 += pow(vocab[i].count, this->unigram_pow) / train_words_pow;
        }
        if (i >= vocab_size)
            i = vocab_size - 1;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "Elpased: " << duration.count() << "s\n";
};
} // namespace w2v