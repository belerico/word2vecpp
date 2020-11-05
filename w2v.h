#ifndef W2V_H
#define W2V_H

#include "vocab.h"
#include <array>
#include <atomic>
#include <chrono>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace w2v
{
class Word2Vec
{
public:
    Word2Vec(std::string train_file_path, std::string out_vectors_path,
        std::string out_vocab_path, std::string in_vocab_path, int emb_dim, int min_count, int window_size,
        int ns_size, int max_sentence_length, int epochs, int num_workers,
        float unigram_pow, float sample, float init_lr, bool cbow,
        bool shrink_window_size, long long unigram_table_size, long log_freq);
    void train();
    void init_net();
    void build_exp_table();
    void build_unigram_table();
    void save_vectors();

private:
    vocab::Vocab vocab;
    std::string train_file_path, out_vectors_path;
    int emb_dim, window_size, ns_size, max_sentence_length, epochs, num_workers;
    float unigram_pow, sample, init_lr;
    bool cbow, shrink_window_size;
    long long unigram_table_size;
    long log_freq;
    long long *unigram_table;
    float *exp_table;
    float *syn0, *syn1;

    void train_thread_sg(int id, std::atomic<long long> &word_count_actual,
                         std::chrono::system_clock::time_point start);
    void train_thread_cbow(int id, std::atomic<long long> &word_count_actual,
                           std::chrono::system_clock::time_point start);
};
} // namespace w2v
#endif