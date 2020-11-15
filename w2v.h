#ifndef W2V_H
#define W2V_H

#include "vocab.h"
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
    Word2Vec(std::string train_file_path,
             std::string out_vectors_path = "./vectors.txt",
             std::string out_vocab_path = "./vocab.txt",
             std::string in_vocab_path = "", int emb_dim = 100,
             int min_count = 5, int window_size = 5, int ns_size = 5,
             int max_sentence_length = 1e3, int epochs = 5, int num_workers = 7,
             float unigram_pow = 0.75, float sample = 1e-3,
             float init_lr = 0.05, bool cbow = 1, bool shrink_window_size = 1,
             long long unigram_table_size = 1e8, long log_freq = 1e4);
    void train();
    void init_net();
    void build_exp_table();
    void build_unigram_table();
    void save_vectors();
    std::vector<float> get_vector_syn0(std::string word);
    std::vector<float> get_vector_syn1(std::string word);
    std::vector<std::vector<float>> get_syn0();
    std::vector<std::vector<float>> get_syn1();
    std::vector<float> get_syn0_flat();
    std::vector<float> get_syn1_flat();
    const vocab::Vocab get_vocab() const;
    void precompute_l2_norm(int num_workers);
    std::vector<std::string> get_most_similar(std::string word, int topn,
                                              int num_workers = 6);

private:
    vocab::Vocab vocab;
    std::string train_file_path, out_vectors_path;
    int emb_dim, window_size, ns_size, max_sentence_length, epochs, num_workers;
    float unigram_pow, sample, init_lr;
    bool cbow, shrink_window_size, precomputed_l2;
    long long unigram_table_size;
    long log_freq;
    long long *unigram_table;
    float *exp_table;
    float *syn0, *syn1, *norm;

    void train_thread_sg(int id, std::atomic<long long> &word_count_actual,
                         std::chrono::system_clock::time_point start);
    void train_thread_cbow(int id, std::atomic<long long> &word_count_actual,
                           std::chrono::system_clock::time_point start);
    void precompute_l2_norm_thread(unsigned long start, unsigned long end);
    void get_most_similar_thread(long word_idx, unsigned long start,
                                 unsigned long end, std::vector<float> &sim);
};
} // namespace w2v
#endif