#ifndef W2V_LOADER_H
#define W2V_LOADER_H

#include "vocab.h"
#include <string>
#include <vector>

namespace w2v
{
class Word2VecLoader
{
public:
    Word2VecLoader(std::string vectors_path, std::string vocab_path);
    std::vector<float> get_vector(std::string word);
    std::vector<std::vector<float>> get_vectors();
    std::vector<float> get_vectors_flat();
    const vocab::Vocab get_vocab() const;
    void precompute_l2_norm(int num_workers = 6);
    std::vector<std::pair<std::string, float>>
    get_most_similar(std::string word, int topn, int num_workers = 6);

private:
    vocab::Vocab vocab;
    bool precomputed_l2;
    int emb_dim;
    unsigned long num_vectors;
    float *syn0, *norm;

    void read_vectors(std::string vectors_path);
    void precompute_l2_norm_thread(unsigned long start, unsigned long end);
    void get_most_similar_thread(long word_idx, unsigned long start,
                                 unsigned long end, float *sim);
};
} // namespace w2v
#endif