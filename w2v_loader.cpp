#include "w2v_loader.h"
#include "utils.h"
#include "vocab.h"
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

namespace w2v
{
Word2VecLoader::Word2VecLoader(string vectors_path, string vocab_path)
    : precomputed_l2(false)
{
    this->vocab = Vocab::read_vocab(vocab_path, 0.003, true);
    this->read_vectors(vectors_path);
};

void Word2VecLoader::read_vectors(string vectors_path)
{
    cout << "\nReading vectors from " << vectors_path << '\n';
    auto t1 = std::chrono::high_resolution_clock::now();

    unsigned long i = 0;
    float f;
    string s;
    ifstream is(vectors_path, ifstream::in);

    is >> s;
    this->num_vectors = atoi(s.c_str());
    is >> s;
    this->emb_dim = atoi(s.c_str());
    cout << "Vectors: " << this->num_vectors << '\n';
    cout << "Embedding dim: " << this->emb_dim << '\n';

    this->syn0 = new float[this->num_vectors * this->emb_dim];

    while (i < this->num_vectors && !is.eof())
    {
        is >> s; // Read word
        for (int j = 0; j < this->emb_dim; ++j)
        {
            is >> s; // Read float
            f = atof(s.c_str());
            this->syn0[i * this->emb_dim + j] = f;
        }
        ++i;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "Elpased: " << duration.count() << "s\n";
};

std::vector<float> Word2VecLoader::get_vector(string word)
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

vector<vector<float>> Word2VecLoader::get_vectors()
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

vector<float> Word2VecLoader::get_vectors_flat()
{
    vector<float> v(this->vocab.size() * this->emb_dim);
    for (size_t i = 0; i < this->vocab.size(); ++i)
    {
        for (int j = 0; j < this->emb_dim; ++j)
            v[i * this->emb_dim + j] = this->syn0[i * this->emb_dim + j];
    }
    return v;
};

const Vocab Word2VecLoader::get_vocab() const { return this->vocab; };

void Word2VecLoader::get_most_similar_thread(long word_idx, unsigned long start,
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

vector<pair<string, float>>
Word2VecLoader::get_most_similar(string word, int topn, int num_workers)
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
    vector<pair<string, float>> sim_words{};
    if (word_idx == -1)
        return sim_words;

    float *cos = new float[this->vocab.size()];
    for (int i = 0; i < num_workers; ++i)
    {
        // cout << "Thread " << i << " working from " << start << " to " << end
        //      << '\n';
        workers[i] = thread(&Word2VecLoader::get_most_similar_thread, this,
                            word_idx, start, end, ref(cos));
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

    sim_words.resize(topn + 1);
    for (int i = 0; i < topn + 1; ++i)
        sim_words[i] = pair<string, float>(this->vocab.id2word(sorted_idxs[i]),
                                           cos[sorted_idxs[i]]);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "Elpased: " << duration.count() << "s\n";

    return sim_words;
};

void Word2VecLoader::precompute_l2_norm_thread(unsigned long start,
                                               unsigned long end)
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

void Word2VecLoader::precompute_l2_norm(int num_workers)
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
        workers[i] = thread(&Word2VecLoader::precompute_l2_norm_thread, this,
                            start, end);
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
} // namespace w2v