#ifndef VOCAB_H
#define VOCAB_H

#include <string>
#include <unordered_map>
#include <vector>

namespace vocab
{
class Vocab
{
    typedef struct Word Word;

public:
    Vocab();
    Vocab(std::string train_file_path);
    Vocab(std::string train_file_path, int min_count, float sample,
          bool verbose);
    Vocab(std::string train_file_path, int min_count, float sample,
          bool verbose, size_t words, size_t train_words,
          std::vector<Word> &vocab,
          std::unordered_map<std::string, size_t> &w2id);
    Word &operator[](size_t i);
    void build_vocab();
    std::string id2word(size_t id);
    size_t word2id(std::string word);
    std::unordered_map<std::string, size_t> &get_word2id();
    const std::unordered_map<std::string, size_t> get_word2id() const;
    std::vector<Word> &get_vocab();
    const std::vector<Word> get_vocab() const;
    void save_vocab(std::string vocab_path);
    void print_vocab();
    size_t size();
    size_t get_train_words();
    float get_keep_prob(std::string word);

private:
    std::string train_file_path;
    int min_count;
    float sample;
    bool verbose;
    size_t words = 0, train_words = 0;
    std::vector<Word> vocab;
    std::unordered_map<std::string, size_t> w2id;
};

struct Word
{
    Word();
    Word(std::string word, size_t count);
    std::string word;
    size_t count;
    float keep_prob;
};

Vocab read_vocab(std::string vocab_path, float sample);
Vocab read_vocab(std::string vocab_path, float sample, bool verbose);

} // namespace vocab
#endif