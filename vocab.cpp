#include "vocab.h"
#include <algorithm> // std::sort
#include <chrono>
#include <fstream>  // std::ifstream
#include <iostream> // std::cout
#include <math.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

#define BUFFER_LENGTH 1 << 20

namespace vocab
{

Word::Word() : word(""), count(0){};

Word::Word(std::string word, size_t count) : word(word), count(count){};

Vocab::Vocab(){};

Vocab::Vocab(string train_file_path) : Vocab(train_file_path, 5, 0.001, true){};

Vocab::Vocab(string train_file_path, int min_count, float sample, bool verbose)
    : train_file_path(train_file_path), min_count(min_count), sample(sample),
      verbose(verbose)
{
    this->vocab.reserve(100000);
    this->w2id.reserve(100000);
    this->build_vocab();
};

Vocab::Vocab(std::string train_file_path, int min_count, float sample,
             bool verbose, unsigned long long words,
             unsigned long long train_words, std::vector<Word> &vocab,
             std::unordered_map<std::string, unsigned long> &w2id)
    : train_file_path(train_file_path), min_count(min_count), sample(sample),
      verbose(verbose), words(words), train_words(train_words)
{
    this->vocab = move(vocab);
    this->w2id = move(w2id);
};

unsigned long long Vocab::get_train_words() { return this->train_words; }

Word &Vocab::operator[](size_t i) { return this->vocab[i]; };

void Vocab::build_vocab()
{
    cout << "Building vocab\n";

    FILE *fp = fopen(this->train_file_path.c_str(), "rb");
    if (ferror(fp) == 0)
    {
        long init, end;
        char buffer[BUFFER_LENGTH];
        unordered_map<string, size_t> freqs{};
        string word;

        word.reserve(100);
        freqs.reserve(100000);

        auto t1 = std::chrono::high_resolution_clock::now();

        while (!feof(fp))
        {
            init = ftell(fp);
            fread(buffer, sizeof(char), BUFFER_LENGTH, fp);
            end = ftell(fp);

            for (int i = 0; i < end - init; i++)
            {
                if (isspace(buffer[i]))
                {
                    ++freqs[word];
                    ++this->words;
                    word.clear();
                }
                else
                    word.push_back(buffer[i]);
            }
        }
        fclose(fp);

        this->vocab.reserve(freqs.size());
        for (const auto &pair : freqs)
        {
            if (pair.second >= (unsigned)this->min_count)
            {
                this->vocab.push_back(Word(pair.first, pair.second));
                this->train_words += pair.second;
            }
        }
        this->vocab.shrink_to_fit();
        sort(this->vocab.begin(), this->vocab.end(),
             [](const Word &a, const Word &b) { return b.count < a.count; });

        this->w2id = unordered_map<string, size_t>(this->vocab.size());
        for (size_t i = 0; i < this->vocab.size(); i++)
        {
            this->w2id[vocab[i].word] = i;
            this->vocab[i].keep_prob =
                (sqrt(this->vocab[i].count / (this->sample * train_words)) +
                 1) *
                (this->sample * train_words) / this->vocab[i].count;
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = t2 - t1;

        if (verbose)
        {
            cout << "Words: " << words << '\n';
            cout << "Unique words: " << freqs.size() << '\n';
            cout << "Train words: " << train_words << ", keeping "
                 << (float)train_words / (float)words * 100.0
                 << "% of train words" << '\n';
            cout << "Unique words after min count: " << vocab.size() << '\n';
            cout << "Elapsed: " << duration.count() << "s\n";
        }
    }
    /* ifstream is(this->train_file_path, ifstream::binary);
    if (is)
    {
        int length = 1 << 20;
        long init, end;
        char *buffer = new char[BUFFER_LENGTH];
        unordered_map<string, long> freqs{};
        string word;

        word.reserve(100);
        freqs.reserve(100000);

        auto t1 = std::chrono::high_resolution_clock::now();

        while (!is.eof())
        {
            init = is.tellg();
            is.read(buffer, BUFFER_LENGTH);
            end = is.tellg();
            for (int i = 0; i < end - init; i++)
            {
                if (isspace(buffer[i]))
                {
                    ++freqs[word];
                    ++this->words;
                    word.clear();
                }
                else
                    word.push_back(buffer[i]);
            }
        }

        for (const auto &pair : freqs)
        {
            if (pair.second >= min_count)
            {
                this->vocab.push_back(Word(pair.first, pair.second));
                this->train_words += pair.second;
            }
        }

        sort(this->vocab.begin(), this->vocab.end(),
             [](const Word &a, const Word &b) { return b.count < a.count; });

        for (long i = 0; i < this->vocab.size(); i++)
            this->w2id[vocab[i].word] = i;

        this->vocab.shrink_to_fit();

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = t2 - t1;

        if (verbose)
        {
            cout << "Words: " << words << '\n';
            cout << "Unique words: " << freqs.size() << '\n';
            cout << "Train words: " << train_words << ", keeping "
                 << (float)train_words / (float)words * 100.0
                 << "% of train words" << '\n';
            cout << "Unique words after min count: " << vocab.size() << '\n';
            cout << "Elapsed: " << duration.count() << '\n';
        }
    } */
    else
        throw runtime_error("Error reading file");
}

size_t Vocab::size() { return this->vocab.size(); }

string Vocab::id2word(size_t id)
{
    return id < this->vocab.size() ? this->vocab[id].word : "";
}

long Vocab::word2id(string word)
{
    auto it = this->w2id.find(word);
    if (it != this->w2id.end())
        return it->second;
    return -1;
}

void Vocab::save_vocab(string vocab_path)
{
    cout << "Saving vocab to " << vocab_path << '\n';
    auto t1 = std::chrono::high_resolution_clock::now();

    ofstream os(vocab_path, ofstream::out);
    char *buffer = new char[BUFFER_LENGTH];
    os.rdbuf()->pubsetbuf(buffer, sizeof(buffer));
    for (const Word &w : this->vocab)
        os << w.word << " " << w.count << '\n';

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = t2 - t1;
    cout << "Elpased: " << duration.count() << "s\n";
}

float Vocab::get_keep_prob(string word)
{
    return this->vocab[this->w2id[word]].keep_prob;
};

void Vocab::print_vocab()
{
    cout << "Word Count" << '\n';
    for (auto &w : this->vocab)
        cout << w.word << " " << w.count << '\n';
}

unordered_map<string, size_t> &Vocab::get_word2id() { return this->w2id; }

const unordered_map<string, size_t> Vocab::get_word2id() const
{
    return this->w2id;
}

vector<Word> &Vocab::get_vocab() { return this->vocab; }

const vector<Word> Vocab::get_vocab() const { return this->vocab; }

Vocab Vocab::read_vocab(string vocab_path, float sample)
{
    return read_vocab(vocab_path, sample, true);
}

Vocab Vocab::read_vocab(string vocab_path, float sample, bool verbose)
{
    FILE *fp = fopen(vocab_path.c_str(), "rb");
    if (ferror(fp) == 0)
    {
        cout << "Loading vocab from " << vocab_path << '\n';

        bool word_found = false;
        long long init, end;
        size_t count = 0, train_words = 0, idx_vocab = 0;
        char buffer[BUFFER_LENGTH];
        string word, number;
        vector<Word> vocab{};
        unordered_map<string, size_t> w2id{};

        word.reserve(100);
        number.reserve(50);
        w2id.reserve(100000);
        vocab.reserve(100000);

        auto t1 = std::chrono::high_resolution_clock::now();

        while (!feof(fp))
        {
            init = ftell(fp);
            fread(buffer, sizeof(char), BUFFER_LENGTH, fp);
            end = ftell(fp);
            for (long i = 0; i < end - init; i++)
            {
                if (buffer[i] == ' ')
                    word_found = true;
                else if (buffer[i] == '\n')
                {
                    count = stoi(number);
                    w2id[word] = idx_vocab;
                    vocab.push_back(Word(word, count));
                    train_words += count;
                    idx_vocab++;
                    word_found = false;
                    word.clear();
                    number.clear();
                }
                else if (word_found)
                    number.push_back(buffer[i]);
                else
                    word.push_back(buffer[i]);
            }
        }
        fclose(fp);

        vocab.shrink_to_fit();
        for (auto &w : vocab)
            w.keep_prob = (sqrt(w.count / (sample * train_words)) + 1) *
                          (sample * train_words) / w.count;

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = t2 - t1;

        if (verbose)
        {
            cout << "Train words: " << train_words << '\n';
            cout << "Unique words: " << vocab.size() << '\n';
            cout << "Elapsed: " << duration.count() << '\n';
        }

        return Vocab(vocab_path, vocab[vocab.size() - 1].count, sample, true,
                     train_words, train_words, vocab, w2id);
    }
    else
        throw runtime_error("Error reading file");
}

} // namespace vocab