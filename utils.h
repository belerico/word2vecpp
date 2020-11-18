#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <numeric>

namespace utils
{
class Utils
{
public:
    static size_t *argsort(const float *v, size_t size);
};
} // namespace utils
#endif