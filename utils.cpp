#include "utils.h"
#include <algorithm>
#include <numeric>

using namespace std;

namespace utils
{
size_t *Utils::argsort(const float *v, size_t size)
{

    // initialize original index locations
    size_t *idx = new size_t[size];
    std::iota(idx, idx + size, 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    std::stable_sort(idx, idx + size,
                     [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

    return idx;
}
} // namespace utils