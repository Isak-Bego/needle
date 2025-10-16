#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

namespace helper {
    template<typename T>
    int find(std::vector<T> &vec, T val) {
        for (std::size_t i = 0; i < vec.size(); i++) {
            if (vec[i] == val) {
                return i;
            }
        }
        return -1;
    }
}

#endif //HELPERFUNCTIONS_H
