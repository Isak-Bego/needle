#ifndef XORDATASET_H
#define XORDATASET_H
#include <vector>
#include <utils/datasets/Dataset.h>

class XORDataset final: public Dataset {
public:

    XORDataset() : Dataset(XORDataset::loadData("")){}

    DatasetFormat loadData(std::string filepath) override{
        DatasetFormat data;

        data.push_back({{0.0, 0.0}, 0.0});  // 0 XOR 0 = 0
        data.push_back({{0.0, 1.0}, 1.0});  // 0 XOR 1 = 1
        data.push_back({{1.0, 0.0}, 1.0});  // 1 XOR 0 = 1
        data.push_back({{1.0, 1.0}, 0.0});  // 1 XOR 1 = 0

        return data;
    }

    int getNumClasses() override{
        return 2;
    }
};

#endif //XORDATASET_H
