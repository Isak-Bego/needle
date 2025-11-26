#ifndef DATASET_H
#define DATASET_H
#include "nnComponents/trainers/trainer.h"

/**
 * The following file should serve as an interface for the dataset objects, so that every dataset loader function
 * operates under the same API.
 */
class Dataset {
protected:
    DatasetFormat data;
    double min;
    double max;

public:
    virtual ~Dataset() = default;

    explicit Dataset(const DatasetFormat& data) {
        this->data = data;
    }
    virtual DatasetFormat loadData(std::string filepath) = 0;
    virtual int getNumClasses() = 0;


    DatasetFormat getData() {
        return data;
    }

    int getNumFeatures() {
        return static_cast<int>(this->data.at(0).first.size());
    };

    void minMaxNormalization(DatasetFormat &dataset) {
        // We are going to apply min-max normalization (rescaling) in the interval [0, 1]
        // Step 1: Find the min and max elements
        double min = dataset.at(0).first[0];
        double max = dataset.at(0).first[0];

        for (const auto& sample: dataset) {
            for(auto i: sample.first) {
                min = std::min(min, i);
                max = std::max(max, i);
            }
        }

        this->min = min;
        this->max = max;

        for (auto& sample: dataset) {
            for(auto& i: sample.first) {
                i = (i - min) / (max - min);
            }
        }
    }
};


#endif //DATASET_H
