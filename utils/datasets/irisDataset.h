#ifndef IRISDATASET_H
#define IRISDATASET_H

#include <vector>
#include <algorithm>  // std::shuffle
#include <random>     // std::mt19937, std::random_device

#include <utils/datasets/Dataset.h>

class IrisDataset final: public Dataset {
public:
    explicit IrisDataset(const std::string& filepath) : Dataset(IrisDataset::loadData(filepath)) {
        minMaxNormalization(this->data);
    }

     DatasetFormat loadData(std::string filepath) override{
        std::string line;
        std::vector<std::string> tokenizedRow;
        DatasetFormat data;
        std::vector<double> properties;
        std::ifstream inputFile(filepath);

        getline(inputFile, line);

        while(getline(inputFile, line)) {
            size_t start = 0;
            for (size_t i = 0; i < line.length(); i++) {
                if(line.at(i) == ',') {
                    tokenizedRow.push_back(line.substr(start, i-start));
                    start = i + 1;
                }else if(i == line.length()-1) {
                    tokenizedRow.push_back(line.substr(start, i+1-start));
                }
            }

            for(size_t j = 1; j < tokenizedRow.size()-1; j++) {
                auto value = std::stod(tokenizedRow.at(j));
                properties.push_back(value);
            }

            double irisClass;
            const std::string className = tokenizedRow.at(tokenizedRow.size()-1);

            if(className == "Iris-setosa") {
                irisClass = 0;
            }else if(className == "Iris-versicolor") {
                irisClass = 1;
            }else {
                irisClass = 2;
            }

            data.emplace_back(properties, irisClass);
            properties.clear();
            tokenizedRow.clear();
        }

        return data;
    }

    int getNumClasses() override{
        return 3;
    }

    static std::vector<std::string> getClassNames() {
        return {"Setosa", "Versicolor", "Virginica"};
    }
};

#endif //IRISDATASET_H
