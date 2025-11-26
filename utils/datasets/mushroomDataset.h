#ifndef MUSHROOMDATASETLOADER_H
#define MUSHROOMDATASETLOADER_H
#include <vector>
#include <fstream>

class MushroomDataset {
public:
    static std::vector<std::pair<std::vector<double>, double> > get_data() {
        std::string line;
        std::vector<std::string> tokenizedRow;
        std::vector<std::pair<std::vector<double>, double>> data;
        std::vector<double> properties;
        std::ifstream inputFile("mushrooms.csv");

        getline(inputFile, line);

        while(getline(inputFile, line)) {
            size_t start = 0;
            for (size_t i = 0; i < line.length(); i++) {
                if(line.at(i) == ',') {
                    tokenizedRow.push_back(line.substr(start, i-start));
                    start = i + 1;
                }else if(i == line.length()-1) {
                    tokenizedRow.push_back(line.substr(start, 1));
                }
            }

            for(size_t j = 1; j < tokenizedRow.size(); j++) {
                auto value = tokenizedRow.at(j).c_str()[0] - 'a' >= 0 ? tokenizedRow.at(j).c_str()[0] - 'a' : 0;
                properties.push_back(value);
            }

            auto mushroomClass = tokenizedRow.at(0) == "e" ? 1 : 0;
            data.emplace_back(properties, mushroomClass);
            properties.clear();
            tokenizedRow.clear();
        }

        return data;
    }
};

#endif //MUSHROOMDATASETLOADER_H
