//
// Created by mike on 07/05/18.
//

#ifndef CARALIBRO_DATASET_H
#define CARALIBRO_DATASET_H

#include <fstream>
#include <vector>
#include "ppmloader/ppmloader.h"
#include <iostream>
#include <cassert>

using namespace std;

typedef enum {
    SIMPLEKNN = 0,
    PCAWITHKNN =1
} MODE;

typedef vector<pair<uchar*,int>> Dataset;

class Model {

public:

    Model(MODE mode);
    void setAlpha(int i);

    void setK(int i);

    void evaluate(const char * string);

    void train(const char string[]);

    void outputResults(const char * string);

private:
    MODE mode;
    int _alpha;
    int _k;

    void loadDataset(const char *trainDatasetName);

    Dataset images;
};


#endif //CARALIBRO_DATASET_H
