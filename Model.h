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
#include <utility>
#include <algorithm>
#include <cmath>

using namespace std;

typedef enum {
    SIMPLEKNN = 0,
    PCAWITHKNN = 1
} MODE;

template <typename T>
using matrix = vector<vector<T>>;

typedef vector<pair<uchar*,int>> Dataset;

class Model {

public:

    Model(MODE mode);
    void setAlpha(int i);

    void setK(int i);

    void evaluate(const char * string);

    void train(const char string[]);

    void outputResults(const char * string);

    matrix<double> calculateCovarianceMatrix(Dataset X);

    //podria querer cambiar MODE desde acá

    //podriamos tener observador de _images
private:
    MODE mode;
    int _alpha;
    unsigned int _k;
    unsigned int _width;
    unsigned int _height;

    void loadDataset(const char *trainDatasetName, Dataset* dest);

    int  kNearestNeighbors(uchar* newImage);

    Dataset images;

};

int getSquaredNorm(uchar* &v1, uchar* &v2, int size);

bool pairCompare(pair<int, int> i, pair<int, int> j);

matrix<double> matrixMultiply(matrix<double> m1, matrix<double> m2);

vector<double> matrixVectorMultiply(matrix<double> m1, vector<double> v1);

matrix<double> vectorMatrixMultiply(vector<double> v1, matrix<double> m1);

double vectorVectorMultiply(vector<double> v1, vector<double> v2);

#endif //CARALIBRO_DATASET_H
