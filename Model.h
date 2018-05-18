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

template <typename T>
using Dataset = vector<pair<T,int>>;

class Model {

public:

    Model(MODE mode);
    void setAlpha(int i);

    void setK(int i);

    void evaluate(const char * string);

    void train(const char * string);

    void outputResults(const char * string);

    matrix<double> calculateCovarianceMatrix(const Dataset<vector<double>> &X);

    //podria querer cambiar MODE desde acá

    //podriamos tener observador de _images

private:
    MODE mode;
    int _alpha;
    unsigned int _k;
    unsigned int _width;
    unsigned int _height;
    void loadDataset(const char *trainDatasetName, Dataset<uchar*>* dest);


    template <typename T, typename X>
    int  kNearestNeighbors(Dataset<X> datasetToValidateAgainst, T newImage);

    template <typename T>
    void getTC(const Dataset<T>&);

    template < typename T>
    void normalizeDataset(Dataset<vector<double>>&, const Dataset<T>&);

    template <typename T>
    void setAveragePixelsVector(const Dataset<T> &src);

    template <typename T, typename X>
    void applyTCToDataset(Dataset<T>& dest, Dataset<X>& src);

    matrix<double> datasetToMatrix(const Dataset<vector<double>> &D);


    Dataset<uchar*> images;
    Dataset<vector<double>> reducedDataset;
    matrix<double> tc;
    vector<double> averagePixels;
    double standardDeviation;
};


template <typename T>
matrix<T> vectorOuterProduct(vector<T>);

template <typename T>
matrix<T> matrixScalarMultiply(const matrix<T> &, T);

template <typename T>
matrix<T> addMatrices(matrix<T>,matrix<T>);

template <typename T>
vector<T> normalizeVector(vector<T>);

template <typename T>
double getSquaredNorm(T &v1, T &v2, int size);

bool pairCompare(pair<int, int> i, pair<int, int> j);

matrix<double> matrixMultiply(matrix<double> &m1, matrix<double> &m2);

vector<double> matrixVectorMultiply(const matrix<double> &m1, const vector<double> &v1);

vector<double> vectorMatrixMultiply(vector<double> v1, matrix<double> m1);

double vectorVectorMultiply(vector<double> v1, vector<double> v2);

pair<vector<double>, double> powerMethod(const matrix<double> &mat);

template <typename T>
matrix<T> transposeAndMultiplyWithItself(const matrix<T> & A);

#endif //CARALIBRO_DATASET_H
