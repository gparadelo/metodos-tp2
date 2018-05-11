//
// Created by mike on 07/05/18.
//

#include "Model.h"
#include "inutils.h"

void Model::evaluate(const char *testDatasetName) {
    ifstream input(testDatasetName);

    Dataset testSet;
    loadDataset(testDatasetName, &testSet);

    vector<int> people;

    if(mode == SIMPLEKNN){
        assert(_k <= images.size());
        for (int i = 0; i < testSet.size(); ++i) {
            people.push_back(kNearestNeighbors(testSet[i].first));
            cout << people[i] << " ";
        }
        cout << endl;
    }
    else {

    }
}

Model::Model(MODE mode) : mode(mode) {

}

void Model::train(const char *trainDatasetName) {
    loadDataset(trainDatasetName, &images);

    if (mode == SIMPLEKNN) return;

    if (mode == PCAWITHKNN) {

    }
}

void Model::setAlpha(int alpha) {
    _alpha = alpha;
}

void Model::setK(int k) {
    _k = k;
}

void Model::loadDataset(const char *trainDatasetName, Dataset * dest) {
    ifstream input(trainDatasetName);

    if (!input.good()) {
        cout << "The input file isn't good";
        assert(false);
    }

    //Aca habria que hacer no solo te guardes la imagen,
    while (input.good()) {
        uchar *data = NULL;
        int width = 0, height = 0;
        PPM_LOADER_PIXEL_TYPE pt = PPM_LOADER_PIXEL_TYPE_INVALID;
        char filename[256] = {0};
        input.getline(filename, 256, ',');

        bool ret = LoadPPMFile(&data, &width, &height, &pt, filename);

        if (!ret || width == 0 || height == 0 || pt != PPM_LOADER_PIXEL_TYPE_GRAY_8B) {
            throw std::runtime_error("test_load failed");
        }

        _width = width;//hace esto para cada imagen, si bien todas tiene el mismo tamaño ¿se puede evitar?
        _height = height;

        int intClass = -1;
        if (input.peek() != '/n') {
            char stringClass[2] = {0};
            input.ignore(1, ' ');
            input.getline(stringClass, 2, ',');
            intClass = stoi(stringClass);
        }
        pair<uchar*, int> trainingInstance = make_pair(data, intClass);
        dest->push_back(trainingInstance);

        input.ignore(1, '/n');
    }
}

void Model::outputResults(const char *outputFileName) {

}

int  Model::kNearestNeighbors(uchar* newImage) {
    vector< pair <int, int> > distances;

    for (int i = 0; i < images.size(); ++i) {
        pair<int, int> dist (getSquaredNorm(images[i].first, newImage, (_width*_height)), images[i].second);
        distances.push_back(dist);
    }

    vector<int> neighbors;
    sort(distances.begin(), distances.end(), pairCompare);

    for (int i = 0; i < _k; ++i) {
        neighbors.push_back(distances[i].second);
    }

    int nearest = neighbors[0];
    int maxCount = count(neighbors.begin(), neighbors.end(), neighbors[0]);
    for (int i = 1; i < neighbors.size(); ++i) {
        int newCount = count(neighbors.begin(), neighbors.end(), neighbors[i]);
        if (newCount > maxCount) {
            maxCount = newCount;
            nearest = neighbors[i];
        }
    }

    return nearest;
}

int getSquaredNorm(uchar* &v1, uchar* &v2, int size) { //Qué onda con el const *uchar?
    int distance = 0;

    for (int i = 0; i <  size; ++i) {
        distance += pow((int)v1[i] - (int)v2[i], 2);
    }
    return distance;
}

bool pairCompare(pair<int, int> i, pair<int, int> j) {
    return (i.first < j.first);
}

vector<vector<double>> Model::calculateCovarianceMatrix(Dataset X) {
    vector<vector<double>> normalizedX(_height, vector<double>(_width, 0));
    vector<double> norm(_width, 0);

    //Calculo la norma de todas las imagenes por cada pixel
    for (int j = 0; j < _width; ++j) {
        double sum = 0;
        for (int i = 0; i < _height; ++i) {
            sum += (double)X[i].first[j];
        }
        norm[j] = sum / sqrt((double)_height - 1);
    }


    //Calculo cada imagen normalizada
    for (int i = 0; i < _height; ++i) {
        for (int j = 0; j < _width*_height; ++j) {
            double normalizedPixel = (((double)X[i].first[j]) - norm[j]) / sqrt((double) _height);
            normalizedX[i][j] = normalizedPixel;
        }
    }

    vector<vector<double>> normalizedXt(_height, vector<double>(_width, 0));

    for (size_t i = 0; i < normalizedX.size(); ++i)
        for (size_t j = 0; j < normalizedX[0].size(); ++j)
            normalizedXt[j][i] = normalizedX[i][j];

    vector<vector<double>> covarianceMatrix(_height, vector<double>(_width, 0));
    for (int i = 0; i < _height; ++i) {
        for (int j = 0; j < _width; ++j) {
            for (int k = 0; k < _height; ++k) {
                covarianceMatrix[i][j] += normalizedXt[i][k] * normalizedX[k][j];
            }
        }
    }

    return covarianceMatrix;
}