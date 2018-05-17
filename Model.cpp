//
// Created by mike on 07/05/18.
//

#include "Model.h"
#include "inutils.h"

Model::Model(MODE mode) : mode(mode) {

}

void Model::setAlpha(int alpha) {
    _alpha = alpha;
}

void Model::setK(int k) {
    _k = k;
}

void Model::evaluate(const char *testDatasetName) {
    ifstream input(testDatasetName);

    Dataset<uchar *> testSet;
    loadDataset(testDatasetName, &testSet);

    vector<int> people;

    if (mode == SIMPLEKNN) {
        assert(_k <= images.size());
        for (int i = 0; i < testSet.size(); ++i) {
            people.push_back(kNearestNeighbors(testSet[i].first));
            cout << people[i] << " ";
        }
        cout << endl;
    } else {

    }
}

void Model::train(const char *trainDatasetName) {
    loadDataset(trainDatasetName, &images);

    if (mode == PCAWITHKNN) {
        normalizeDataset();
        getPCADataset();
    }

}

void Model::normalizeDataset() {
    int numberOfPixels = _width * _height;

    vector<double> avg(numberOfPixels, 0);

    //Calculo la norma de todas las imagenes por cada pixel
    for (int j = 0; j < numberOfPixels; ++j) {
        int sum = 0;
        for (int i = 0; i < images.size(); ++i) {
            sum += images[i].first[j];
        }
        avg[j] = (double) sum / (double) images.size();
    }

    averagePixels = avg;
    standardDeviation = sqrt((double) images.size() - 1);

    //Calculo cada imagen normalizada
    for (int i = 0; i < images.size(); ++i) {
        vector<double> currentNormalizedImage;
        for (int j = 0; j < numberOfPixels; ++j) {
            double normalizedPixel = (((double) images[i].first[j]) - avg[j]) / standardDeviation;
            currentNormalizedImage.push_back(normalizedPixel);
        }
        pair<vector<double>, int> normalizedTrainingInstance = make_pair(currentNormalizedImage, images[i].second);
        normalizedDataset.push_back(normalizedTrainingInstance);
    }


}

void Model::getPCADataset() {
    getTC();
    applyTCToDataset();
}


void Model::loadDataset(const char *trainDatasetName, Dataset<uchar *> *dest) {
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
        pair<uchar *, int> trainingInstance = make_pair(data, intClass);
        dest->push_back(trainingInstance);

        input.ignore(1, '/n');
    }
}


void Model::outputResults(const char *outputFileName) {

}

int Model::kNearestNeighbors(uchar *newImage) {
    vector<pair<int, int> > distances;

    for (int i = 0; i < images.size(); ++i) {
        pair<int, int> dist(getSquaredNorm(images[i].first, newImage, (_width * _height)), images[i].second);
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

matrix<double> Model::datasetToMatrix(Dataset<vector<double>> &D) {

    matrix<double> ret(D.size(), vector<double>(D[0].first.size(), 0));

    for (int i = 0; i < D.size(); ++i) {
        for (int j = 0; j < D[0].first.size(); ++j) {
            ret[i][j] = D[i].first[j];
        }
    }

    return ret;
}

int getSquaredNorm(uchar *&v1, uchar *&v2, int size) { //Qué onda con el const *uchar?
    int distance = 0;

    for (int i = 0; i < size; ++i) {
        distance += pow((int) v1[i] - (int) v2[i], 2);
    }
    return distance;
}


bool pairCompare(pair<int, int> i, pair<int, int> j) {
    return (i.first < j.first);
}

matrix<double> Model::calculateCovarianceMatrix(Dataset<vector<double>> &X) {
    int numberOfPixels = _height * _width;

    matrix<double> normalizedX = datasetToMatrix(X);


    matrix<double> covarianceMatrix = transposeAndMultiplyWithItself(normalizedX);

    return covarianceMatrix;
}

template<typename T>
matrix<T> transposeAndMultiplyWithItself(matrix<T> &A) {
    matrix<double> ret(A[0].size(), vector<double>(A[0].size(), 0));

    for (int i = 0; i < A[0].size(); ++i) {
        for (int j = 0; j < A[0].size(); ++j) {
            ret[i][j] = 0;
            for (int k = 0; k < A.size(); ++k) {
                //CHEQUEAR ESTO A VER SI ESTA BIEN
                ret[i][j] += A[k][i] * A[k][i];
            }
        }
    }
    return ret;
}

void Model::getTC() {
    //Get TC tiene que aplicar el método de la potencia para conseguir los alpha autovectores
    //arma V
    int numberOfPixels = _height * _width;

    vector<pair<vector<double>, double>> eigenVectorsAndValues;

    matrix<double> currentMatrix = calculateCovarianceMatrix(normalizedDataset);

    for (int i = 0; i < _alpha; ++i) {
        pair<vector<double>, double> currentEigenVectorsAndValues = powerMethod(currentMatrix);

        eigenVectorsAndValues.push_back(currentEigenVectorsAndValues);


        //DEFLACION:
        matrix<double> vvt = vectorOuterProduct(currentEigenVectorsAndValues.first);

        vvt = matrixScalarMultiply(vvt, currentEigenVectorsAndValues.second);

        vvt = matrixScalarMultiply(vvt, (double) -1);

        currentMatrix = addMatrices(currentMatrix, vvt);
    }


    //Armamos V, la transformacion carácteristica tc
    matrix<double> a(numberOfPixels, vector<double>(_alpha, 0));

    tc = a;
    for (int j = 0; j < _alpha; ++j) {
        for (int i = 0; i < numberOfPixels; ++i) {
            tc[i][j] = eigenVectorsAndValues[j].first[i];
        }
    }
}

matrix<double> vectorMatrixMultiply(vector<double> v1, matrix<double> m1) {
    return matrix<double>();
}


void Model::applyTCToDataset() {
    int numberOfPixels = _height * _width;
    //Se cuenta con tc de tamaño numberOfPixels x alpha
    //hay que hacer XV
    //X tiene que ser normalizedDataset

    for (int i = 0; i < normalizedDataset.size(); ++i) {

        vector<double> currentVector(normalizedDataset.size(),0);

        double currentValue = 0;


        for (int j = 0; j < _alpha; ++j) {

            for (int k = 0; k < numberOfPixels; ++k) {

                currentValue += normalizedDataset[i].first[k] * tc[k][j];
            }
            currentVector[i] = (currentValue);
        }
        pair<vector<double>,int> reducedTrainingInstance = make_pair(currentVector, normalizedDataset[i].second);
        reducedDataset.push_back(reducedTrainingInstance);
    }

}

pair<vector<double>, double> powerMethod(matrix<double> &mat) {
    vector<double> v(mat[0].size(), 1);

    v = normalizeVector(v);

    assert(mat.size() == mat[0].size());

    int niter = 10;
    for (int i = 0; i < niter; ++i) {
        v = matrixVectorMultiply(mat, v);
        v = normalizeVector(v);
    }

    double lambda = vectorVectorMultiply(v, matrixVectorMultiply(mat, v))
                    / vectorVectorMultiply(v, v);

    pair<vector<double>, double> ret = make_pair(v, lambda);

    return ret;
}

template<typename T>
vector<T> normalizeVector(vector<T> v) {
    double sum = 0;
    for (int i = 0; i < v.size(); ++i) {
        sum += v[i];
    }
    for (int j = 0; j < v.size(); ++j) {
        v[j] = v[j] / sum;
    }
    return v;
}

matrix<double> matrixMultiply(matrix<double> &m1, matrix<double> &m2) {
    assert(m1[0].size() == m2.size());
    matrix<double> ret(m1.size(), vector<double>(m2[0].size(), 0));

    for (int i = 0; i < m1.size(); ++i) {
        for (int j = 0; j < m2[0].size(); ++j) {
            for (int k = 0; k < m2.size(); ++k) {
                ret[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    return ret;
}

vector<double> matrixVectorMultiply(matrix<double> &m1, vector<double> &v1) {
    assert(m1[0].size() == v1.size());

    vector<double> ret(m1.size(), 0);

    for (int i = 0; i < m1.size(); ++i) {
        for (int k = 0; k < v1.size(); ++k) {
            ret[i] += m1[i][k] * v1[k];
        }
    }

    return ret;

}

double vectorVectorMultiply(vector<double> v1, vector<double> v2) {


}


template<typename T>
matrix<T> vectorOuterProduct(vector<T> v) {
    matrix<T> ret(v.size(), vector<T>(v.size(), 0));
    for (int i = 0; i < v.size(); ++i) {
        for (int j = 0; j < v.size(); ++j) {
            ret[i][j] = v[i] * v[j];
        }
    }
    return ret;
}

template<typename T>
matrix<T> matrixScalarMultiply(matrix<T> m, T s) {
    matrix<T> ret(m.size(), vector<T>(m[0].size(), 0));

    for (int i = 0; i < m.size(); ++i) {
        for (int j = 0; j < m[0].size(); ++j) {
            ret[i][j] = m[i][j] * s;
        }
    }

    return ret;

}

template<typename T>
matrix<T> addMatrices(matrix<T> A, matrix<T> B) {

    assert(A.size() == B.size());
    assert(A[0].size() == B[0].size());

    matrix<T> ret(A.size(), vector<T>(A[0].size(), 0));

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size(); ++j) {
            ret[i][j] = A[i][j] + B[i][j];
        }
    }

    return ret;
}