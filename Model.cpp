//
// Created by mike on 07/05/18.
//

#include "Model.h"
#include "utils.h"

Model::Model(MODE mode) : mode(mode), measuringMetrics(false), measuringTimes(false) {

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

    assert(_k <= images.size());


    if (mode == PCAWITHKNN) {
        Dataset<vector<double>> normalizedTestSet;
        normalizeDataset(normalizedTestSet, testSet);

        Dataset<vector<double>> reducedTestSet;
        applyTCToDataset(reducedTestSet, normalizedTestSet);

        assert(reducedTestSet.size() == normalizedTestSet.size());
        assert(reducedTestSet[0].first.size() == reducedDataset[0].first.size());

        startEvaluate = high_resolution_clock::now();
        for (int i = 0; i < reducedTestSet.size(); ++i) {
            rawPredictions.push_back(kNearestNeighbors(reducedDataset, reducedTestSet[i].first));
//            cout << rawPredictions[i] << " ";
        }
        endEvaluate = high_resolution_clock::now();

//        cout << endl;


    } else {
        startEvaluate = high_resolution_clock::now();
        for (int i = 0; i < testSet.size(); ++i) {
            rawPredictions.push_back(kNearestNeighbors(images, testSet[i].first));
//            cout << rawPredictions[i] << " ";
        }
        endEvaluate = high_resolution_clock::now();
//        cout << endl;
    }

    analyzePredictions(testSet);
}


void Model::train(const char *trainDatasetName) {
    loadDataset(trainDatasetName, &images);

    startTraining = high_resolution_clock::now();
    if (mode == PCAWITHKNN) {
        //Params: Dataset destination, Dataset src
        setAveragePixelsVector(images);

        Dataset<vector<double>> normalizedDataset;
        normalizeDataset(normalizedDataset, images);

        getTC(normalizedDataset);
        //Signature meaning: destination, source
        applyTCToDataset(reducedDataset, normalizedDataset);

        //matrix<double> covariance = calculateCovarianceMatrix(reducedDataset);
        //Tendria que dar diagonal?
    }
    endTraining = high_resolution_clock::now();


}

template<typename X>
void Model::setAveragePixelsVector(const Dataset<X> &src) {
    int numberOfPixels = _width * _height;
    vector<double> avg(numberOfPixels, 0);

    //Calculo la norma de todas las imagenes por cada pixel
    for (int j = 0; j < numberOfPixels; ++j) {
        int sum = 0;
        for (int i = 0; i < src.size(); ++i) {
            sum += src[i].first[j];
        }
        avg[j] = (double) sum / (double) src.size();
    }

    averagePixels = avg;
    standardDeviation = sqrt((double) src.size() - 1);
}

template<typename X>
void Model::normalizeDataset(Dataset<vector<double>> &dst, const Dataset<X> &src) {
    int numberOfPixels = _width * _height;

    //Calculo cada imagen normalizada
    for (int i = 0; i < src.size(); ++i) {
        vector<double> currentNormalizedImage;
        for (int j = 0; j < numberOfPixels; ++j) {
            double normalizedPixel = (((double) src[i].first[j]) - averagePixels[j]) / standardDeviation;
            currentNormalizedImage.push_back(normalizedPixel);
        }
        pair<vector<double>, int> normalizedTrainingInstance = make_pair(currentNormalizedImage, (int) src[i].second);
        dst.push_back(normalizedTrainingInstance);
    }
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
            input >> stringClass;
//            cout << stringClass;
            intClass = stoi(stringClass);
        }
        pair<uchar *, int> trainingInstance = make_pair(data, intClass);
        dest->push_back(trainingInstance);

        input.ignore(1, '/n');
    }

//    cout << "Dataset size: " << dest->size() << endl;
}


template<typename T, typename X>
int Model::kNearestNeighbors(const Dataset<X> &datasetToValidateAgainst, T newImage) {
    vector<pair<double, int> > distances;

    int size;
    if (mode == SIMPLEKNN) {
        size = (_width * _height);
    } else {
        size = _alpha;
    }

    for (int i = 0; i < datasetToValidateAgainst.size(); ++i) {
        double squaredNorm = getSquaredNorm(datasetToValidateAgainst[i].first, newImage, size);
        pair<double, int> dist = make_pair(squaredNorm, (int) datasetToValidateAgainst[i].second);
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

matrix<double> Model::datasetToMatrix(const Dataset<vector<double>> &D) {

    matrix<double> ret(D.size(), vector<double>(D[0].first.size(), 0));

    for (int i = 0; i < D.size(); ++i) {
        for (int j = 0; j < D[0].first.size(); ++j) {
            ret[i][j] = D[i].first[j];
        }
    }

    return ret;
}

template<typename T>
double getSquaredNorm(const T &v1, const T &v2, int size) {
    double distance = 0;

    for (int i = 0; i < size; ++i) {
        distance += pow((double) v1[i] - (double) v2[i], 2);
    }
    return sqrt(distance);
}


bool pairCompare(pair<int, int> i, pair<int, int> j) {
    return (i.first < j.first);
}

matrix<double> Model::calculateCovarianceMatrix(const matrix<double> &X) {

//    matrix<double> normalizedX = datasetToMatrix(X);

//    printMatrix(normalizedX);

    //DEVUELVE XX^t
    matrix<double> covarianceMatrix = transposeAndMultiplyWithItself(X);

    return covarianceMatrix;
}


template<typename T>
matrix<T> transposeAndMultiplyWithItself(const matrix<T> &A) {
    matrix<double> ret(A.size(), vector<double>(A.size(), 0));



    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A.size(); ++j) {
            for (int k = 0; k < A[0].size(); ++k) {
                //CHEQUEAR ESTO A VER SI ESTA BIEN
                ret[i][j] += A[i][k] * A[j][k];
            }
        }
    }
    return ret;
}

template<typename T>
void Model::getTC(const Dataset<T> &src) {
    //Get TC tiene que aplicar el método de la potencia para conseguir los alpha autovectores
    //arma V
    int numberOfPixels = _height * _width;

    vector<pair<vector<double>, double>> eigenVectorsAndValues;


    matrix<double> X = datasetToMatrix(src);
    matrix<double> currentMatrix = calculateCovarianceMatrix(X);
    matrix<double> Xt = traspose(X);


    for (int i = 0; i < _alpha; ++i) {
        cout << "Calculating eigenvector: " << i + 1 << "/" << _alpha << endl;
        pair<vector<double>, double> currentEigenVectorsAndValues = powerMethod(currentMatrix);


        //DEFLACION:
        matrix<double> vvt = vectorOuterProduct(currentEigenVectorsAndValues.first);

        vvt = matrixScalarMultiply(vvt, currentEigenVectorsAndValues.second);

        vvt = matrixScalarMultiply(vvt, (double) -1);

        currentMatrix = addMatrices(currentMatrix, vvt);


        //Tenemos los autovectores de la M que era de dimensiones pequeñas
        //Para conseguir los autovectores de la M grande, multiplicamos a izquierda por X^t

        currentEigenVectorsAndValues.first = matrixVectorMultiply(Xt,currentEigenVectorsAndValues.first);

        eigenVectorsAndValues.push_back(currentEigenVectorsAndValues);

        cout << "Root of the eigenvalue: " << sqrt(currentEigenVectorsAndValues.second)
             << endl;
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

template<typename T, typename X>
void Model::applyTCToDataset(Dataset<T> &dst, Dataset<X> &src) {
    assert(dst.size() == 0);
    int numberOfPixels = _height * _width;
    //Se cuenta con tc de tamaño numberOfPixels x alpha
    //hay que hacer XV
    //X tiene que ser normalizedDataset

    for (int i = 0; i < src.size(); ++i) {

        vector<double> currentVector;

        for (int j = 0; j < _alpha; ++j) {
            double currentValue = 0;
            for (int k = 0; k < numberOfPixels; ++k) {
                currentValue += src[i].first[k] * tc[k][j];
            }
            currentVector.push_back(currentValue);
        }
        pair<vector<double>, int> reducedTrainingInstance = make_pair(currentVector, (int) src[i].second);
        dst.push_back(reducedTrainingInstance);
    }

}

template<typename T>

void Model::printMatrix(T A) {
    cout << "[";
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size(); ++j) {
            cout << A[i][j] << " ";
        }
        if (i < A.size() - 1) {
            cout << ";";
        }
    }
    cout << "]" << endl;

}

void Model::setOutputFile(ofstream *oFile) {
    outputFile = oFile;
}

void Model::setTimesFile(ofstream *tFile) {
    timesFile = tFile;
    measuringTimes = true;
}

void Model::setMetricsFile(ofstream *mFile) {
    metricsFile = mFile;
    measuringMetrics = true;
}

pair<vector<double>, double> powerMethod(const matrix<double> &mat) {
    vector<double> v(mat[0].size(), 1);

    v = normalizeVector(v);

    assert(mat.size() == mat[0].size());

    int niter = 0;
    while (true) {
        vector<double> newV = matrixVectorMultiply(mat, v);
        newV = normalizeVector(newV);

        if (getSquaredNorm(newV, v, v.size()) < 10e-7 || niter > 250) {
            break;
        }
        v = newV;
        niter++;
    }

    double vBv = (vectorVectorMultiply(v, matrixVectorMultiply(mat, v)));
    double vtv = vectorVectorMultiply(v, v);
//    cout << "vBv: "<< vBv << "          vtv: " << vtv << endl;
    double lambda = vBv / vtv;

    pair<vector<double>, double> ret = make_pair(v, lambda);

    return ret;
}

template<typename T>
vector<T> normalizeVector(vector<T> &v) {
    double sum = 0;
    for (int i = 0; i < v.size(); ++i) {
        sum += pow(v[i], 2);
    }
    for (int j = 0; j < v.size(); ++j) {
        v[j] = v[j] / sqrt(sum);
    }
    return v;
}

vector<double> matrixVectorMultiply(const matrix<double> &m1, const vector<double> &v1) {
    assert(m1[0].size() == v1.size());

    vector<double> ret(m1.size(), 0);

    for (int i = 0; i < m1.size(); ++i) {
        for (int k = 0; k < v1.size(); ++k) {
            ret[i] += m1[i][k] * v1[k];
        }
    }

    return ret;

}


double vectorVectorMultiply(const vector<double> &v1, const vector<double> &v2) {
    assert(v1.size() == v2.size());
    double sum = 0;
    for (int i = 0; i < v1.size(); ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

template<typename T>
matrix<T> vectorOuterProduct(const vector<T> &v) {
    matrix<T> ret(v.size(), vector<T>(v.size(), 0));
    for (int i = 0; i < v.size(); ++i) {
        for (int j = 0; j < v.size(); ++j) {
            ret[i][j] = v[i] * v[j];
        }
    }
    return ret;
}


template<typename T>
matrix<T> matrixScalarMultiply(const matrix<T> &m, T s) {
    matrix<T> ret(m.size(), vector<T>(m[0].size(), 0));

    for (int i = 0; i < m.size(); ++i) {
        for (int j = 0; j < m[0].size(); ++j) {
            ret[i][j] = m[i][j] * s;
        }
    }

    return ret;

}

template<typename T>
matrix<T> addMatrices(const matrix<T> &A, const matrix<T> &B) {

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

template<typename T>
void Model::analyzePredictions(Dataset<T> testSet) {
    //Need to get true/false positives/negatives

    map<int, metric> classMetrics;

    //Init
    for (int i = 0; i < images.size(); ++i) {
        bool write = true;
        if (classMetrics.find(images[i].second) != classMetrics.end()) {
            bool write = false;
        }
        if (write) {
            metric p;
            p.realClass = images[i].second;
            p.fn = 0;
            p.fp = 0;
            p.tn = 0;
            p.tp = 0;
            p.accurracy = 0;
            p.precision = 0;
            p.recall = 0;
            p.f1 = 0;
            classMetrics.insert(pair<int, metric>(images[i].second, p));
        };
    }


    for (int j = 0; j < rawPredictions.size(); ++j) {
        int real = testSet[j].second;
        int predicted = rawPredictions[j];
        assert(real != 0);
        assert(predicted != 0);

//        cout << "predicted: " << predicted << " should have been: " << real << endl;
        if (real == predicted) {
            assert(classMetrics[real].realClass == real);
            classMetrics[real].tp += 1;
            for (int i = 0; i < classMetrics.size(); ++i) {
                if (i + 1 != real) {
                    classMetrics[i + 1].tn += 1;
                }
            }
        }
        if (real != predicted) {
            assert(classMetrics[real].realClass == real);
            classMetrics[real].fn += 1;
            classMetrics[predicted].fp += 1;
            for (int i = 0; i < classMetrics.size(); ++i) {
                if ((i + 1) != real && (i + 1) != predicted) {
                    classMetrics[i + 1].tn += 1;
                }
            }
        }
    }

    for (int k = 0; k < classMetrics.size(); ++k) {
        int i = k + 1;

        metric currentMetric = classMetrics[i];
        currentMetric.accurracy = (double) (currentMetric.tp + currentMetric.tn) /
                                  (double) (currentMetric.tp + currentMetric.tn + currentMetric.fn + currentMetric.fp);


        currentMetric.precision = (currentMetric.tp + currentMetric.fp) == 0 ? 1 : (double) (currentMetric.tp) /
                                                                                   (double) (currentMetric.tp +
                                                                                             currentMetric.fp);

        currentMetric.recall = (currentMetric.tp + currentMetric.fn) == 0 ? 1 : (double) (currentMetric.tp) /
                                                                                (double) (currentMetric.tp +
                                                                                          currentMetric.fn);

        currentMetric.f1 = (currentMetric.precision + currentMetric.recall) == 0 ? 1 : (
                2 * (currentMetric.precision * currentMetric.recall) /
                (currentMetric.precision + currentMetric.recall));

        classMetrics[i] = currentMetric;


//        cout << currentMetric.precision << ";" << currentMetric.recall << ";" << currentMetric.f1 << endl;
    }


    averageAccurracy = 0;
    averageRecall = 0;
    averagePrecision = 0;
    averageF1 = 0;

    for (int l = 0; l < classMetrics.size(); ++l) {
        int i = l + 1;
        averageAccurracy += classMetrics[i].accurracy;
        averagePrecision += classMetrics[i].precision;
        averageRecall += classMetrics[i].recall;
        averageF1 += classMetrics[i].f1;
    }

    averageAccurracy /= classMetrics.size();
    averagePrecision /= classMetrics.size();
    averageRecall /= classMetrics.size();
    averageF1 /= classMetrics.size();


}

void Model::outputResults() {
    if (measuringMetrics) {
        *metricsFile << _k << ";" << _alpha << ";" << averageAccurracy << ";" << averagePrecision << ";"
                     << averageRecall << ";" << averageF1 << endl;
    }

    cout << "***************Resultados***************" << endl<< "Accuracy promedio: "<< averageAccurracy << endl << "Precision promedio: " << averagePrecision << endl <<"Recall promedio: " << averageRecall << endl << "F1 promedio: " << averageF1 << endl << "****************************************" << endl;

    cout<< "Se utilizó:" << endl << "k: " << _k << endl << "alpha: " << _alpha << endl;


    if (measuringTimes) {


        duration<double> elapsedTraining = duration_cast<duration<double>>(endTraining - startTraining);
        duration<double> elapsedEvaluate = duration_cast<duration<double>>(endEvaluate - startEvaluate);

        *timesFile << mode << ";" << _k << ";" << _alpha << ";" << elapsedTraining.count() << ";"
                   << elapsedEvaluate.count() << endl;
    }

//
    for (int i = 0; i < rawPredictions.size(); ++i) {
        *outputFile << rawPredictions[i] << "," << endl;
    }


}

matrix<double> Model::traspose(matrix<double> X) {
    matrix<double> Xt(X[0].size(), vector<double>(X.size(), 0));
    for (int i = 0; i < X.size(); ++i) {
        for (int j = 0; j < X[0].size(); ++j) {
            Xt[j][i] = X[i][j];
        }
    }
    return Xt;
}


