//
// Created by mike on 07/05/18.
//

#include "Model.h"
#include "utils.h"

void Model::evaluate(const char *testDatasetName) {
    ifstream input(testDatasetName);

    if(mode == SIMPLEKNN){
        //ACA HABRIA QUE LLAMAR AL CODIGO DE GONZA
        //las imagenes de training estan guardadas en la propiedad dataset

        //Habria que hacer algo como foreach image in test set, run:
        //kNearestNeighbors();
    }
}

Model::Model(MODE mode) : mode(mode) {

}

void Model::train(const char *trainDatasetName) {
    loadDataset(trainDatasetName);

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

void Model::loadDataset(const char *trainDatasetName) {
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
        char filename[256];
        input.getline(filename, 256, ',');
        bool ret = LoadPPMFile(&data, &width, &height, &pt, filename);

        if (!ret || width == 0 || height == 0 || pt != PPM_LOADER_PIXEL_TYPE_RGB_8B) {
            throw std::runtime_error("test_load failed");
        }

        char stringClass[8];
        input.getline(stringClass, 8, ',');
        int intClass = stoi(stringClass);

        pair<uchar*, int> trainingInstance = make_pair(data, intClass);
        images.push_back(trainingInstance);
    }
}

void Model::outputResults(const char *outputFileName) {

}
