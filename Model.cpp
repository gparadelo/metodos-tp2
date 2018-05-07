//
// Created by mike on 07/05/18.
//

#include "Model.h"
#include "utils.h"

void Model::evaluate(const char *testDataseName) {
    ifstream input(testDataseName);

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
    trainDatasetName = trainDatasetName;

    loadDataset();


    if (mode == SIMPLEKNN) return;

    if (mode == PCAWITHKNN) {

    }
}

void Model::setAlpha(int alpha) {
    alpha = alpha;
}

void Model::setK(int i) {
    k = k;
}

void Model::loadDataset() {
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
        std::string filename = "buda.0.ppm";
        bool ret = LoadPPMFile(&data, &width, &height, &pt, filename.c_str());

        if (!ret || width == 0 || height == 0 || pt != PPM_LOADER_PIXEL_TYPE_RGB_8B) {
            throw std::runtime_error("test_load failed");
        }


        //ESTO HAY QUE CAMBIARLO POR CODIGO QUE TE DIGA LA CLASE O PERSONA A LA QUE PERTENECE LA IMAGEN
        int clase = 1;


        pair<uchar*, int> trainingInstance = make_pair(data, clase);
        images.push_back(trainingInstance);
    }


}

void Model::outputResults(const char *outputFileName) {

}
