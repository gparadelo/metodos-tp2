#include <iostream>
#include "inutils.h"
#include "Model.h"
#include <fstream>
#include <chrono>
#include <algorithm>


using namespace std;
using namespace std::chrono;



//Mover a utils?
char *getCmdOption(char **begin, char **end, const string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char **begin, char **end, const string &option) {
    return std::find(begin, end, option) != end;
}
//Gracias stackoverflow


int main(int argc, char *argv[]) {

    if (!cmdOptionExists(argv, argv + argc, "-m")) {
        cout << "necesito: -m <method>";
        return 0;
    }

    if (!cmdOptionExists(argv, argv + argc, "-i")) {
        cout << "necesito: -i <train_set>";
        return 0;
    }

    if (!cmdOptionExists(argv, argv + argc, "-q")) {
        cout << "necesito: -q <train_set>";
        return 0;
    }


    char *method = getCmdOption(argv,argv + argc, "-m");

    char *trainSetName = getCmdOption(argv, argv + argc, "-i");

    char *testSetName = getCmdOption(argv, argv + argc, "-q");

    //Creamos una instancia de nuestra clase model y la instanciamos en modo SIMPLEKNN
    //Puede ser eso o PCAWITHKNN
    MODE mod;
    if (stoi(method) == 1) { mod = PCAWITHKNN; }
    else              { mod = SIMPLEKNN; }

    Model pcaWithKnn(mod);

    pcaWithKnn.setK(10);
    pcaWithKnn.setAlpha(2);

//    Le pasamos la direccion al dataset de training

    pcaWithKnn.train(trainSetName);

//  SavePPMFile("../test.ppm", pcaWithKnn.images[0].first, 92, 112,PPM_LOADER_PIXEL_TYPE_GRAY_8B, " ");

//    Evaluamos los tests y el modelo se guarda adentro los resultados
    pcaWithKnn.evaluate(testSetName);


//    Le pasamos el archivo donde guardarlos
//  pcaWithKnn.outputResults("path/to/results.csv");


    return 0;

    //ESTO ES CODIGO DEL TP PASADO QUE ME AYUDA A SABER COMO CARGAR ARCHIVOS PARA LEER Y ESCRIBIR
//    ifstream input(argv[1]);
//    ofstream resultsFile;
//    ofstream timeFile;
//    resultsFile.open("../experimentacion/results/results.out");
//    timeFile.open("../experimentacion/results/time",std::ios_base::app);
}

