#include <iostream>
#include "utils.h"
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

    if (!cmdOptionExists(argv, argv + argc, "-o")) {
        cout << "necesito: -o <outputPreditionsFileName>";
        return 0;
    }


    char *method = getCmdOption(argv,argv + argc, "-m");

    char *trainSetName = getCmdOption(argv, argv + argc, "-i");

    char *testSetName = getCmdOption(argv, argv + argc, "-q");

    char *outputPreditionsFileName = getCmdOption(argv, argv + argc, "-o");

    ofstream outputFile;

    outputFile.open(outputPreditionsFileName);


    //Creamos una instancia de nuestra clase model y la instanciamos en modo SIMPLEKNN
    //Puede ser eso o PCAWITHKNN
    MODE mod;
    if (stoi(method) == 1) { mod = PCAWITHKNN; }
    else              { mod = SIMPLEKNN; }

    Model ourModel(mod);

    ourModel.setK(10);
    ourModel.setAlpha(15);
    ourModel.setOutputFile(outputFile);

//    Le pasamos la direccion al dataset de training

    ourModel.train(trainSetName);

//  SavePPMFile("../test.ppm", ourModel.images[0].first, 92, 112,PPM_LOADER_PIXEL_TYPE_GRAY_8B, " ");

//    Evaluamos los tests y el modelo se guarda adentro los resultados
    ourModel.evaluate(testSetName);


//    Le pasamos el archivo donde guardarlos
//  ourModel.outputResults("path/to/results.csv");


    return 0;

    //ESTO ES CODIGO DEL TP PASADO QUE ME AYUDA A SABER COMO CARGAR ARCHIVOS PARA LEER Y ESCRIBIR
//    ifstream input(argv[1]);
//    ofstream resultsFile;
//    ofstream timeFile;
//    resultsFile.open("../experimentacion/results/results.out");
//    timeFile.open("../experimentacion/results/time",std::ios_base::app);
}

