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
    return "";
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


    char *alphaStr = getCmdOption(argv, argv + argc, "-alpha");

    char *kStr = getCmdOption(argv, argv + argc, "-k");

    int k = kStr == "" ? 10 : stoi(kStr);

    int alpha = alphaStr == "" ? 15 : stoi(alphaStr);

    char *outputMetricsFileName = getCmdOption(argv, argv + argc, "-metrics");

    char *outputTimesFileName = getCmdOption(argv, argv + argc, "-time");

    char *outputPredictionsFileName = getCmdOption(argv, argv + argc, "-o");




    MODE mod;
    if (stoi(method) == 1) { mod = PCAWITHKNN; }
    else              { mod = SIMPLEKNN; }

    Model ourModel(mod);

    ourModel.setK(k);
    ourModel.setAlpha(alpha);


    if(outputMetricsFileName != "") {
        ofstream outputMetricsFile;
        outputMetricsFile.open(outputMetricsFileName, std::ios_base::app);
        ourModel.setMetricsFile(outputMetricsFile);
    }

    if(outputTimesFileName != "") {
        ofstream outputTimesFile;
        outputTimesFile.open(outputTimesFileName, std::ios_base::app);
        ourModel.setTimesFile(outputTimesFile);
    }


    ourModel.train(trainSetName);

//  SavePPMFile("../test.ppm", ourModel.images[0].first, 92, 112,PPM_LOADER_PIXEL_TYPE_GRAY_8B, " ");

//    Evaluamos los tests y el modelo se guarda adentro los resultados
    ourModel.evaluate(testSetName);


//    Le pasamos el archivo donde guardarlos
  ourModel.outputResults();


    return 0;

    //ESTO ES CODIGO DEL TP PASADO QUE ME AYUDA A SABER COMO CARGAR ARCHIVOS PARA LEER Y ESCRIBIR
//    ifstream input(argv[1]);
//    ofstream resultsFile;
//    ofstream timeFile;
//    resultsFile.open("../experimentacion/results/results.out");
//    timeFile.open("../experimentacion/results/time",std::ios_base::app);
}

