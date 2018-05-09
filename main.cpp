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
/*
    test_load();
    test_save();
    test_image();
*/
//    ASI SE USA, HAY QUE HACER QUE LEA FLAGS SIN IMPORTAR EL ORDEN
//    $ ./tp2 -m 1 -i train.csv -q test.csv -o result.csv

/*
    if (!cmdOptionExists(argv, argv + argc, "-m")) {
        cout << "necesito: -m <method>";
        return 0;
    }
*/
    if (!cmdOptionExists(argv, argv + argc, "-i")) {
        cout << "necesito: -i <train_set>";
        return 0;
    }
/*
    if (!cmdOptionExists(argv, argv + argc, "-1")) {
        cout << "necesito: -q <train_set>";
        return 0;
    }
*/

    char *trainSetName = getCmdOption(argv, argv + argc, "-i") + 3;

//  char *testSetName = getCmdOption(argv, argv + argc, "-q") + 3;




    //ESTO ES CODIGO DEL TP PASADO QUE ME AYUDA A SABER COMO CARGAR ARCHIVOS PARA LEER Y ESCRIBIR
//    ifstream input(argv[1]);
//    ofstream resultsFile;
//    ofstream timeFile;
//    resultsFile.open("../experimentacion/results/results.out");
//    timeFile.open("../experimentacion/results/time",std::ios_base::app);



    //Creamos una instancia de nuestra clase model y la instanciamos en modo SIMPLEKNN
    //Puede ser eso o PCAWITHKNN
    Model simpleKnn(SIMPLEKNN);


//   Descomentar esto solo en PCA
//    simpleKnn.setAlpha(5);


    simpleKnn.setK(10);


//    Le pasamos la direccion al dataset de training

    simpleKnn.train(trainSetName);


//    Evaluamos los tests y el modelo se guarda adentro los resultados
//  simpleKnn.evaluate(testSetName);


//    Le pasamos el archivo donde guardarlos
    simpleKnn.outputResults("path/to/results.csv");


    return 0;
}

