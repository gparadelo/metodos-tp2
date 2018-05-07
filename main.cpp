#include <iostream>
#include "utils.h"
#include "Model.h"
#include <fstream>
#include <chrono>


using namespace std;
using namespace std::chrono;
// Ejemplo de como  acceder a los pixeles de una imagen RGB


int main(int argc, char *argv[]) {

  test_load();
  test_save();
  test_image();

//    ASI SE USA, HAY QUE HACER QUE LEA FLAGS SIN IMPORTAR EL ORDEN
//    $ ./tp2 -m 1 -i train.csv -q test.csv -o result.csv


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
    simpleKnn.train("path/to/train.csv");


//    Evaluamos los tests y el modelo se guarda adentro los resultados
    simpleKnn.evaluate("test.csv");


//    Le pasamos el archivo donde guardarlos
    simpleKnn.outputResults("path/to/results.csv");


  return 0;
}

