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

//    $ ./tp2 -m 1 -i train.csv -q test.csv -o result.csv

//    ifstream input(argv[1]);
//    ofstream resultsFile;
//    ofstream timeFile;
//    resultsFile.open("../experimentacion/results/results.out");
//    timeFile.open("../experimentacion/results/time",std::ios_base::app);

//    if (!input->good()) {
//        cout << "The input file isn't good";
//        assert(false);
//    }
//
//    *(input) >> totalPages >> totalLinks;


    Model simpleKnn(static_cast<MODE>(0));

    simpleKnn.setAlpha(5);
    simpleKnn.setK(10);

    simpleKnn.train("path/to/train.csv");


    simpleKnn.evaluate("test.csv");

    simpleKnn.outputResults("path/to/results.csv");


  return 0;
}

