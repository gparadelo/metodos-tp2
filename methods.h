#ifndef METHODS_H
#define METHODS_H

#include <vector>
#include <iostream>
#include <string>

using namespace std;

vector<vector<int>> createTrainMatrix(string trainFaces);
    //Crea la matriz X que tiene una imagen en cada fila.

vector<int> getPeople(string trainFaces);
    //Crea un vector que indica en la posicion i de que persona es la cara en la fila i de X.

int kNearestNeighbors(vector<vector<int>> trainMatrix, vector<int> people, vector<int> newFace, int k);
    //Dado una matriz con las imagenes, un vector que indica quien pertenece cada imagen, una nueva cara
    //y un entero k, indica a quien pertenece la nueva cara utilizando los k vecinos mas cercanos.



#endif