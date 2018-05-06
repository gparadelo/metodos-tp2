#ifndef METHODS_H
#define METHODS_H

#include <vector>
#include <iostream>
#include <string>
#include <utility>
#include <algorithm>

using namespace std;

vector<vector<int>> createTrainMatrix(string trainFaces);
    //Crea la matriz X que tiene una imagen en cada fila.


vector<int> getPeople(string trainFaces);
    //Crea un vector que indica en la posicion i de que persona es la cara en la fila i de X.


int kNearestNeighbors(vector< vector <int> > trainMatrix, vector<int> people, vector<int> newFace, int k);
    //Dado una matriz con las imagenes, un vector que indica quien pertenece cada imagen, una nueva cara
    //y un entero k, indica a quien pertenece la nueva cara utilizando los k vecinos mas cercanos.


int getSquaredNorm(vector<int> v1,vector<int> v2);
    //Calcula la distancia entre dos vectores utilizando la norma Euclidea (||v1 -v2||_2).
    //No realiza la raiz cuadrada final ya que no es necesaria para nuestro uso.


bool pairCompare(pair<int, int> i, pair<int, int> j);



#endif