#ifndef METHODS_H
#define METHODS_H

#include <vector>
#include <iostream>
#include <string>
#include <utility>
#include <algorithm>

using namespace std;
typedef unsigned char uchar;

class imageMatrix {
private:
    vector<vector<uchar*>> faceMatrix;
    //Contiene en cada fila una imagen.

    vector<int> people;
    //Contiene en la posicion i, a la persona que le corresponde la imagen en la fila i de faceMatrix.

    bool pairCompare(pair<uchar*, int> i, pair<uchar*, int> j);
    //Para comparar pares (distancia, persona).

    int getSquaredNorm(const vector<uchar*> &v1,const vector<uchar*> &v2);
    //Calcula la distancia entre dos vectores utilizando la norma Euclidea (||v1 -v2||_2).
    //No realiza la raiz cuadrada final ya que no es necesaria para nuestro uso.

public:
    imageMatrix();
    //Toma las imagenes y crea faceMatrix y people
    //IMPLEMENTAR


    int kNearestNeighbors(const vector<uchar*> &newFace, int k);
    //Dado una matriz con las imagenes, un vector que indica quien pertenece cada imagen, una nueva cara
    //y un entero k, indica a quien pertenece la nueva cara utilizando los k vecinos mas cercanos.



};


#endif