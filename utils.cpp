//
// Created by mike on 07/05/18.
//

#include "utils.h"

unsigned int get_pixel_average(uchar* data, int i, int j, int height, int width){
    if(i > height)
        throw std::runtime_error("El direccionamiento vertical no puede ser mayor a la altura.");
    if(j > width)
        throw std::runtime_error("El direccionamiento horizontal no puede ser mayor al ancho.");
    unsigned int red = (unsigned int)(data[i*width*3 + j*3 + 0]);
    unsigned int green = (unsigned int)(data[i*width*3 + j*3 + 1]);
    unsigned int blue = (unsigned int)(data[i*width*3 + j*3 + 2]);
    return (unsigned int)((red+green+blue) / 3);
}

void read_image(std::string filename, uchar** data, int* width, int* height){
    *data = NULL;
    *width = 0;
    *height = 0;
    PPM_LOADER_PIXEL_TYPE pt = PPM_LOADER_PIXEL_TYPE_INVALID;

    bool ret = LoadPPMFile(data, width, height, &pt, filename.c_str());
    if (!ret || width == 0|| height == 0|| pt!=PPM_LOADER_PIXEL_TYPE_RGB_8B){
        throw std::runtime_error("Fallo al leer la imagen.");
    }
}

void test_image(){
    uchar* data = NULL;
    int width = 0, height = 0;
    std::string filename = "../prueba.ppm";
    read_image(filename, &data, &width, &height); // Ejemplo de llamada

    for (int h = 0; h < height; ++h){
        for (int w = 0; w < width; ++w){
            cout << get_pixel_average(data, h, w, height, width) << " "; // Ejemplo de lectura de un pixel
        }
        cout << endl;
    }
    delete [] data;
}

void test_load(){

    uchar* data = NULL;
    int width = 0, height = 0;
    PPM_LOADER_PIXEL_TYPE pt = PPM_LOADER_PIXEL_TYPE_INVALID;
    std::string filename = "../buda.0.ppm";
//  std::string filename = "../prueba.ppm";

    bool ret = LoadPPMFile(&data, &width, &height, &pt, filename.c_str());
    if (!ret || width == 0|| height == 0|| pt!=PPM_LOADER_PIXEL_TYPE_RGB_8B)
    {
        throw std::runtime_error("test_load failed");
    }

    delete [] data;
}

void test_save(){

    char comments[100];
    sprintf(comments, "%s", "Hello world");

    int width = 3, height =1;
    uchar* data = new uchar[width*height*3];
    data[0] = data[1] = data[2] = 100; // RGB
    data[3] = data[4] = data[5] = 150; // RGB
    data[6] = data[7] = data[8] = 245; // RGB
    std::string filename = "../prueba.ppm";

    bool ret = SavePPMFile(filename.c_str(),data,width,height,PPM_LOADER_PIXEL_TYPE_RGB_8B, comments);
    if (!ret)
    {
        std::cout << "ERROR: couldn't save Image to ppm file" << std::endl;
    }
}




int kNearestNeighbors(const vector<uchar*> &newFace, int k) {
//    vector< pair <int, int> > distances;
//
//    for (int i = 0; i < faceMatrix.size(); ++i) {
//        pair<int, int> dist (getSquaredNorm(faceMatrix[i], newFace), people[i]);
//        distances.push_back(dist);
//    }
//
//    vector<int> neighbors;
//    sort(distances.begin(), distances.end(), pairCompare);
//
//    for (int i = 0; i < k; ++i) {
//        neighbors.push_back(distances[i].second);
//    }
//
//    int nearest = neighbors[0];
//    int maxCount = count(neighbors.begin(), neighbors.end(), neighbors[0]);
//    for (int i = 1; i < neighbors.size(); ++i) {
//        int newCount = count(neighbors.begin(), neighbors.end(), neighbors[i]);
//        if (newCount > maxCount) {
//            maxCount = newCount;
//            nearest = neighbors[i];
//        }
//    }
//
//    return nearest;
}


int getSquaredNorm(const vector<uchar*> &v1,const vector<uchar*> &v2) {
//    int distance = 0;
//
//    for (int i = 0; i < v1.size(); ++i) {
//        distance += ((int)v1[i] - (int)v2[i]) ^ 2;  //Cuidado que la resta de uchar puede dar cualquier cosa.
//    }
//
//    return distance;
}


bool pairCompare(pair<uchar*, int> i, pair<uchar*, int> j) { return (i.first < j.first); }