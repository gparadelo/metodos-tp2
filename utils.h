//
// Created by mike on 07/05/18.
//

#ifndef CARALIBRO_UTILS_H
#define CARALIBRO_UTILS_H

#include "ppmloader/ppmloader.h"
#include <iostream>

using namespace std;
unsigned int get_pixel_average(uchar* data, int i, int j, int height, int width);
void read_image(std::string filename, uchar** data, int* width, int* height);
void test_image();
void test_load();
void test_save();

#endif //CARALIBRO_UTILS_H
