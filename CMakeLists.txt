cmake_minimum_required(VERSION 3.7.2)
project(caralibro)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        main.cpp
        ppmloader/ppmloader.cpp
        ppmloader/ppmloader.h inutils.cpp inutils.h Model.cpp Model.h)

add_executable(caralibro ${SOURCE_FILES})
