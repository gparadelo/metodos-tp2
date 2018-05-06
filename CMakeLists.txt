cmake_minimum_required(VERSION 3.8)
project(caralibro)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp methods.cpp methods.h)
add_executable(caralibro ${SOURCE_FILES})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")