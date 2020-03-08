cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 14)

project(nnetnke)

add_executable(nnetnke
        file_man.cpp
        global.cpp
        nnet2nk.cpp
        util_functions.cpp)
