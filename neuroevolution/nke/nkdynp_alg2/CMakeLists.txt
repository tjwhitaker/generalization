cmake_minimum_required(VERSION 3.14)
project(nkdynpe)

set(CMAKE_CXX_STANDARD 14)

add_executable(nkdynpe
        file_man.cpp
        global.cpp
        nkdynp.cpp)
