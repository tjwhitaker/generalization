cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 14)

project(NNcommittee)

add_executable(NNcommitteee
        file_man.cpp
        global.cpp
        NNcommittee.cpp
        util_functions.cpp)

