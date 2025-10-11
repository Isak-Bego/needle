cmake_minimum_required(VERSION 3.29)
project(needle)

set(CMAKE_CXX_STANDARD 14)

add_executable(needle
    main.cpp
        layers/neuron.h
        layers/layer.h
        layers/network.h)
