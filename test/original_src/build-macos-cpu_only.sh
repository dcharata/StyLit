#!/bin/sh
clang++ -g src/ebsynth.cpp src/ebsynth_cpu.cpp src/ebsynth_nocuda.cpp -I"include" -std=c++11 -o bin/ebsynth
