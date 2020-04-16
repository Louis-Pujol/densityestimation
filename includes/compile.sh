# !/usr/bin/bash

c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` pybind_module.cc -o kde_cpu`python3-config --extension-suffix`
