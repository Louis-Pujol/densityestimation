#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "kernels.h"
#include "evaluate.h"

namespace py = pybind11;

PYBIND11_MODULE(kde_cpu, m){
    
    // First argument : name of the module, Second argument : py::module variable
    m.doc() = "pybind11 kde plugin";
    
    // Evaluation
    m.def("evaluate_bwvector", &evaluate_bwvector, py::arg("x"), py::arg("X"), py::arg("h"), py::arg("n_jobs")=1);
    m.def("evaluate_bwmatrix", &evaluate_bwmatrix, py::arg("x"), py::arg("X"), py::arg("h"), py::arg("n_jobs")=1);
    
    //attributes
    
}
