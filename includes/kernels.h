#ifndef KERNELS_h
#define KERNELS_h

#include <Eigen/Dense>
#include <cmath>

namespace eg = Eigen;

double gaussian_kernel(eg::VectorXd x){

    const double n = (float) x.size();
    return  exp( - x.dot(x) / 2 ) / pow(2*M_PI, n/2);
    
}


#endif
