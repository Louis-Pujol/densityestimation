#ifndef EVALUATE
#define EVALUATE

#include "kernels.h"
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <iostream>

namespace py = pybind11;
namespace eg = Eigen;

//headers
void evaluate_bwvector_bythread(int thread_id, int n_jobs, eg::ArrayXXd x, eg::ArrayXXd X, eg::ArrayXd h, std::vector<double>& output);
void evaluate_bwmatrix_bythread(int thread_id, int n_jobs, eg::MatrixXd x, eg::MatrixXd X, eg::MatrixXd h, std::vector<double>& output);

// Two function : one if bw is a vector, one if bw is a matrix
std::vector<double> evaluate_bwvector(eg::ArrayXXd x, eg::ArrayXXd X, eg::ArrayXd h, int n_jobs){
    
    const int n_output = x.rows();
    const int N = X.rows();
    const int d = X.cols();
    int i;
    
    // Define the number of threads. If n_jobs == -1, use all available threads
    int available_threads(std::thread::hardware_concurrency());
    int num_thread;
    if (n_jobs == -1){num_thread = available_threads;}
    else if (n_jobs > 0 and n_jobs <= available_threads){num_thread = n_jobs;}
    else if (n_jobs > available_threads){
        std::cout << "Error, too many threads asked" << std::endl;
        std::vector<double> err(0,5); 
        return err;
    }
    else{
        std::cout << "Error : incorrect value for n_jobs" << std::endl;
        std::vector<double> err(0,5);
        return err;
    }
    
    //Create the vector of threads and initialize them with by_thread function
    std::vector<std::thread> t(num_thread);
    
    std::vector<double> output(n_output, 0);
    output.reserve(n_output);
    
    
    for (int i=0; i<num_thread; ++i){
        t[i] = std::thread(evaluate_bwvector_bythread, i, num_thread,x, X, h, std::ref(output));
    }
    
    //Waiting for the end of the parallel execution
    for (int i=0; i<num_thread; ++i){
        t[i].join();
    }
    
    
    double prod_h(1.0);
    for (i=0; i<d; ++i){
        prod_h *= h(i);
    }
    
    for (i=0; i<n_output; i++){
    
        output[i] /= N*prod_h;
    }
    
    return output;
    
}
std::vector<double> evaluate_bwmatrix(eg::MatrixXd x, eg::MatrixXd X, eg::MatrixXd h, int n_jobs){
    
    const int n_output = x.rows();
    const int N = X.rows();
    //const int d = X.cols();
    int i;
    
    std::cout << h << std::endl << h.inverse() << std::endl << h.determinant() << std::endl;
    
    // Define the number of threads. If n_jobs == -1, use all available threads
    int available_threads(std::thread::hardware_concurrency());
    int num_thread;
    if (n_jobs == -1){num_thread = available_threads;}
    else if (n_jobs > 0 and n_jobs <= available_threads){num_thread = n_jobs;}
    else if (n_jobs > available_threads){
        std::cout << "Error, too many threads asked" << std::endl;
        std::vector<double> err(0,5); 
        return err;
    }
    else{
        std::cout << "Error : incorrect value for n_jobs" << std::endl;
        std::vector<double> err(0,5);
        return err;
    }
    
    //Create the vector of threads and initialize them with by_thread function
    std::vector<std::thread> t(num_thread);
    
    std::vector<double> output(n_output, 0);
    output.reserve(n_output);
    
    
    for (int i=0; i<num_thread; ++i){
        t[i] = std::thread(evaluate_bwmatrix_bythread, i, num_thread,x, X, h, std::ref(output));
    }
    
    //Waiting for the end of the parallel execution
    for (int i=0; i<num_thread; ++i){
        t[i].join();
    }
    
    
    double det(h.determinant());
    for (i=0; i<n_output; i++){
    
        output[i] /= N*det;
    }
    
    return output;
    
}

// Routines to parallelize on several threads
void evaluate_bwvector_bythread(int thread_id, int n_jobs, eg::ArrayXXd x, eg::ArrayXXd X, eg::ArrayXd h, std::vector<double>& output){
    
    const int n_output = x.rows();
    const int N = X.rows();
    
    
    double local_sum(0.0);
    for (int i=thread_id; i<n_output; i+=n_jobs){
        local_sum = 0.0;
        for (int j=0; j<N; j++){
            local_sum += gaussian_kernel( (x.row(i) - X.row(j)) / h.transpose() );
        }
        output[i] = local_sum;
    }
}
void evaluate_bwmatrix_bythread(int thread_id, int n_jobs, eg::MatrixXd x, eg::MatrixXd X, eg::MatrixXd h, std::vector<double>& output){
    
    const int n_output = x.rows();
    const int N = X.rows();
    
    eg::MatrixXd h_inv = h.inverse();
    
    double local_sum(0.0);
    for (int i=thread_id; i<n_output; i+=n_jobs){
        local_sum = 0.0;
        for (int j=0; j<N; j++){
            std::cout << x.row(i) << std::endl;
            local_sum += gaussian_kernel( h_inv * (eg::VectorXd)(x.row(i) - X.row(j)) );
        }
        output[i] = local_sum;
    }
}

#endif
