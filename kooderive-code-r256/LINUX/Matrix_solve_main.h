

#ifndef MATRIX_SOLVE_MAIN_H
#define MATRIX_SOLVE_MAIN_H
#include <thrust/device_vector.h>
#include <gold/math/basic_matrix_gold.h>
// solve A^t x = y and over write y to put x in y
//void culaSolve(int A_columns, int A_rows, float* A_global, float* y_global, char trans ='N');
void thrustMultiplyByTransposeAAT(thrust::device_vector<float>& A, int rows, int columns, MatrixFacade<float>& output_mat);
void thrustMultiplyATx(thrust::device_vector<float>& A, int rows, int columns, thrust::device_vector<float>& x,
					   std::vector<float>& output_vec);

#endif
