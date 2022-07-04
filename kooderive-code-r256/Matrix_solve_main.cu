#include <Matrix_solve_main.h>
#include <LS_main_cula.h>
//#include <cula_lapack_device.h>

#include <gold/Timers.h>
#include <cutil.h>
#include <thrust/device_vector.h>
#include <gold/math/basic_matrix_gold.h>
#include <Errors.h>
#include <thrust/inner_product.h >
#include <string>
/*
void checkStatus(culaStatus status)
{
    char buffer[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buffer, sizeof(buffer));
	std::cout << buffer << "\n";;
}

// solve A^t x = y least squares and over write y to put x in y
void culaSolve(int A_columns, int A_rows, float* A_global, float* y_global, char trans)
{
//	static double totalTime =0.0;
	
	int m = A_rows;
	int n= A_columns ;
	int nrhs = 1;
	int lda = m;
	int ldb = std::max(m,n);

//	Timer h1;

	culaStatus status= culaDeviceSgels(trans,m,n,nrhs,A_global,lda,y_global,ldb);
    
//	double t=h1.timePassed();
///	totalTime += t;

//	std::cout << " regression time is " << t << " "<< A_columns << " " << A_rows <<  " " << " "  << totalTime <<" \n";

	checkStatus(status);
};

*/

void thrustMultiplyByTransposeAAT(thrust::device_vector<float>& A, int rows, int columns, MatrixFacade<float>& output_mat)
{
	
	static double totalTime =0.0;

	if (output_mat.rows() != columns || output_mat.rows() != columns)
		GenerateError("in thrustMultiplyByTransposeAAT output_mat must be square of size columns ");

	Timer h1;


	for (int i=0; i < columns; ++i)
		for (int j=0; j <=i; ++j)
		{
			output_mat(i,j) = output_mat(j,i) = thrust::inner_product(A.begin()+rows*i,A.begin()+rows*(i+1),A.begin()+rows*j,0.0f);
		}

	double t=h1.timePassed();
	totalTime += t;

	std::cout << " multiplication AAT time is " << t << " " << totalTime <<" \n";


}

void thrustMultiplyATx(thrust::device_vector<float>& A, int rows, int columns, thrust::device_vector<float>& x,
					   std::vector<float>& output_vec)
{
	
	static double totalTime =0.0;

	if (output_vec.size() != columns )
		GenerateError("in thrustMultiplyATx output_vec must be of size columns ");

	if (x.size() != rows )
		GenerateError("in thrustMultiplyATx x must be of size rows ");

	Timer h1;


	for (int i=0; i < columns; ++i)
		output_vec[i] = thrust::inner_product(A.begin()+rows*i,A.begin()+rows*(i+1),x.begin(),0.0f);

	double t=h1.timePassed();
	totalTime += t;

	std::cout << " multiplication matrix vec time is " << t << " " << totalTime <<" \n";


}
