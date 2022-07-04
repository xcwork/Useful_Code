
// (c) Mark Joshi 2009, 2010, 2013
// This code is released under the GNU public licence version 3


//  Utilities.h


#ifndef UTILITIES_H
#define UTILITIES_H
#include <gold/pragmas.h> 
#include <ctime>
#include <iostream> 
#include <thrust/device_vector.h>
#include <gold/MatrixFacade.h> 
#include <gold/math/cube_gold.h>
#include <gold/math/basic_matrix_gold.h>

double doInnerProduct(thrust::device_vector<float>& one_dev, thrust::device_vector<float>& two_dev);

inline
void TimeSoFar(int start, std::string mess)
{
	std::cout << mess.c_str() << " ";
	std::cout  << (clock()- start)/(static_cast<float>(CLOCKS_PER_SEC)) << "\n";
}


template<class T> 
thrust::device_vector<T> deviceVecFromStlVec(const std::vector<T>& input)
{
	thrust::host_vector<T> host(input.size());

	std::copy(input.begin(), input.end(), host.begin());

	return thrust::device_vector<T>(host);
}

template<class T> 
thrust::host_vector<T> hostVecFromStlVec(const std::vector<T>& input)
{
	thrust::host_vector<T> host(input.size());

	std::copy(input.begin(), input.end(), host.begin());

	return host;
}

template<class T> 
std::vector<T> stlVecFromHostVec(const thrust::host_vector<T>& input)
{
	std::vector<T> host(input.size());

	std::copy(input.begin(), input.end(), host.begin());

	return host;
}

template<class T> 
std::vector<T> stlVecFromDevVec(const thrust::device_vector<T>& input)
{
	thrust::host_vector<T> host(input.size());

	thrust::copy(input.begin(), input.end(), host.begin());

	return stlVecFromHostVec<T>(host);
}


template<class S, class T> 
thrust::host_vector<S> hostVecCastStlVec(const std::vector<T>& input)
{
	thrust::host_vector<S> host(input.size());

	for (int i=0; i < static_cast<int>(host.size()); ++i)
		host[i] = static_cast<S>(input[i]);

	return host;
}


template<class S, class T> 
std::vector<S> stlVecCastStlVec(const std::vector<T>& input)
{
	std::vector<S> newVec(input.size());

	for (int i=0; i < static_cast<int>(newVec.size()); ++i)
		newVec[i] = static_cast<S>(input[i]);

	return newVec;
}


template<class S, class T> 
Matrix_gold<S> MatrixCastDeviceVec(const thrust::device_vector<T>& input, int rows, int columns)
{
	thrust::host_vector<T> host(input);
	Matrix_gold<S> result(rows,columns,0.0);
	MatrixFacade<T> host_mat(&host[0],rows,columns);

	for (int i=0; i < rows; ++i)
		for (int j=0; j < columns; ++j)
			result(i,j) = static_cast<S>(host_mat(i,j));

	return result;
}

template<class S, class T> 
Matrix_gold<S> MatrixCastMatrix(const MatrixConstFacade<T>& input)
{
	
	Matrix_gold<S> result(input.rows(),input.columns(),static_cast<S>(0.0));

	for (int i=0; i < input.rows(); ++i)
		for (int j=0; j < input.columns(); ++j)
			result(i,j) = static_cast<S>(input(i,j));

	return result;
}
template<class T> 
thrust::device_vector<T> deviceVecFromCube(const Cube_gold<T>& input)
{
	return deviceVecFromStlVec(input.getDataVector());
}


void PartialSumsInt(thrust::device_vector<int>& input, int paths, thrust::device_vector<int>& output);

void doScatter(thrust::device_vector<float>& source, int offset, int points,int newPoints, thrust::device_vector<int>& indices,
			   thrust::device_vector<int>& selections_dev, 
			   thrust::device_vector<float>& wsp_dev,
			   int outOffset
			   );


void doScatterMulti(thrust::device_vector<float>& source, 
					int dataSize, 
					int points,
					int newPoints, 
					thrust::device_vector<int>& indices,
					thrust::device_vector<int>& selections_dev, 
					thrust::device_vector<float>& wsp_dev
			   );

#endif



