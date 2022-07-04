//
//
//                                                                                                  ConvertsQLToThrust.h
//
//

#ifndef CONVERTS_TO_THRUST_H
#define CONVERTS_TO_THRUST_H

#include <thrust/host_vector.h>
#include <vector>
#include <ql/math/matrix.hpp>
#include <gold/MatrixFacade.h>

template<class T> 
thrust::host_vector<T>  cubeFromMatrixVector(const std::vector<Matrix>& input);

template<class T> 
void  cubeFromMatrixVector(const std::vector<Matrix>& input, thrust::host_vector<T>& output);

/////////////// implementations

template<class T> 
thrust::host_vector<T>  cubeFromMatrixVector(const std::vector<Matrix>& input)
{
    int layers = static_cast<int>(input.size());
    if (layers ==0)
        return thrust::host_vector<T>(0);

    int rows = static_cast<int>(input[0]. rows());
    int cols = static_cast<int>(input[0]. columns());

    thrust::host_vector<T>  output(layers*rows*cols);
    
    CubeFacade<T> C(&output[0],
        layers,
        rows,
        cols);

    for (int i=0; i < layers; ++i)
        for (int j=0; j < rows; ++j)
            for (int k =0; k < cols; ++k)
                C(i,j,k) = static_cast<T>(input[i][j][k]);

    return output;

}

template<class T> 
void  cubeFromMatrixVector(const std::vector<Matrix>& input, thrust::host_vector<T>& output)
{
    int layers = static_cast<int>(input.size());
    if (layers ==0)
    {
        output.resize(0);
        return;
    }

    int rows = static_cast<int>(input[0]. rows());
    int cols = static_cast<int>(input[0]. columns());

    output.resize(layers*rows*cols);
    
    CubeFacade<T> C(&output[0],
        layers,
        rows,
        cols);

    for (int i=0; i < layers; ++i)
        for (int j=0; j < rows; ++j)
            for (int k =0; k < cols; ++k)
                C(i,j,k) = static_cast<T>(input[i][j][k]);


}

#endif
