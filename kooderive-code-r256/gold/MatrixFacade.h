//
//
//                                                                                                                                         MatrixFacade.h
//
//
/*
Note does not contain memory allocators or deallocators, 
Simply intended to make it easy to access two-dim or three dim data stored in a vector-type structure. 

MatrixFacade - 2d non-const data
MatrixConstFacade 2d const data 
CubeFacade 3d non const data 
CubeConstFacade 3d const data 

*/


#ifndef MATRIX_FACADE_H
#define MATRIX_FACADE_H
#include "Errors.h"
#include <vector>

#include <iostream>


template<class T>
class MatrixFacade
{
public:

    MatrixFacade(T* dataStart,
        int rows,
        int columns);

    MatrixFacade( std::vector<T>& dataStart,
        int rows,
        int columns);

    T& operator()(int i, int j);

    T* operator[](int i);
    const T*  operator[](int i) const;

    const T& operator()(int i, int j) const;

    int rows() const 
    {
        return rows_;
    }

    int columns() const 
    {
        return columns_;
    }

private:
    T* data_start_;

    int rows_;
    int columns_;

};

template<class T>
MatrixFacade<T>::MatrixFacade(T* dataStart,
                              int rows,
                              int columns)
                              :
data_start_(dataStart), 
rows_(rows), 
columns_(columns)
{
}
template<class T>
MatrixFacade<T>::MatrixFacade( std::vector<T>& dataStart,
        int rows,
        int columns)
        : 
data_start_(&dataStart[0]), 
rows_(rows), 
columns_(columns)
{
#ifdef RANGE_CHECKING
    if (columns_* rows_>static_cast<int>(dataStart.size()))
        GenerateError("Matrix facade bigger than vector being facaded");
   
#endif
}
template<class T>
inline    
T& MatrixFacade<T>::operator()(int i, int j)
{
#ifdef RANGE_CHECKING
    if (i >= rows_|| i < 0)
        GenerateError("row number out of range");
    if (j >= columns_ || j < 0 )
        GenerateError("column number out of range");
#endif
    return data_start_[i*columns_+j];
}

template<class T>
inline
T* MatrixFacade<T>::operator[](int i)
{
#ifdef RANGE_CHECKING
    if (i >= rows_|| i < 0)
        GenerateError("row number out of range ");
   
#endif
    return data_start_+i*columns_;
}

template<class T>
inline
const T* MatrixFacade<T>::operator[](int i) const
{
#ifdef RANGE_CHECKING
    if (i >= rows_|| i < 0 )
        GenerateError("row number out of range");
   
#endif
    return data_start_+i*columns_;
}


template<class T>
inline
const T& MatrixFacade<T>::operator()(int i, int j) const
{
#ifdef _DEBUG
    if (i >= rows_|| i < 0)
        GenerateError("row number too big or negative");
    if (j >= columns_ || i < 0)
        GenerateError("column number too big or negative ");
#endif

    return data_start_[i*columns_+j];
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<class T>
class MatrixConstFacade
{
public:

    MatrixConstFacade(const T* const  dataStart,
        int rows,
        int columns);

    MatrixConstFacade(const std::vector<T>& dataStart,
        int rows,
        int columns);


    // not a copy constructor, it allows implicit conversion of matrixfacades to MatrixConstFacade
    MatrixConstFacade(const MatrixFacade<T>& original);




    const T* MatrixConstFacade<T>::operator[](int i) const;
    const T& operator()(int i, int j) const;

    int rows() const 
    {
        return rows_;
    }

    int columns() const 
    {
        return columns_;
    }


private:
   // const T* const data_start_;
     const T* data_start_;

    int rows_;
    int columns_;

};

template<class T>
MatrixConstFacade<T>::MatrixConstFacade(const MatrixFacade<T>& original) : data_start_(original[0]),rows_(original.rows()), columns_(original.columns())
{
    
}


template<class T>
MatrixConstFacade<T>::MatrixConstFacade(const T* const dataStart,
                                        int rows,
                                        int columns)
                                        :
data_start_(dataStart), 
rows_(rows), 
columns_(columns)
{
}

template<class T>
MatrixConstFacade<T>::MatrixConstFacade(const std::vector<T>& dataStart,
        int rows,
        int columns)
        : 
data_start_(&dataStart[0]), 
rows_(rows), 
columns_(columns)
{
#ifdef RANGE_CHECKING
    if (columns_* rows_>static_cast<int>(dataStart.size()) )
        GenerateError("Matrix const facade bigger than vector being facaded");
   
#endif
}


template<class T>
inline
const T*  MatrixConstFacade<T>::operator[](int i) const
{
#ifdef RANGE_CHECKING
    if (i >= rows_|| i < 0)
        GenerateError("row number too big");
   
#endif
    return data_start_+i*columns_;
}

template<class T>
inline
const T& MatrixConstFacade<T>::operator()(int i, int j) const
{
#ifdef RANGE_CHECKING
    if (i >= rows_|| i < 0)
        GenerateError("row number too big");
    if (j >= columns_ || j <0)
        GenerateError("column number too big");
#endif
    return data_start_[i*columns_+j];
}








//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class CubeFacade
{
public:


    CubeFacade(T* dataStart,
        int layers,
        int rows,
        int columns);

    CubeFacade( std::vector<T>& data,
        int layers,
        int rows,
        int columns);
	
	CubeFacade( MatrixFacade<T>& input);


    T& operator()(int i, int j, int k);

    const T& operator()(int i, int j, int k) const;

    int numberRows() const
    {   
        return rows_;
    }

    int numberColumns() const
    {   
        return columns_;
    }

    int numberLayers() const
    {   
        return layers_;
    }

	MatrixFacade<T>& operator[](int i)
	{
		return MatrixFacade<T>(data_start_+i*colsTimesRows_,rows_,columns_);

	}

private:
    T* data_start_;

    int layers_;
    int rows_;
    int columns_;

    int colsTimesRows_;

};

template<class T>
CubeFacade<T>::CubeFacade(T* data,
                          int layers,
                          int rows,
                          int columns)
                          :
data_start_(data), 
layers_(layers),
rows_(rows), 
columns_(columns),
colsTimesRows_(columns*rows)
{
}


template<class T>
CubeFacade<T>::CubeFacade( std::vector<T>& dataStart,
                          int layers,
                          int rows,
                          int columns)
                          :
data_start_(&dataStart[0]), 
layers_(layers),
rows_(rows), 
columns_(columns),
colsTimesRows_(columns*rows) 
{
    if (dataStart.size() < layers_*rows_*columns_)
        GenerateError("data vector too small for cube facade");
}


template<class T>
CubeFacade<T>::CubeFacade( MatrixFacade<T>& input) : data_start_(input[0]),layers_(1),rows_(input.rows()), columns_(input.columns()),colsTimesRows_(columns_*rows_)
{
}


template<class T>
inline
T& CubeFacade<T>::operator()(int i, int j,int k)
{
#ifdef RANGE_CHECKING
    if (j > rows_ || j < 0)
        GenerateError("row number too big");
    if (k > columns_ || k < 0)
        GenerateError("column number too big");
    if (i > layers_ || i < 0)
        GenerateError("layer number too big");
#endif
    return data_start_[i*colsTimesRows_ + j *columns_+k];
}

template<class T>
inline
const T& CubeFacade<T>::operator()(int i, int j, int k) const
{
#ifdef RANGE_CHECKING
    if (j > rows_ || j <0)
        GenerateError("row number too big or negative");
    if (k > columns_ || k <0)
        GenerateError("column number too big or negative");
    if (i > layers_ || i < 0)
        GenerateError("layer number too big or negative");
#endif
    return data_start_[i*colsTimesRows_ + j *columns_+k];
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class CubeConstFacade
{
public:


    CubeConstFacade(const T * const  dataStart,
        int layers,
        int rows,
        int columns);

    CubeConstFacade( const std::vector<T>& data,
        int layers,
        int rows,
        int columns);


    CubeConstFacade(const CubeFacade<T>& original);

    CubeConstFacade( const MatrixConstFacade<T>& input);


    const T& operator()(int i, int j, int k) const;

    MatrixConstFacade<T> operator[](int i) const;

     int numberRows() const
    {   
        return rows_;
    }

    int numberColumns() const
    {   
        return columns_;
    }

    int numberLayers() const
    {   
        return layers_;
    }


private:
    const T * const   data_start_;

    int layers_;
    int rows_;
    int columns_;

    int colsTimesRows_;

};

template<class T>
CubeConstFacade<T>::CubeConstFacade(const CubeFacade<T>& original) : data_start_(&original(0,0,0)), layers_(original.numberLayers()), rows_(original.numberRows()),
                                                                                 columns_(original.numberColumns())
{
    colsTimesRows_ =rows_*columns_;
}


template<class T>
CubeConstFacade<T>::CubeConstFacade(const T * const dataStart,
                                    int layers,
                                    int rows,
                                    int columns)
                                    :
data_start_(dataStart), 
layers_(layers),
rows_(rows), 
columns_(columns),
colsTimesRows_(columns*rows)
{
}


template<class T>
CubeConstFacade<T>::CubeConstFacade(const MatrixConstFacade<T>& input)
 : data_start_(input[0]),layers_(1),rows_(input.rows()), columns_(input.columns()),colsTimesRows_(columns_*rows_)
{
}

template<class T>
CubeConstFacade<T>::CubeConstFacade(const std::vector<T>& data,
                                    int layers,
                                    int rows,
                                    int columns)
                                    :
data_start_(&data[0]), 
layers_(layers),
rows_(rows), 
columns_(columns),
colsTimesRows_(columns*rows)
{
}



template<class T>
inline
const T& CubeConstFacade<T>::operator()(int i, int j, int k) const
{
#ifdef RANGE_CHECKING
    if (j > rows_ || j <0)
        GenerateError("row number too big or negative");
    if (k > columns_ || k <0)
        GenerateError("column number too big or negative");
    if (i > layers_ || i < 0)
        GenerateError("layer number too big or negative");
#endif
    return data_start_[i*colsTimesRows_ + j *columns_+k];
}

template<class T>
inline
MatrixConstFacade<T>  CubeConstFacade<T>::operator[](int i) const
{
        return MatrixConstFacade<T>(data_start_+i*colsTimesRows_, rows_,columns_);
}

template<class T>  void debugDumpVector(const T& input, const char* name )
{
#ifdef _DEBUG
    std::cout << "\n" << name << "\n";
for (size_t i=0; i < input.size(); ++i)
    std::cout << "," << input[i];
    
std::cout << "\n";
#endif
}

template<class T>  void debugDumpMatrix(const T& input, const char* name, int x, int y )
{
#ifdef _DEBUG
    std::cout << "\n" << name << "\n";

for (int i=0; i < x; ++i)
{
	std::cout << "\n";
    for (int j=0; j < y; ++j)
            std::cout   << input[i*y+j] << ",";
}

	std::cout << "\n";
#endif
}
template<class T>  void debugDumpMatrix(const MatrixConstFacade<T>& input, const char* name)
{
	debugDumpMatrix(&input(0,0),name,input.rows(),input.columns());
}

template<class T>  void debugDumpMatrix(const MatrixFacade<T>& input, const char* name)
{
	debugDumpMatrix(&input(0,0),name,input.rows(),input.columns());
}
template<class T>  void debugDumpCube(const T& input, const char* name, int x, int y, int z )
{
#ifdef _DEBUG
    std::cout << "\n" << name <<"\n";

for (int i=0; i < x; ++i)
{ std::cout << "\n" << i ;
    for (int j=0; j < y; ++j)
	{
		std::cout << "\n";
        for (int k=0; k < z; ++k)
            std::cout << "," << input[i*y*z+j*z+k];
	}
}
std::cout << "\n";
#endif
}


template<class T>  void debugDumpCube(const CubeFacade<T>& input, const char* name)
{
	debugDumpCube(&input(0,0,0),name,input.numberLayers(), input.numberRows(), input.numberColumns());
}


template<class T>  void debugDumpCube(const CubeConstFacade<T>& input, const char* name)
{
	debugDumpCube(&input(0,0,0),name,input.numberLayers(), input.numberRows(), input.numberColumns());
}
/// non debug versions

template<class T>  void dumpVector(const T& input, const char* name )
{

    std::cout << "\n" << name << "\n";
for (size_t i=0; i < input.size(); ++i)
    std::cout << "," << input[i];
    
std::cout << "\n";

}

template<class T>  void dumpMatrix(const T& input, const char* name, int x, int y )
{

    std::cout << "\n" << name << "\n";

for (int i=0; i < x; ++i)
{
	std::cout << "\n";
    for (int j=0; j < y; ++j)
            std::cout   << input[i*y+j] << ",";
}

	std::cout << "\n";

}
template<class T>  void dumpMatrix(const MatrixConstFacade<T>& input, const char* name)
{
	dumpMatrix(&input(0,0),name,input.rows(),input.columns());
}

template<class T>  void dumpMatrix(const MatrixFacade<T>& input, const char* name)
{
	dumpMatrix(&input(0,0),name,input.rows(),input.columns());
}
template<class T>  void dumpCube(const T& input, const char* name, int x, int y, int z )
{

    std::cout << "\n" << name <<"\n";

for (int i=0; i < x; ++i)
{ std::cout << "\n" << i ;
    for (int j=0; j < y; ++j)
	{
		std::cout << "\n";
        for (int k=0; k < z; ++k)
            std::cout << "," << input[i*y*z+j*z+k];
	}
}
std::cout << "\n";

}


template<class T>  void dedumpCube(const CubeFacade<T>& input, const char* name)
{
	dumpCube(&input(0,0,0),name,input.numberLayers(), input.numberRows(), input.numberColumns());
}


template<class T>  void dumpCube(const CubeConstFacade<T>& input, const char* name)
{
	dumpCube(&input(0,0,0),name,input.numberLayers(), input.numberRows(), input.numberColumns());
}

#endif
