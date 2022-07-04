//
//
//                                             BASIC_MATRIX_GOLD_H
//
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef BASIC_MATRIX_GOLD_H
#define BASIC_MATRIX_GOLD_H
#include <vector>
#include <gold/MatrixFacade.h>
#include <algorithm>

template<class T>
void MatrixArrayProduct(const MatrixConstFacade<T>& matrix, const T* inputarray, T* output)
{
    for (int i=0; i < matrix.rows(); ++i)
    {
        output[i]=static_cast<T>(0.0);
        for (int j=0; j < matrix.columns(); ++j)
            output[i]+= matrix(i,j)*inputarray[j];

    }

}

template<class T, class IT> T RobustReduction(IT begin, IT end)
{
	if (end-begin <=10)
	{
		T x=0;
		for (IT i = begin; i !=end; ++i)
			x+=*i;

		return x;
	}
	else
	{

		IT mid = begin+(end-begin)/2;

		return RobustReduction<T,IT>(begin,mid) + RobustReduction<T,IT>(mid,end);
	}
}

template<class T>
void MatrixArrayProduct(const MatrixConstFacade<T>& matrix, const std::vector<T>& inputarray, std::vector<T>& output)
{
   if (matrix.rows()!= output.size())
       GenerateError("matrix target mismatch in MatrixArrayProduct");

   if (matrix.columns()!= inputarray.size())
     GenerateError("matrix inputarray mismatch in MatrixArrayProduct");


    MatrixArrayProduct<T>(matrix,&inputarray[0],&output[0]);
}
template<class T>
void MatrixArrayProductRobust(const MatrixConstFacade<T>& matrix, const std::vector<T>& inputarray, std::vector<T>& output,std::vector<T>& workspace)
{
   if (matrix.rows()!= output.size())
       GenerateError("matrix target mismatch in MatrixArrayProduct");

   if (matrix.columns()!= inputarray.size())
     GenerateError("matrix inputarray mismatch in MatrixArrayProduct");

   workspace.resize(matrix.columns());

    for (int i=0; i < matrix.rows(); ++i)
    {
        output[i]=static_cast<T>(0.0);
        for (int j=0; j < matrix.columns(); ++j)
			workspace[j] =matrix(i,j)*inputarray[j]; 

//		std::sort(workspace.begin(),workspace.end());

//		for (int j=0; j < matrix.columns(); ++j)
  //          output[i]+= workspace[j];

		output[i] = RobustReduction<T,typename std::vector<T>::const_iterator >(workspace.begin(),workspace.end());

    }

}

template<class T>
void MatrixMatrixProduct(const MatrixConstFacade<T>& matrix1, const MatrixConstFacade<T>& matrix2, MatrixFacade<T>& matrix3)
{
#ifdef _DEBUG

    if (matrix1.columns() != matrix2.rows() || matrix1.rows() != matrix3.rows() || matrix2.columns() != matrix3.columns())
        GenerateError("matrix matrix product size mismatch");

#endif


    for (int i=0; i < matrix1.rows(); ++i)
    {
        for (int j=0; j < matrix2.columns(); ++j)
        {
            matrix3(i,j)=static_cast<T>(0.0);

            for (int k=0; k < matrix1.columns(); ++k)
                 matrix3(i,j)+= matrix1(i,k)*matrix2(k,j);




        }
 
    }

}

template<class T>
void MatrixMatrixProductRobust(const MatrixConstFacade<T>& matrix1, const MatrixConstFacade<T>& matrix2, MatrixFacade<T>& matrix3,std::vector<T>& workspace)
{


    if (matrix1.columns() != matrix2.rows() || matrix1.rows() != matrix3.rows() || matrix2.columns() != matrix3.columns())
        GenerateError("matrix matrix product size mismatch");


	workspace.resize(matrix1.columns());


    for (int i=0; i < matrix1.rows(); ++i)
    {
        for (int j=0; j < matrix2.columns(); ++j)
        {
            matrix3(i,j)=static_cast<T>(0.0);

            for (int k=0; k < matrix1.columns(); ++k)
                 workspace[k] = matrix1(i,k)*matrix2(k,j);

	//		std::sort(workspace.begin(),workspace.end());
			
		//	for (int k=0; k < matrix1.columns(); ++k)
			//	matrix3(i,j)+=workspace[k];

			matrix3(i,j) = RobustReduction<T,typename std::vector<T>::const_iterator>(workspace.begin(),workspace.end());

        }
 
    }

}

template<class T> 
void MatrixTimesItsTranspose(const MatrixConstFacade<T>& input,  MatrixFacade<T>& output)
{
    int r = input.rows();
    int f = input.columns();

    for (int i=0; i < r; ++i)
        for (int j=0; j <= r; ++j)
        {
            T res =0.0;

            for (int k=0; k < f; ++k)

				res += input(i,k)*input(j,k);

            output(i,j) = output(j,i) = res;

        }

}



template<class T> 
void MatrixTimesItsTranspose(const MatrixFacade<T>& input,  MatrixFacade<T>& output)
{
    int r = input.rows();
    int f = input.columns();

    for (int i=0; i < r; ++i)
        for (int j=0; j <= r; ++j)
        {
            T res =0.0;

            for (int k=0; k < f; ++k)
                res += input(i,k)*input(j,k);

            output(j,i) = res;
            output(i,j) = res;

        }

}



template<class T> 
void MatrixTranspose(const MatrixFacade<T>& input,  MatrixFacade<T>& output)
{
    int r = input.rows();
    int f = input.columns();

    for (int i=0; i < r; ++i)
        for (int j=0; j <f; ++j)
        {
            output(j,i) = input(i,j);

        }

}

/*
Basic matrix class designed to mesh well with MatrixFacades
so user can use Matrices naturally when desired but also
use Facades when needed.

Note 
operator MatrixFacade<T>() allows you to supply a Matrix_gold when a MatrixFacade is expected.

ditto for operator MatrixConstFacade<T>() const

*/
template<class T>
class Matrix_gold
{
public:
	Matrix_gold();
	Matrix_gold(int rows, int columns, T val);
	Matrix_gold(int rows, int columns, const std::vector<T>& data);
	Matrix_gold(const MatrixFacade<T>& input);
	Matrix_gold(const MatrixConstFacade<T>& input);
	Matrix_gold(const Matrix_gold<T>& input); // copy constructor
	Matrix_gold<T>& operator=(const Matrix_gold<T>& input); //assigment operator

template<class it>
	Matrix_gold(int rows, int columns, it dataIterator);

	operator MatrixFacade<T>() 
	{
		return dataFacade_;
	}

	operator MatrixConstFacade<T>() const
	{
		return dataConstFacade_;
	}

	T& operator()(int i, int j)
	{
		return dataFacade_(i,j);
	}

	T operator()(int i, int j) const
	{
		return dataConstFacade_(i,j);
	}


	std::vector<T>& getDataVector()
	{
		return data_;
	}

	const std::vector<T>& getDataVector() const
	{
		return data_;
	}

	int rows() const
	{
		return rows_;
	}

	int columns() const
	{
		return columns_;
	}

	MatrixFacade<T>& Facade()
	{	
		return dataFacade_;
	}

	

	const MatrixConstFacade<T>& ConstFacade() const
	{
		return dataConstFacade_;
	}




	T* operator[](int i)
	{	
		return dataFacade_[i];

	}

	const T* operator[](int i) const
	{	
		return &data_[i*columns_];

	}


private:

	std::vector<T> data_;
	MatrixFacade<T> dataFacade_;
	MatrixConstFacade<T> dataConstFacade_;

	int rows_;
	int columns_;



};

template<class T>
Matrix_gold<T>::Matrix_gold(const Matrix_gold<T>& input) : data_(input.data_), dataFacade_(data_,input.rows(),input.columns()),
 dataConstFacade_(data_,input.rows(),input.columns()), rows_(input.rows_), columns_(input.columns_)
{
}

template<class T>
Matrix_gold<T>& Matrix_gold<T>::operator=(const Matrix_gold<T>& input) //assigment operator
{
	data_= input.data_;
	rows_ = input.rows_;
	columns_ = input.columns_;

	dataFacade_ = MatrixFacade<T>(data_,rows_,columns_);
	dataConstFacade_ = MatrixConstFacade<T>(data_,rows_,columns_);
	return *this;
}

template<class T>
Matrix_gold<T>::Matrix_gold() : data_(1), dataFacade_(data_,0,0),
dataConstFacade_(data_,0,0), rows_(0), columns_(0)
{}

template<class T>
Matrix_gold<T>::Matrix_gold(int rows, int columns, T val) : data_(rows*columns,val), dataFacade_(data_,rows,columns),
dataConstFacade_(data_,rows,columns), rows_(rows), columns_(columns)
{}

template<class T>
Matrix_gold<T>::Matrix_gold(int rows, int columns, const std::vector<T>& data)
: data_(rows*columns), dataFacade_(data_,rows,columns),
dataConstFacade_(data_,rows,columns), rows_(rows), columns_(columns)
{
	std::copy(data.begin(),data.begin()+rows*columns,data_.begin());
}

template<class T>
template<class it>
Matrix_gold<T>::Matrix_gold(int rows, int columns, it dataIterator): data_(rows*columns), dataFacade_(data_,rows,columns),
dataConstFacade_(data_,rows,columns), rows_(rows), columns_(columns)
{
	std::copy(dataIterator,dataIterator+rows*columns,data_.begin());
}

template<class T>
Matrix_gold<T>::Matrix_gold(const MatrixFacade<T>& input): data_(input.rows()*input.columns()), dataFacade_(data_,input.rows(),input.columns()),
dataConstFacade_(data_,input.rows(),input.columns()), rows_(input.rows()), columns_(input.columns())
{
	std::copy(input[0],input[0]+data_.size(),data_.begin()); 
}

template<class T>
Matrix_gold<T>::Matrix_gold(const MatrixConstFacade<T>& input): data_(input.rows()*input.columns()), dataFacade_(data_,input.rows(),input.columns()),
dataConstFacade_(data_,input.rows(),input.columns()), rows_(input.rows()), columns_(input.columns())
{
	std::copy(input[0],input[0]+data_.size(),data_.begin()); 
}

#endif

