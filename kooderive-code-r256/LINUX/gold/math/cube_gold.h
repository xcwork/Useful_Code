//
//
//                                             CUBE_GOLD_H
//
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef CUBE_GOLD_H
#define CUBE_GOLD_H
#include <vector>
#include <gold/MatrixFacade.h>
#include <algorithm>


/*
Basic cube class designed to mesh well with CubeFacades
so user can use Cubes naturally when desired but also
use Facades when needed.

Note 
operator CubeFacade<T>() allows you to supply a Cube_gold when a CubeFacade is expected.

ditto for operator CubeConstFacade<T>() const

*/
template<class T>
class Cube_gold
{
public:
	Cube_gold(int layers=1, int rows=1, int columns=1, T val=0.0);
	Cube_gold(int layers, int rows, int columns, const std::vector<T>& data);
	Cube_gold(const CubeFacade<T>& input);
	Cube_gold(const CubeConstFacade<T>& input);
	


template<class it>
	Cube_gold(int layers, int rows, int columns, it dataIterator);

	operator CubeFacade<T>() 
	{
		return Facade();
	}

	operator CubeConstFacade<T>() const
	{
		return  ConstFacade();
	}

	T& operator()(int i, int j, int k)
	{
		return data_[i*rows_*columns_+j*columns_+k];
	}

	T operator()(int i, int j, int k) const
	{
		return data_[i*rows_*columns_+j*columns_+k];
	}

	MatrixFacade<T> operator[](int i)
	{
		return MatrixFacade<T>(&data_[i*rows_*columns_],rows_,columns_);
	}

	MatrixConstFacade<T> operator[](int i) const
	{
		return MatrixConstFacade<T>(&data_[i*rows_*columns_],rows_,columns_);
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

	int layers() const
	{
		return layers_;
	}

	CubeFacade<T> Facade()
	{	
		return CubeFacade<T>(data_,layers_,rows_,columns_);
	}

	CubeConstFacade<T> ConstFacade() const
	{
		return CubeConstFacade<T>(data_,layers_,rows_,columns_);
	}



private:

	std::vector<T> data_;
//	CubeFacade<T> dataFacade_;
//	CubeConstFacade<T> dataConstFacade_;

	int layers_;
	int rows_;
	int columns_;



};


template<class S, class T>
S caster(const T& obj)
{
	return static_cast<S>(obj);
}


template<class S, class T>
Cube_gold<S> CubeTypeConvert(const Cube_gold<T>& input)
{
	
	Cube_gold<S> res(input.layers(),input.rows(),input.columns(),static_cast<S>(0));

	std::transform(input.getDataVector().begin(),input.getDataVector().end(), res.getDataVector().begin(),caster<S,T>);

	return res;
}





// template implementations

template<class T>
Cube_gold<T>::Cube_gold(int layers, int rows, int columns, T val) : data_(layers*rows*columns,val), 
//dataFacade_(data_,layers,rows,columns),
//dataConstFacade_(data_,layers,rows,columns),
layers_(layers), rows_(rows), columns_(columns)
{}

template<class T>
Cube_gold<T>::Cube_gold(int layers,int rows, int columns, const std::vector<T>& data)
: data_(layers*rows*columns), 
//dataFacade_(data_,layers,rows,columns),
//dataConstFacade_(data_,layers,rows,columns),
layers_(layers), rows_(rows), columns_(columns)
{
    if (data.size()!=data_.size())
        GenerateError("Missized data vector in cube.");
	std::copy(data.begin(),data.begin()+layers*rows*columns,data_.begin());
}

template<class T>
template<class it>
Cube_gold<T>::Cube_gold(int layers,int rows, int columns, it dataIterator): data_(layers*rows*columns), 
//dataFacade_(data_,layers,rows,columns),
//dataConstFacade_(data_,layers,rows,columns),
layers_(layers), rows_(rows), columns_(columns)
{
	std::copy(dataIterator,dataIterator+layers*rows*columns,data_.begin());
}

template<class T>
Cube_gold<T>::Cube_gold(const CubeFacade<T>& input): data_(input.numberLayers()*input.numberRows()*input.numberColumns()), 
//dataFacade_(data_,input.numberLayers(),input.numberRows(),input.numberColumns()),
//dataConstFacade_(data_,input.numberLayers(),input.numberRows(),input.numberColumns()),
layers_(input.numberLayers()), rows_(input.numberRows()), columns_(input.numberColumns())
{
	std::copy(&input(0,0,0),&input(0,0,0)+data_.size(),data_.begin()); 
}

template<class T>
Cube_gold<T>::Cube_gold(const CubeConstFacade<T>& input): data_(input.numberLayers()*input.numberRows()*input.numberColumns()), 
//dataFacade_(data_,input.numberLayers(),input.numberRows(),input.numberColumns()),
//dataConstFacade_(data_,input.numberLayers(),input.numberRows(),input.numberColumns()),
layers_(input.numberLayers()), rows_(input.numberRows()), columns_(input.numberColumns())
{
	std::copy(&input(0,0,0),&input(0,0,0)+data_.size(),data_.begin()); 
}
#endif

