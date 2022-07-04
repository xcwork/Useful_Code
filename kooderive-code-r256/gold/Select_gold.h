//
//
//								Select_gold.h
//
//

#ifndef SELECT_GOLD_H
#define SELECT_GOLD_H

#include <vector>
#include <gold/MatrixFacade.h>

template<class T, class S>
void SelectAndConcat(const std::vector<T>& input, const std::vector<S>& selections, std::vector<T>& output)
{
	output.resize(0);
	output.reserve(input.size());

	for (size_t i=0; i < input.size(); ++i)
		if (selections[i])
			output.push_back(input[i]);


}


template<class T, class S>
int SelectAndConcatMulti(const MatrixConstFacade<T>& input, const std::vector<S>& selections, std::vector<T>& output)
{

	int count=0; 
	for (size_t i=0; i < input.columns(); ++i)
		if (selections[i])
			++count;

	output.resize(input.rows()*count);

	MatrixFacade<T> output_mat(output,input.rows(),count);

	int j=0;
	for (int i=0; i < input.columns(); ++i)
		if (selections[i])
		{
			for (int k=0; k < input.rows(); ++k)
				output_mat(k,j) = input(k,i);
			++j;
		}


	return count;

}



#endif
