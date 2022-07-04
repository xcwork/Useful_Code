//
//
//											Select_gold.cpp
//
//

#include <gold/Select_gold.h>

// purpose of this function is purely to force the compilation of template code
// so that any compiler errors are detected here
void TestCompilation_Select_gold()
{
	std::vector<float> input(10000);
	std::vector<bool> selections(10000);
	std::vector<float> output(10000);
	

	 SelectAndConcat( input, selections,  output);

	 MatrixConstFacade<float> matCF(input,10,1000);

	 selections.resize(1000);

	 int s= SelectAndConcatMulti(matCF, selections,  output);
}


