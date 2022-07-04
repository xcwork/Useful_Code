//
//
//                                  EarlyExercisableMultiEquityPayoff_gold.cpp
//
//
#include <gold/EarlyExercisableMultiEquityPayoff_gold.h>
#include <gold/Errors.h>
#include <algorithm>

MaxCall::MaxCall(int stocks, double Strike) : stocks_(stocks), Strike_(Strike)
{
}


double MaxCall::operator()(const std::vector<double>& currentStocks) const
{
#ifdef _DEBUG
    if (currentStocks.size() != stocks_)
        GenerateError("size mismatch in maxcall");
#endif 

    double m = *std::max_element(currentStocks.begin(),currentStocks.end());

    return std::max(m-Strike_,0.0);


}

