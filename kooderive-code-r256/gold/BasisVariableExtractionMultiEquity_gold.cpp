//
//
//                                  BasisVariableExtractionMultiEquity_gold.cpp
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence version 3

#include <gold/BasisVariableExtractionMultiEquity_gold.h>
#include <algorithm>
#include <gold/BSFormulas_gold.h>

MaxCallVariables::MaxCallVariables(int stocks, 
                                   int numberStockVariables,
                                   const std::vector<double>& vols,const std::vector<double>& expiries,const std::vector<double>& rhos ,
                                   const std::vector<double>& strikes_vec ,
                                   const std::vector<double>& rs_vec,
                                   const std::vector<double>& ds_vec
                                   ) : stocks_(stocks), 
                                   numberStockVariables_(numberStockVariables),
                                   numberOptionVariables_(static_cast<int>(vols.size())), vols_(vols),  expiries_(expiries),  
                                   rhos_(rhos),strikes_vec_(strikes_vec),rs_vec_(rs_vec), ds_vec_(ds_vec)
{
    numberVariables_=numberStockVariables_+numberOptionVariables_;
    if (vols_.size() != expiries_.size())
        GenerateError("Size mismatch in MaxCallVariables betweens vols and expiries");
    if (vols_.size() != rhos_.size())
        GenerateError("Size mismatch in MaxCallVariables betweens vols and rho");

}

void MaxCallVariables::operator()(const std::vector<double>& currentStocks_vec,std::vector<double>& variablesForOutput_vec) const
{
#ifdef _DEBUG
    if (variablesForOutput_vec.size()!=numberVariables_)
        GenerateError("size mismatch in MaxCallVariables::operator()");
#endif
    ordered_vec_ = currentStocks_vec;
    std::sort(ordered_vec_.rbegin(),ordered_vec_.rend());

    std::copy(ordered_vec_.begin(),ordered_vec_.begin()+numberStockVariables_,variablesForOutput_vec.begin());

    for (int i=0; i < numberOptionVariables_; ++i)
        variablesForOutput_vec[i+numberStockVariables_] = BSCallTwoMaxDrezner(ordered_vec_[0] ,ordered_vec_[1],strikes_vec_[i] , expiries_[i], 
         rs_vec_[i] , 
       ds_vec_[i] ,ds_vec_[i],vols_[i] ,vols_[i] , rhos_[i] );
        //MargrabeBlackScholes(ordered_vec_[0],ordered_vec_[1],expiries_[i],rhos_[i],vols_[i],vols_[i],0.0,0.0);


}

  