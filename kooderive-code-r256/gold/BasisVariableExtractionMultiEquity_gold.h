//
//
//                                  BasisVariableExtractionMultiEquity_gold.h
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence version 3

#ifndef BasisVariableExtractionMultiEquity_gold_H4534543
#define BasisVariableExtractionMultiEquity_gold_H4534543
#include <vector>

#include <gold/MatrixFacade.h>

class MaxCallVariables
{
public:
    MaxCallVariables(int stocks, int numberStockVariables,const std::vector<double>& vols,const std::vector<double>& expiries,const std::vector<double>& rhos 
        , const std::vector<double>& strikes_vec ,
        const std::vector<double>& rs_vec_,
        const std::vector<double>& ds_vec_);

    void operator()(const std::vector<double>& currentStocks_vec,std::vector<double>& variablesForOutput_vec) const;

    int numberUnderlyings() const
    {
        return stocks_;
    }
    int numberVariables() const
    {
        return numberVariables_;
    }

private:
    int stocks_;
    int numberVariables_;
    int numberOptionVariables_;
    int numberStockVariables_;

    std::vector<double> vols_,  expiries_,  rhos_,strikes_vec_,rs_vec_,ds_vec_;
    mutable std::vector<double> ordered_vec_;

};

// stock price values are dimensioned step, stock number, path

//basisFunctionVariables_cube(stepNumber,variable number,path);
template<class T>
void GenerateMultiDEquityBasisVariables(const CubeConstFacade<double>& stockPriceValues, 
                                        const T& variableSpecifier, 
                                        int pathOffsetForOutput,
                                        int StepsToSkip,
                                        int StepsToSkipForOutput,
                                        CubeFacade<double>& valuesOfBasisVariables_cube)
{

    int numberStocks = variableSpecifier.numberUnderlyings();
    int numberVariables = variableSpecifier.numberVariables();

    if (stockPriceValues.numberRows() != numberStocks)
        GenerateError("dimension mismatch in GenerateMultiDEquityBasisVariables");

    if (valuesOfBasisVariables_cube.numberRows() != numberVariables)
        GenerateError("dimension mismatch in GenerateMultiDEquityBasisVariables");

    if (stockPriceValues.numberLayers() != valuesOfBasisVariables_cube.numberLayers())
        GenerateError("dimension mismatch in GenerateMultiDEquityBasisVariables: number of steps does not agree");

    if (stockPriceValues.numberColumns() > valuesOfBasisVariables_cube.numberColumns()+pathOffsetForOutput)
        GenerateError("dimension mismatch in GenerateMultiDEquityBasisVariables: insufficient output paths in valuesOfBasisVariables matrix");

    int paths= stockPriceValues.numberColumns();
    int steps = stockPriceValues.numberLayers();

    std::vector<double> stock_vec(numberStocks);
    std::vector<double> variable_vec(numberVariables);

    for (int i=0; i < paths; ++i)
    {
        for (int s=StepsToSkip, outStep=StepsToSkipForOutput; s < steps; ++s,++outStep)
        {
            for (int k=0; k < numberStocks; ++k)
                stock_vec[k] = stockPriceValues(s,k,i);

            variableSpecifier(stock_vec,variable_vec);

            for (int v=0; v < numberVariables; ++v)
                valuesOfBasisVariables_cube(outStep,v,i+pathOffsetForOutput) = variable_vec[v];
        }
    }


}

#endif
