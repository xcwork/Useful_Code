//
//
//                      MultiD_path_gen_Partial_gold.h
//
//

// (c) Mark Joshi2015
// This code is released under the GNU public licence version 3

#ifndef MULTI_PATH_GEN_PARTIAL_GOLD_H
#define MULTI_PATH_GEN_PARTIAL_GOLD_H
#include<gold/MatrixFacade.h>

#include <gold/math/basic_matrix_gold.h>

#include <gold/math/cube_gold.h>
template<class D>
void ComputeDriftsBSModel(int numberStocks, 
                          const std::vector<D>& timesToStep,
                          double r,
                          const std::vector<D>& dividendrates,
                          MatrixFacade<D>& drifts_mat
                          )
{
    int steps = timesToStep.size();
    if (drifts_mat.columns() != numberStocks)
        GenerateError("size mismatch in ComputeDriftsBSModel");

    if (drifts_mat.rows() != steps)
        GenerateError("size mismatch in ComputeDriftsBSModel");

    double lt=0.0;
    for (int i=0; i < steps; ++i)
    {
        double dT=  timesToStep[i]-lt;
        lt=timesToStep[i];

        for (int j=0; i < numberStocks; ++j)
            drifts_mat(i,j) = (r-dividendrates[j])*dT;

    }

}

// optimized for pseudo-randoms not quasi
// generate path starting in the middle
// pseudo roots are step stock factor
// drifts are step stock
// output is step stock path
// normals are step factor path
template<class D>
class MultiDBS_PathGen_Partial_gold
{
public:
    MultiDBS_PathGen_Partial_gold( 
        int numberStocks, 
        int factors, 
        int stepsForEvolution, 
        const std::vector<D>& pseudoRoots_vec,
        const std::vector<D>& drifts_rd_vec
        );

    D getPaths(int paths,
        int pathOffsetForOutput, 
        int stepsToSkip,
        const std::vector<D>& currentStocks,
        CubeConstFacade<D>& randoms_normals_cube,
        int randomsStartStep, // specifies  how to associate normals in cube with steps in simulation
        CubeFacade<D>& stocks_out_cube) const;

private:
    int numberStocks_;
    int factors_;
    int stepsForEvolution_;
    Cube_gold<D> pseudoRoots_cube_;
    Matrix_gold<D> drifts_mat_;

    mutable std::vector<D> currentStocks_;
    mutable std::vector<D> correlatedVariates_;

};

template<class D>
MultiDBS_PathGen_Partial_gold<D>::MultiDBS_PathGen_Partial_gold(
    int numberStocks, 
    int factors, 
    int stepsForEvolution, 
    const std::vector<D>& pseudoRoots_vec,
    const std::vector<D>& drifts_rd_vec
    )
    :
numberStocks_(numberStocks),
    factors_(factors),
    stepsForEvolution_(stepsForEvolution),
    pseudoRoots_cube_(stepsForEvolution, numberStocks, factors,   pseudoRoots_vec),
    drifts_mat_(stepsForEvolution,numberStocks,drifts_rd_vec),correlatedVariates_(numberStocks)
{

    for (int i =0; i < stepsForEvolution; ++i)
    {
        for (int j=0; j < numberStocks; ++j)
        {
             D x=static_cast<D>(0.0);
      
            for (int f=0; f< factors; ++f)
                x+= pseudoRoots_cube_(i,j,f)*pseudoRoots_cube_(i,j,f);
            drifts_mat_(i,j)-=0.5*x;
        }
    }
}

template<class D>
D MultiDBS_PathGen_Partial_gold<D>::getPaths(int paths,
                                             int pathOffsetForOutput, 
                                             int stepsToSkip,
                                             const std::vector<D>& currentStocks,
                                             CubeConstFacade<D>& randoms_normals_cube,
                                             int randomStartStep,
                                             CubeFacade<D>& stocks_out_cube) const
{

    if (randomStartStep + stepsForEvolution_ - stepsToSkip > randoms_normals_cube.numberLayers())
        GenerateError("too few variates given in randoms_normals_cube MultiDBS_PathGen_Partial_gold ");


    if (randoms_normals_cube.numberRows() != factors_)
          GenerateError("incorrect factors given in randoms_normals_cube MultiDBS_PathGen_Partial_gold ");

    if (randoms_normals_cube.numberColumns() != paths)
          GenerateError("incorrect paths  given in randoms_normals_cube MultiDBS_PathGen_Partial_gold ");


    for (int p=0; p< paths; ++p)
    {
        currentStocks_ = currentStocks;

        for (int s=stepsToSkip, rstep = randomStartStep; s < stepsForEvolution_; ++s,++rstep)
        {
            for (int stock=0; stock<numberStocks_; ++stock)
            {
                double diff = drifts_mat_(s,stock);
                for (int f=0; f <factors_; ++f)
                    diff += pseudoRoots_cube_(s,stock,f)* randoms_normals_cube(rstep,f,p);
                double stockVal = currentStocks_[stock];
                stockVal*=exp(diff);
                stocks_out_cube(s,stock,p+pathOffsetForOutput) = stockVal;
                currentStocks_[stock] = stockVal;
            }

        }

    }

    return 0.0;
}


#endif
