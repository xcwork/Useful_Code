//
//
//                                  cashFlowGeneration_gold.h
//
//

// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3

#ifndef CASH_FLOW_GENERATION_GOLD_H
#define CASH_FLOW_GENERATION_GOLD_H

#include <vector>
#include <gold/MatrixFacade.h> 

template<class T,class D>
void cashFlowGenerator_gold(
                            std::vector<D>& genFlows1,
                            std::vector<D>& genFlows2, 
                            const std::vector<D>& auxData, 
                            int paths, 
                            int numberSteps,                       
                            const std::vector<D>& rates1, 
                            const std::vector<D>& rates2, 
                            const std::vector<D>& rates3, 
                            const std::vector<D>& forwards, 
                            const std::vector<D>& discRatios)   
{    
    int numberRates = numberSteps;

    T product(auxData, numberSteps);

    MatrixConstFacade<D> rates1Matrix(&rates1[0],numberSteps,paths);
    MatrixConstFacade<D> rates2Matrix(&rates2[0],numberSteps,paths);
    MatrixConstFacade<D> rates3Matrix(&rates3[0],numberSteps,paths);
    CubeConstFacade<D> forwardsCube(&forwards[0],numberSteps, numberRates, paths);
    CubeConstFacade<D> discountsCube(&discRatios[0],numberSteps, numberRates+1, paths);


    genFlows1.resize(paths*numberSteps);
    genFlows2.resize(paths*numberSteps);

    
    for (int p=0; p < paths; ++p)
    { 
        product.newPath();

        bool done=false;

        for (int i =0; i < numberSteps; ++i)        
        {

            D flow1=0.0;        
            D flow2=0.0;

            if (!done)
            {               
                 D rate1 = rates1Matrix(i,p);
                D rate2 = rates2Matrix(i,p);   
                D rate3 = rates3Matrix(i,p); 

                done = product.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwardsCube,discountsCube,p);

            }

            int location = i*paths+p;

            genFlows1[location] = flow1;
            genFlows2[location] = flow2;
        
        }               
    }
}


// this runs over the same path up to the conditional step and uses a different path thereafter
// to make something more efficient would require the product to remember its state at the conditional step
// the purpose is to permit the running of subsimulations
template<class T,class D>
void cashFlowGeneratorConditional_gold(
                            std::vector<D>& genFlows1,
                            std::vector<D>& genFlows2, 
							// unconditional data
                            const std::vector<D>& auxData, 
                            int paths, 
                            int numberSteps,                       
                            const std::vector<D>& rates1, 
                            const std::vector<D>& rates2, 
                            const std::vector<D>& rates3, 
                            const std::vector<D>& forwards, 
                            const std::vector<D>& discRatios,
							// conditional data
							int distinguishedPath,
							int subPaths,
							int conditionalStep,
							const std::vector<D>& conditionalRates1, 
                            const std::vector<D>& conditionalRates2, 
                            const std::vector<D>& conditionalRates3, 
                            const std::vector<D>& conditionalForwards, 
                            const std::vector<D>& conditionalDiscRatios
							)   
{    
    int numberRates = numberSteps;

    T product(auxData, numberSteps);

    MatrixConstFacade<D> rates1Matrix(&rates1[0],numberSteps,paths);
    MatrixConstFacade<D> rates2Matrix(&rates2[0],numberSteps,paths);
    MatrixConstFacade<D> rates3Matrix(&rates3[0],numberSteps,paths);
	MatrixConstFacade<D> rates1CondMatrix(&conditionalRates1[0],numberSteps,subPaths);
    MatrixConstFacade<D> rates2CondMatrix(&conditionalRates2[0],numberSteps,subPaths);
    MatrixConstFacade<D> rates3CondMatrix(&conditionalRates3[0],numberSteps,subPaths);
 
    CubeConstFacade<D> forwardsCube(&forwards[0],numberSteps, numberRates, paths);
    CubeConstFacade<D> discountsCube(&discRatios[0],numberSteps, numberRates+1, paths);
    CubeConstFacade<D> forwardsCondCube(&conditionalForwards[0],numberSteps, numberRates, subPaths);
    CubeConstFacade<D> discountsCondCube(&conditionalDiscRatios[0],numberSteps, numberRates+1, subPaths);



    genFlows1.resize(subPaths*numberSteps);
    genFlows2.resize(subPaths*numberSteps);

    
    for (int p=0; p < subPaths; ++p)
    { 
        product.newPath();

        bool done=false;

        for (int i =0; i < numberSteps; ++i)        
        {

            D flow1=0.0;        
            D flow2=0.0;

            if (!done)
            {   
				if (i <conditionalStep )
				{

	                D rate1 = rates1Matrix(i,distinguishedPath);
		            D rate2 = rates2Matrix(i,distinguishedPath);   
			        D rate3 = rates3Matrix(i,distinguishedPath); 

				    done = product.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwardsCube,discountsCube,distinguishedPath);
				}
				else
				{
					D rate1 = rates1CondMatrix(i,p);
		            D rate2 = rates2CondMatrix(i,p);   
			        D rate3 = rates3CondMatrix(i,p); 

				    done = product.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwardsCondCube,discountsCondCube,p);

				}
            }

            int location = i*paths+p;

            genFlows1[location] = flow1;
            genFlows2[location] = flow2;
        
        }               
    }
}

#endif
