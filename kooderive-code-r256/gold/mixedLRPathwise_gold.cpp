//
//
//                                      mixedLRPathwise_gold.cpp
//
//
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3
#include <gold/mixedLRPathwise_gold.h>

void mixedLRPathwiseDeltas_gold(const MatrixConstFacade<Realv>& discountedCashFlows,
                                const MatrixConstFacade<Realv>& discountedCashFlowStateDerivatives,
                                const MatrixConstFacade<int>& cashFlowGenerationIndices,
                                const std::vector<int>& terminationSteps,
                                int paths, 
                                const MatrixConstFacade<Realv>& deltas_omegas,
                                const MatrixConstFacade<Realv>& deltas_w,
                                std::vector<Realv>& pathDeltaEstimates  
                              )
{
    for (int p=0; p < paths;++p)
    {
        Realv totalDelta=0.0f;
        int cashFlowIndex=0;
        pathDeltaEstimates[p] =0.0f;

        Realv runningW =0.0f;
        Realv runningOmega = 1.0f;
        for (int s=0; s < terminationSteps[p] && cashFlowIndex< cashFlowGenerationIndices.columns(); ++s)
        {
            runningW += runningOmega*deltas_w(p,s);
            runningOmega *= deltas_omegas(p,s);

            if (s == cashFlowGenerationIndices(p,cashFlowIndex))
            {
                Realv flow = discountedCashFlows(p,cashFlowIndex);
                Realv flowDerivative = discountedCashFlowStateDerivatives(p,cashFlowIndex);
                pathDeltaEstimates[p] += runningW*flow;
                pathDeltaEstimates[p] += runningOmega*discountedCashFlowStateDerivatives(p,cashFlowIndex);

                ++cashFlowIndex;

            }

        }

    }
}
void mixedLRPathwiseDeltasVega_gold(const MatrixConstFacade<Realv>& discountedCashFlows,
                                    const MatrixConstFacade<Realv>& discountedCashFlowStateDerivatives,
                                    const MatrixConstFacade<int>& cashFlowGenerationIndices,
                                    const std::vector<int>& terminationSteps,
                                    int maxSteps,
                                    int paths, 
                                    const MatrixConstFacade<Realv>& deltas_omegas,
                                    const MatrixConstFacade<Realv>& deltas_w,
                                    const MatrixConstFacade<Realv>& vegas_omegas,
                                    const MatrixConstFacade<Realv>& vegas_w,                                  
                                    std::vector<Realv>& pathDeltaEstimates , 
                                    std::vector<Realv>& pathVegaEstimates  
                              )
{
    std::vector<Realv> alongPathDeltas(maxSteps+1); 
    std::vector<Realv> futureFlows_vec(maxSteps);

    for (int p=0; p < paths;++p)
    {
        Realv totalDelta=0.0f;

        int cashFlowIndex=0;

        pathDeltaEstimates[p] =0.0;

        std::fill(alongPathDeltas.begin(),alongPathDeltas.end(),0.0);
        std::fill(futureFlows_vec.begin(),futureFlows_vec.end(),0.0);

        int termStep = terminationSteps[p];

        if (termStep > cashFlowGenerationIndices(p,0))
        {
            int lastIndex =0;
            while (lastIndex <cashFlowGenerationIndices.columns()-1 && cashFlowGenerationIndices(p,lastIndex+1) < termStep  )
                ++lastIndex;

            int finalStep = cashFlowGenerationIndices(p,lastIndex);

            alongPathDeltas[finalStep+1] =  discountedCashFlowStateDerivatives(p,lastIndex);
            Realv futureFlows =  discountedCashFlows(p,lastIndex);

            int prevIndex = lastIndex-1;
            int prevRelevantStep = -1;

            if (prevIndex >= 0)
                prevRelevantStep = cashFlowGenerationIndices(p,prevIndex);


            for (int s = finalStep; s >=0; --s)
            {
                

                if ( s== prevRelevantStep) // add on cash flow
                {
                    futureFlows+=discountedCashFlows(p,prevIndex);
                    alongPathDeltas[s+1] += discountedCashFlowStateDerivatives(p,prevIndex);
                    --prevIndex;
                    prevRelevantStep = -1;
                    if (prevIndex >= 0)
                        prevRelevantStep = cashFlowGenerationIndices(p,prevIndex);

                }

                alongPathDeltas[s] = alongPathDeltas[s+1] *deltas_omegas(p,s);
        
                futureFlows_vec[s] = futureFlows; 
                alongPathDeltas[s] += deltas_w(p,s) * futureFlows;
          
            }

            pathDeltaEstimates[p] = alongPathDeltas[0];

            // we still need to do the vega
            // we'll do it in forward mode

            Realv thisVega =0.0;
            Realv runningVegaW=0.0;
       
            
            

            for (int s=0; s <= finalStep; ++s)
            {
               
                Realv thisOmega = vegas_omegas(p,s);
                thisVega += thisOmega*alongPathDeltas[s+1];
                Realv thisWeight = vegas_w(p,s);
                thisVega += thisWeight*futureFlows_vec[s];

               
            }

            pathVegaEstimates[p]=thisVega;


        }
        else
        { // no cash-flows before termination on this path
            pathVegaEstimates[p]=0.0;            
        }

     

    }
}
