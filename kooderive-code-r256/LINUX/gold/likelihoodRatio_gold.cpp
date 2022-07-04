
//
//                              likelihoodRatio_gold.cpp
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3


#include <gold/likelihoodRatio_gold.h> 


void likelihoodRatiosBS_gold (const MatrixConstFacade<Realv>& inputPaths,
                              const MatrixConstFacade<Realv>& inputNormals,
                              int normalOffset,
                              const std::vector<Realv>& stepSizeSquareRoots,
                              Realv sigma,
                              Realv S0,
                              int paths,
                              int steps,
                              const std::vector<int>& stepNumbersForVegas,
                              bool cumulativeWeights,
                              std::vector<Realv>& LR_deltas,
                              std::vector<Realv>& LR_vegas
                              )
{
    if (inputPaths.columns() != steps)
        GenerateError("Input size mismatch for paths in likelihoodRatiosBS_gold");

    if (inputPaths.rows() < paths)
        GenerateError("Input size mismatch for paths in likelihoodRatiosBS_gold");

    if (inputNormals.rows() != steps)
        GenerateError("Input size mismatch for steps  in likelihoodRatiosBS_gold");

    LR_deltas.resize(paths);

    LR_vegas.resize(paths*stepNumbersForVegas.size());

    MatrixFacade<Realv> LR_vegas_matrix(&LR_vegas[0],paths, stepNumbersForVegas.size());

    Realv deltaW =1.0/(S0*sigma*stepSizeSquareRoots[0]);

    for (int p=0; p < paths; ++p)
    {
        Realv deltaLR = inputNormals(0,p+normalOffset)*deltaW;
        LR_deltas[p] = deltaLR;

        Realv vegaWeight =0.0;
        int m=0; 

        for (unsigned int v=0; v < stepNumbersForVegas.size(); ++v)
        {
            if (!cumulativeWeights)
                vegaWeight =0.0;
                

            while (m <= stepNumbersForVegas[v])
            {
                Realv z= inputNormals(m,p+normalOffset);
                vegaWeight += (z*z-1.0f)/sigma - z*stepSizeSquareRoots[m];
                ++m;
            }


            LR_vegas_matrix(p,v) = vegaWeight;


        }
    }


}
