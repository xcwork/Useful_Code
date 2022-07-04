//
//
//                                      pathwise_gold.cpp
//
//
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3
#include <gold/pathwise_gold.h>
  
void pathwiseBSweights_gold(const MatrixConstFacade<Realv>& inputPaths,
                              const MatrixConstFacade<Realv>& inputNormals,
                              int normalOffset,
                              const std::vector<Realv>& stepSizeSquareRoots,
                              Realv sigma,
                              Realv S0,
                              int paths,
                              int steps,
                              MatrixFacade<Realv>& w_deltas,
                              MatrixFacade<Realv>& w_vegas,
                              MatrixFacade<Realv>& omega_deltas,
                              MatrixFacade<Realv>& omega_vegas
                              )
{
    for (int p=0; p < paths; ++p)
    {
        Realv s_s =  inputPaths(p,0);
        omega_deltas(p,0) = s_s/S0;
        omega_vegas(p,0) = s_s*(stepSizeSquareRoots[0]*(inputNormals(0,normalOffset+p) -sigma*stepSizeSquareRoots[0]));
     
        w_deltas(p,0)=0.0;
        w_vegas(p,0)=0.0;

        for (int s=1; s < steps;++s)
        {
               omega_deltas(p,s) = inputPaths(p,s)/inputPaths(p,s-1);
               omega_vegas(p,s) = inputPaths(p,s)*(stepSizeSquareRoots[s]*(inputNormals(s,normalOffset+p) -sigma*stepSizeSquareRoots[s]));
               w_deltas(p,s)=0.0;
               w_vegas(p,s)=0.0;

        }

    }
}
