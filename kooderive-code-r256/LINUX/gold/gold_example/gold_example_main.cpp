#include "LMM_Bermudan_gold.h"
#include "BSGreeks.h"

#include <ctime>
#include <iostream>
int main()
{
  //  int biasPaths = 8192/64;
//    int innerPaths =32;
 
    int numberCases = 9;

    int biasPaths[]={0,64,8192,0,64,8192,0,64,8192};
    int innerPaths[]={256,256,256,512,512,512,1024,1024,1024};



        for (int i=0; i < numberCases; ++i)
        {
            int t0=clock();
            BermudanExample(false, //bool useLogBasis, 
                true,     //bool useCrossTerms,
                1024,    //int pathsPerBatch, 
                1,   // int numberOfBatches,
                5,   // int numberOfExtraRegressions, 
                1,   // int duplicate,
                false,  //  bool normalise,
                65536,  //  int pathsPerSecondPassBatch,
                1,//  int numberSecondPassBatches
                2, // choice
                2048, // paths for outer sims
                223233,
                innerPaths[i], // pahts for inner sims
                2,
                biasPaths[i], // paths for bias removal
                2343434,
                innerPaths[i] // minpaths to use
                ); 
            int t1=clock();

            double timeTaken = (t1-t0+0.00)/CLOCKS_PER_SEC;
            std::cout << " time taken : " << timeTaken << ",biasPaths,"<<biasPaths[i] << ",innerPaths," << innerPaths[i] << ",," << "\n";
        }

    return 0;
}
