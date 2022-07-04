#include "LMM_Bermudan_gold.h"
#include "BSGreeks.h"
#include "Max_call_bermudan.h"
#include "Bput.h"
#include "Heston_cf.h"
#include <ctime>
#include <iostream>

int BSmain()
{
    return BSGreeksDDmain();
//    return BSGreeksMain();
}

int Bermudanmain()
{
  //  int biasPaths = 8192/64;
//    int innerPaths =32;
 
  //  return  Max_call_bermudan();

    return BermudanPut();

    int numberCases = 1;

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
                1024,  //  int pathsPerSecondPassBatch,
                1,//  int numberSecondPassBatches
                2, // choice
                2048, // paths for outer sims
                223233,
                innerPaths[i], // pahts for inner sims
                0,
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

int main()
{
    OptionCosVGExample(
                    //, double beta0 // Imaginary u
                    //, 
                    25000,// numberPoints
                    50, // cases to run
                    1.0/480,//  -0.1, //  double theta, 
                   20.0/3.0 ,//  2, // double nu, 
                      0.5, //double T, 
                    0.25/sqrt(15.0),// 0.25, // double sigma, 
                    1.0,// 1.0,
                   1.1, //1.0,
                   0.1,
                   0.0
                    );
    /*
    OptionCosHestonExample(
                    //, double beta0 // Imaginary u
                    //, 
                    5000, 
                    4.0, 0.25,1.0, -0.5, 0.01, 0.01, 
                    100.0,
                    100.0,
                    0.01,
                    0.02
                    );*/
    // DigitalOptionBSExample();
     /*
    HestonCFDumper(0 // alpha0 // Real u
                    , 0.0// beta0 // Imaginary u
                    , 1000 // int numberPoints
                    , 5 //double alphaStep
                    , 0.0// double betaStep
                    , 0.8//  double a, 
                    , 0.9 //double b
                    , 0.7 //double c
                    , 0.25 //double rho, 
                    , 0.01 // double t, 
                    , 0.01 //double v0, 
                    , 0 //double x0
                    );
                    */
    return 0;
}
