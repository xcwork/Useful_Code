// (c) Mark Joshi 2009, 2013,2014
// This code is released under the GNU public licence version 3
#include "Sobol_int_test.h"
#include "mainsCuda.h"
#include "Asian_Test.h"
#include "Bridge_Test.h"
#include "Correlation_test.h"
#include "constant_memory_test.h"
#include "drift_add_test.h"
#include "MultiD_Path_Gen_BS_Test.h"
#include "LMM_evolver_test.h"
#include "correlate_drift_test.h"
#include "Sobol_test.h"


#include "scramble_test.h"
#include "stream_test.h"
#include "LS_test.h"
#include "LSmulti_test.h"
#include "outer_test.h"
#include "cula_test.h"
#include "matrix_test.h"
#include <cutil_inline.h>
#include <iostream>


int mainTest()
{


    int fails =0;

    int numberGPUs;
    cutilSafeCall( cudaGetDeviceCount(&numberGPUs) );

    for (int i=0; i < numberGPUs; ++i)
    {
        DeviceChooserSpecific chooser(i);
       fails+= MatrixTransposeTest( chooser);
      
         fails += Test_Multi_LS_etc(true,chooser);

       
     fails +=OuterTest(chooser); 

        fails  += MultiDBridgeTestRoutine(true,true,true,chooser); 

        fails += scrambleTestRoutine(true,chooser);
       fails +=  SobolTestRoutineForInts(true,chooser); 

     fails += MultiDBSTestRoutine(true,true,chooser);

        fails += MultiDBridgeTestOrderingRoutine(true,false,chooser);
        fails += BridgeTestRoutine(true,true,chooser);
        
        fails += PointwiseTest(chooser);
        fails += LMMPCSKTestRoutine(true,1,2,64,64,16,chooser);

        fails += LMMPCTestRoutine(true,true,chooser);
     

        //  fails +=  Test_cula_solve(true);
          //     fails+= StreamTestRoutine(true,chooser);





        fails += LMMLogEulerTestRoutine(true,true,chooser);

        fails  += CorrelationTestRoutine(true,true,chooser);

        fails  += CorrelationTestRoutine(true,false,chooser);

        fails += DriftAddTestRoutine(true,false,chooser);
        fails += DriftAddTestRoutine(true,true,chooser);

        fails += CorrelateDriftTestRoutine(true,chooser);

        fails  += MultiDBridgeTestRoutine(true,true,true,chooser);
        fails+= Brownian_bridge_test_routine(true,false,chooser);

        fails += AsianTestRoutine(true,chooser);

        fails += Matrix_solving_test_routines(true,chooser);
   
    }

    /*
    */
    std::cout << "\n\nTesting complete. Number of test failures " << fails << "\n";


    return fails;

}


int main()
{
    mainTest();

    return  0;
}


