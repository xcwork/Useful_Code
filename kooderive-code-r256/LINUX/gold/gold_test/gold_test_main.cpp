//
//
//                                              gold_test_main.cpp
//
//
//
// (c) Mark Joshi 2012, 2013
// This code is released under the GNU public licence version 3
/*
For test routines of gold routines not involving GPU code. i.e. test gold against gold
*/

#include <gold/gold_test/test_BSFormulas.h>
#include <gold/gold_test/svd_test.h>
#include <ctime>
#include <iostream>
#include <gold/Timers.h>

int main()
{
    int totalFails=0;
    bool verbose = true;

    int t = clock();
	Timer t1;
	t1.StartTimer();

    totalFails+=svdtest(verbose);
    totalFails+=pseudoRootTest( verbose);
    totalFails+=TestBSGreekFormulas(verbose);
    totalFails+=BSFormulasTest(verbose);  
    totalFails+=NormalFormulasTest(verbose);
    
    totalFails+=TestBivariateNormals( verbose);

    int taken = clock() - t;
	double highPerformanceTime = t1.timePassed();

    std::cout << "Testing complete. " << totalFails << " tests failed.\n";
    std::cout << "Total time taken " << taken*1E-3 << " seconds \n";
    std::cout << "Total time taken using high frequency timer " << highPerformanceTime<< " seconds \n";

    return 0;

}
