//
//											max_call_bermudan.cpp
//
//

// (c) Mark Joshi 2015
// released under GPL v 3.0
#include "Max_call_bermudan.h"
#include <gold/gold_test/MultiDBS_gold_test.h>
#include <gold/MultiD_BS_evolver_classes_gold.h>
#include <gold/volstructs_gold.h>
#include <gold/MonteCarloStatistics_concrete_gold.h>
#include <gold/BSFormulas_gold.h>
#include <gold/EarlyExercisableMultiEquityPayoff_gold.h>
#include <gold/BasisVariableExtractionMultiEquity_gold.h>
#include <gold/volstructs_gold.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <gold/Regression_Selector_concrete_gold.h>
#include <gold/LS_basis_examples_gold.h>
#include <gold/ExerciseIndices_gold.h>
#include <gold/LMM_evolver_classes_gold.h>
#include <gold/LS_Basis_gold.h>
#include <gold/cashFlowGeneration_product_gold.h>
#include <gold/cashFlowDiscounting_gold.h>
#include <gold/CashFlowAggregation_gold.h>
#include <gold/MultiLS_regression.h>
#include <gold/cashFlowGeneration_earlyEx_product.h>
#include <gold/early_exercise_value_generator_gold.h>
#include <float.h>
#include <gold/Mersenne_gold.h>
#include <gold/LMM_evolver_partial_gold.h>
#include <gold/Andersen_Broadie_gold.h>
#include <gold\MultiD_Path_Gen_Partial_gold.h>

namespace
{
    class addPower
    {
    public:
        addPower(int k) : k_(k)
        {
        }

        double operator()(double sum, double y)
        {
            double v=y;
            for (int l=1; l<k_; ++l)
                v*=y;

            return sum +v;
        }


    private:
        int k_;
    };
}

int Max_call_bermudan()
{

    int paths =16384*16;

    int indicativePaths =paths; 
    int pathsPerSecondPassBatch=paths;
    int numberSecondPassBatches =2;//9;
    int duplicate =1;

    int numberStocks=5;
    int factors=5;
    int stepsForEvolution=9; 
    int powerOfTwoForVariates=4;

    double sigma=0.2;
    double rho=0;
    double T=3.0;
    double r =0.05;
    double d =0.1;
    double S0 =110 ;
    double strike=100;

    // regresion parameters
    bool useCrossTerms=true;
    bool normalise = false;
    int numberOfExtraRegressions=5;//5;
    int lowerPathCutoff = 5000;
    double lowerFrac = 0.25;
    double upperFrac=0.26;
    double initialSDguess = 2.5;


    // parameters for outer sim for upper bound
    int outerPaths = 8192;
    int upperSeed = 23232;
    // parameters for inner sim for upper bound
    //  int subsimPaths =64;
    std::vector<int> subsimPaths_vec(1);

    subsimPaths_vec[0]=16384;
    /*
    subsimPaths_vec[1]=64;
    subsimPaths_vec[2]=128;
    subsimPaths_vec[3]=256;
    subsimPaths_vec[4]=512;
    subsimPaths_vec[5]=1024;
    */

    int minPathsForUpper=32;


    // parameters for MC bias estimation
    int biasseed=3234;

    std::vector<int> extraPaths_vec;
    //    extraPaths_vec[0]=10000;
    //    extraPaths_vec[1]=20000;
    //   extraPaths_vec[2]=40000;
    //   extraPaths_vec[3]=80000;
    //   extraPaths_vec[4]=160000;
    int powerToUse = 1;

    bool dumpBiases=false;
    bool dump=false;
    // int basePathsForRework = 64;

    std::vector<int> pathsForGaussianEstimation_vec(1); //(2);

    pathsForGaussianEstimation_vec[0]=16;
    //   pathsForGaussianEstimation_vec[1]=8192;


    return Max_call_bermudan(paths,
        pathsPerSecondPassBatch,
        numberSecondPassBatches ,
        duplicate,
        numberStocks,
        factors,
        stepsForEvolution, 
        powerOfTwoForVariates,
        sigma,
        rho,
        T,
        r,
        d,
        S0,
        strike,
        // regresion parameters
        useCrossTerms,
        normalise ,
        numberOfExtraRegressions,
        lowerPathCutoff ,
        lowerFrac ,
        upperFrac,
        initialSDguess ,
        outerPaths , // parameters for outer sim for upper bound
        upperSeed ,
        subsimPaths_vec,     // parameters for inner sim for upper bound
        minPathsForUpper,
        pathsForGaussianEstimation_vec,  // parameters for MC bias estimation
        biasseed,
        extraPaths_vec,
        powerToUse ,
        dumpBiases, 
        dump
        );

}

int Max_call_bermudan(int paths,
                      int pathsPerSecondPassBatch,
                      int numberSecondPassBatches ,
                      int duplicate,
                      int numberStocks,
                      int factors,
                      int stepsForEvolution, 
                      int powerOfTwoForVariates,
                      double sigma,
                      double rho,
                      double T,
                      double r,
                      double d,
                      double S0,
                      double strike,
                      // regresion parameters
                      bool useCrossTerms,
                      bool normalise ,
                      int numberOfExtraRegressions,
                      int lowerPathCutoff ,
                      double lowerFrac ,
                      double upperFrac,
                      double initialSDguess ,
                      int outerPaths , // parameters for outer sim for upper bound
                      int upperSeed ,
                      const std::vector<int>& subsimPaths_vec,
                      //  int subsimPaths,     // parameters for inner sim for upper bound
                      int minPathsForUpper,
                      std::vector<int> pathsForGaussianEstimation_vec,  // parameters for MC bias estimation
                      int biasseed,
                      const std::vector<int>& extraPaths_vec,
                      int powerToUse ,
                      bool dumpBiases, 
                      bool dump
                      )
{
std::cout << "paths,"<<paths<<
             ",pathsPerSecondPassBatch,"<<pathsPerSecondPassBatch<<
             ",numberSecondPassBatches," <<numberSecondPassBatches <<
             ",duplicate,"<<duplicate<<
             ",numberStocks,"<<numberStocks<<
             ",factors,"<<factors<<
             ",stepsForEvolution,"<<stepsForEvolution<< 
             ",powerOfTwoForVariates,"<<powerOfTwoForVariates<<
             ",sigma,"<<sigma<<
             ",rho,"<< rho<<
             ",T,"<<T<<
             ",r,"<<r<<
             ",d,"<<d<<
             ",S0,"<<S0<<
             ",strike,"<<strike<<
             ",useCrossTerms,"<<useCrossTerms<<
             ",normalise,"<<normalise <<
             ",numberOfExtraRegressions,"<<numberOfExtraRegressions<<
             ",lowerPathCutoff ,"<<lowerPathCutoff<<
             ",lowerFrac ,"<<lowerFrac<<
             ",upperFrac,"<<upperFrac<<
             ",initialSDguess ,"<<initialSDguess<<
             ",outerPaths ,"<< outerPaths<<
             ",upperSeed ,"<<upperSeed<<
             ",minPathsForUpper,"<<minPathsForUpper<<
             ",biasseed,"<<biasseed<<
             ",powerToUse ,"<<powerToUse<<
             ",dumpBiases, "<<dumpBiases<<
             ",dump,"<<dump
             ;

    double ticksize = 1.0/CLOCKS_PER_SEC;
    int indicativePaths =paths; 

    std::vector<double> initial_stocks_vec(numberStocks);     
    for (int i=0; i < numberStocks; ++i)
        initial_stocks_vec[i] =S0;

    std::vector<double> evolutionTimes_vec(stepsForEvolution);
    Matrix_gold<double> fixed_drifts_mat(stepsForEvolution,numberStocks,0.0);

    for (int i=0; i < stepsForEvolution; ++i)
    {
        evolutionTimes_vec[i] = (i+1)*T/stepsForEvolution;

        for (int stock=0; stock<numberStocks; ++ stock)
        {
            fixed_drifts_mat(i,stock) = (r-d)*T/stepsForEvolution;
        }

    }





    Cube_gold<double> pseudo_roots_cube(FlatVolConstCorrelationPseudoRoots( sigma, 
        evolutionTimes_vec,
        numberStocks,
        factors,
        rho));




    int pathOffset=0;
    std::vector<unsigned int> scrambler_vec(static_cast<int>(pow(2,powerOfTwoForVariates))*factors,0);

    std::vector<double> evolved_stocks_vec;     
    std::vector<double> evolved_log_stocks_vec;     


    int t0=clock();


    MultiDBS_PathGen_Sobol_gold<double> pathGenerator( indicativePaths, 
        numberStocks, 
        factors, 
        stepsForEvolution, 
        powerOfTwoForVariates,
        pseudo_roots_cube.getDataVector(),
        fixed_drifts_mat.getDataVector(),
        initial_stocks_vec
        );

    pathGenerator.getPaths( paths, pathOffset, scrambler_vec,  
        evolved_stocks_vec,
        evolved_log_stocks_vec);


    CubeConstFacade<double> evolved_stocks_cube(evolved_stocks_vec,stepsForEvolution,numberStocks,paths);




    MaxCall thePayOff(numberStocks,strike);

    int pathOffsetForOutput=0;

    Matrix_gold<double> payoffs(stepsForEvolution,paths+pathOffsetForOutput,0.0);



    GenerateMultiDEquityPayoffs<MaxCall>(evolved_stocks_cube, thePayOff, pathOffsetForOutput,0,0, payoffs.Facade());

    int numberStockVariables = std::max(numberStocks-2,1);

    std::vector<double> vols(3,sigma);
    std::vector<double> expiries(3);
    std::vector<double> rhos(3,rho);
    std::vector<double> strikes_basis_vec(3,strike);

    expiries[0]  = T/stepsForEvolution;
    expiries[1]  = (T*(stepsForEvolution-1.0))/stepsForEvolution;
    expiries[2]  =( expiries[0]+expiries[1])/2.0;

    std::vector<double> r_vec(3,r);
    std::vector<double> d_vec(3,d);


    MaxCallVariables theVariables(numberStocks,numberStockVariables,vols,expiries,rhos,strikes_basis_vec,r_vec,d_vec);

    int numberVariables = theVariables.numberVariables();

    Cube_gold<double> basisVariables(stepsForEvolution,numberVariables,paths+pathOffsetForOutput,0.0);

    GenerateMultiDEquityBasisVariables<MaxCallVariables>(evolved_stocks_cube, 
        theVariables, 
        pathOffsetForOutput,
        0,
        0,
        basisVariables.Facade());


    // now run through the least squares code

    double multiplier = numberOfExtraRegressions > 0 ? 1-pow(0.9,1.0/numberOfExtraRegressions) : 0;
    RegressionSelector_gold_Fraction regressionSelector(lowerFrac, upperFrac, initialSDguess,  multiplier);
    int maxRegressionDepth = numberOfExtraRegressions+1;

    int numberOfExerciseDates = stepsForEvolution;

    std::vector<int> exerciseIndices_vec; // the indices of the exercise times amongst the evolution times
    std::vector<int> exerciseIndicators_vec(stepsForEvolution,0);
    for (int i=0; i < stepsForEvolution;++i) // 
    {
        exerciseIndicators_vec[i] =1;
    }


    GenerateIndices(exerciseIndicators_vec, exerciseIndices_vec);
    int maxBasisVariables=theVariables.numberVariables();

    int numberBasisFunctions =useCrossTerms ? quadraticPolynomialCrossGenerator(maxBasisVariables).numberDataPoints()
        : quadraticPolynomialGenerator(maxBasisVariables).numberDataPoints();

    std::vector<double> productsM_cube_vec;
    std::vector<double> targetsM_mat_vec;

    std::vector<double> ls_coefficients_multi_vec(maxRegressionDepth*exerciseIndices_vec.size()*numberBasisFunctions);
    CubeFacade<double> regression_coefficients_cube(ls_coefficients_multi_vec,numberOfExerciseDates,
        maxRegressionDepth,numberBasisFunctions);

    Matrix_gold<double> lowerCuts_mat(numberOfExerciseDates,maxRegressionDepth,0.0);
    Matrix_gold<double> upperCuts_mat(numberOfExerciseDates,maxRegressionDepth,0.0);

    Cube_gold<double> means_cube_gold(numberOfExerciseDates,numberOfExtraRegressions+1, 
        maxBasisVariables,0.0);
    Cube_gold<double> sds_cube_gold(numberOfExerciseDates,numberOfExtraRegressions+1,maxBasisVariables,0.0);

    std::vector<int> basisVariablesPerIndex(numberOfExerciseDates);

    for (int s=0; s < numberOfExerciseDates; ++s)
        basisVariablesPerIndex[s]=  theVariables.numberVariables();

    std::vector<double> some_numeraireValues_vec(exerciseIndices_vec.size()*paths);

    MatrixFacade<double> some_numeraireValues_matrix(&some_numeraireValues_vec[0],numberOfExerciseDates,paths);


    for (int s=0; s < stepsForEvolution; ++s)
    {
        double bankaccount = exp(s*r*T/stepsForEvolution);
        for (int p=0; p < paths; ++p)
            some_numeraireValues_matrix(s,p) = bankaccount;
    }


    Matrix_gold<double> aggregatedFlows_matrix(stepsForEvolution,paths,0.0);

    double lsestM = generateRegressionCoefficientsViaLSMultiQuadratic_gold(numberOfExerciseDates,
        productsM_cube_vec,
        targetsM_mat_vec,
        regression_coefficients_cube,
        lowerCuts_mat.Facade(),
        upperCuts_mat.Facade(),
        means_cube_gold.Facade(),
        sds_cube_gold.Facade(),
        normalise,
        basisVariables,
        basisVariablesPerIndex,
        maxBasisVariables,
        aggregatedFlows_matrix, // deflated to current exercise time
        payoffs, // deflated to current exercise time
        some_numeraireValues_matrix,
        //         deflation_locations_vec,
        paths,
        numberOfExtraRegressions,
        lowerPathCutoff,
        regressionSelector,
        useCrossTerms);

    double initialDf = exp(-r*T/stepsForEvolution);

    double firstPassValue = initialDf*lsestM;

    std::cout << "least squares multiple regression ran!, First pass value, " << firstPassValue <<"\n";

    int t1=clock();


    /// NOW SECOND PASS


    std::vector<double> AndersenShifts_vec(stepsForEvolution);

    std::vector<quadraticPolynomialCrossGenerator> functionCrossProducers;

    for (size_t i=0; i < exerciseIndices_vec.size(); ++i)
        functionCrossProducers.push_back(quadraticPolynomialCrossGenerator(basisVariablesPerIndex[i]));

    std::vector<quadraticPolynomialGenerator> functionProducers;

    for (size_t i=0; i < exerciseIndices_vec.size(); ++i)
        functionProducers.push_back(quadraticPolynomialGenerator(basisVariablesPerIndex[i]));

    LSAMultiExerciseStrategy<quadraticPolynomialCrossGenerator,double> exMCrossStrategy(static_cast<int>(exerciseIndices_vec.size()),
        basisVariablesPerIndex,
        AndersenShifts_vec,
        ls_coefficients_multi_vec, 
        maxRegressionDepth,
        lowerCuts_mat.ConstFacade(),
        upperCuts_mat.ConstFacade(),
        means_cube_gold.ConstFacade(),
        sds_cube_gold.ConstFacade(),
        functionCrossProducers,
        maxBasisVariables
        );

    LSAMultiExerciseStrategy<quadraticPolynomialGenerator,double> exMStrategy(static_cast<int>(exerciseIndices_vec.size()),
        basisVariablesPerIndex,
        AndersenShifts_vec,
        ls_coefficients_multi_vec, 
        maxRegressionDepth,
        lowerCuts_mat.ConstFacade(),
        upperCuts_mat.ConstFacade(),
        means_cube_gold.ConstFacade(),
        sds_cube_gold.ConstFacade(),
        functionProducers,
        maxBasisVariables
        );


    std::vector<double> floatingInputs(1);
    floatingInputs[0] = strike;
    std::vector<int> integerInputs;

    earlyExerciseCallGold<double> exerciseValue(floatingInputs,
        integerInputs, stepsForEvolution); 

    std::vector<double> flow1_multi_vec(pathsPerSecondPassBatch*stepsForEvolution);


    std::vector<double> flow2_multi_vec(pathsPerSecondPassBatch*stepsForEvolution);

    MatrixConstFacade<double> flow1_multi_mat(flow1_multi_vec,stepsForEvolution,pathsPerSecondPassBatch);
    MatrixConstFacade<double> flow2_multi_mat(flow2_multi_vec,stepsForEvolution,pathsPerSecondPassBatch);


    std::vector<double> nullAccruals_vec(stepsForEvolution,0.0);
    Swap_gold<double> nullProduct( stepsForEvolution, 0.0, 0.0, nullAccruals_vec,nullAccruals_vec);

    std::vector<bool> exerciseIndicatorsbool_vec(stepsForEvolution,true);


    std::vector<double>   rate1_vec(pathsPerSecondPassBatch*stepsForEvolution);
    std::vector<double> rate2_vec(pathsPerSecondPassBatch*stepsForEvolution);
    std::vector<double>  rate3_vec(pathsPerSecondPassBatch*stepsForEvolution);
    std::vector<double>  rate1_cf_vec(pathsPerSecondPassBatch*stepsForEvolution);
    std::vector<double>  rate2_cf_vec(pathsPerSecondPassBatch*stepsForEvolution);
    std::vector<double>  rate3_cf_vec(pathsPerSecondPassBatch*stepsForEvolution);

    evolved_stocks_vec.resize(pathsPerSecondPassBatch*stepsForEvolution*numberStocks);    
    evolved_log_stocks_vec.resize(pathsPerSecondPassBatch*stepsForEvolution*numberStocks);    

    // dummies for putting stuff throught IRD code

    //             std::vector<double>  evolved_rates_vec; //(pathsPerSecondPassBatch*stepsForEvolution*number_rates);       
    std::vector<double>  discounts_vec(pathsPerSecondPassBatch*stepsForEvolution*(stepsForEvolution+1));

    Cube_gold<double> batch_basis_variables_cube(
        numberOfExerciseDates,
        maxBasisVariables,pathsPerSecondPassBatch,0.0);

    std::vector<std::string> names(1);


    names[0] = "max call price";

    MonteCarloStatisticsSimple statsBig(1,names);

    std::vector<double> data_vec(1);
    std::vector<double> data_vec_big(1);

    std::vector<double> pathsPayoffs(pathsPerSecondPassBatch);
    std::vector<int> pathsPayoffIndices(pathsPerSecondPassBatch);

    std::vector<double> discounts_const_vec(stepsForEvolution);
    for (int s=0; s < stepsForEvolution; ++s)
    {
        double df = initialDf/some_numeraireValues_matrix(s,0);
        discounts_const_vec[s]=df;

    }

    MersenneTwisterUniformRng rngSecondPass(213232UL);


    for (int secondBatch=0; secondBatch < numberSecondPassBatches; ++secondBatch)
    {
        int pathOffset = duplicate*paths+secondBatch*pathsPerSecondPassBatch;

        int pathOffsetForOutput=0;

        // generate equity paths

        rngSecondPass.getInts( scrambler_vec.begin(),scrambler_vec.end());

        pathGenerator.getPaths( pathsPerSecondPassBatch, pathOffset, scrambler_vec,  
            evolved_stocks_vec,
            evolved_log_stocks_vec);
        // generates basis variables


        GenerateMultiDEquityBasisVariables<MaxCallVariables>(evolved_stocks_cube, 
            theVariables, 
            pathOffsetForOutput,
            0,
            0,
            basisVariables.Facade());

        // generate payoffs

        GenerateMultiDEquityPayoffs<MaxCall>(evolved_stocks_cube, thePayOff, pathOffsetForOutput,0,0, payoffs.Facade());



        // generate rates
        for (int s=0; s < stepsForEvolution; ++s)
        {

            for (int p=0; p < pathsPerSecondPassBatch; ++p)
            {
                rate1_cf_vec[p+s*paths] = basisVariables.Facade()(s,0,p);
                rate2_cf_vec[p+s*paths] = basisVariables.Facade()(s,1,p);
                rate3_cf_vec[p+s*paths] = basisVariables.Facade()(s,2,p);

            }
        }

        if (useCrossTerms)
        {

            cashFlowGeneratorEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialCrossGenerator,double>, double>(
                pathsPayoffs, // output  
                pathsPayoffIndices, // output for discounting
                payoffs.ConstFacade(),
                exMCrossStrategy,
                pathsPerSecondPassBatch, 
                stepsForEvolution,
                0, //int stepsToSkip,
                exerciseIndicatorsbool_vec,
                rate1_cf_vec, 
                rate2_cf_vec, 
                rate3_cf_vec,  
                basisVariables.ConstFacade(),
                evolved_stocks_vec ,
                discounts_vec )  ; 



        }
        else
        {


            cashFlowGeneratorEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialGenerator,double>, double>(
                pathsPayoffs, // output  
                pathsPayoffIndices, // output for discounting
                payoffs.ConstFacade(),
                exMStrategy,
                pathsPerSecondPassBatch, 
                stepsForEvolution,
                0, //int stepsToSkip,
                exerciseIndicatorsbool_vec,
                rate1_cf_vec, 
                rate2_cf_vec, 
                rate3_cf_vec,  
                basisVariables.ConstFacade(),
                evolved_stocks_vec ,
                discounts_vec )  ; 

        }
        MonteCarloStatisticsSimple stats(1,names);

        for (int p=0; p < pathsPerSecondPassBatch; ++p)
        {
            double pathValue = pathsPayoffs[p]*discounts_const_vec[pathsPayoffIndices[p]];
            data_vec[0] = pathValue;
            stats.AddDataVector(data_vec);
        }
        std::vector<std::vector<double> > resSmall(stats.GetStatistics());

        data_vec_big[0] = resSmall[0][0];
        statsBig.AddDataVector(data_vec_big);
        std::cout << " ,"<< data_vec_big[0] << ", ";

    }

    std::vector<std::vector<double> > res(statsBig.GetStatistics());

    double secondPassValue = res[0][0];

    std::cout << "\nSecond pass mean value:, " << secondPassValue <<"\n";

    double se = res[1][0];

    std::cout << " standard error:, " << se << "\n";
    int t2=clock();

    // we now need an outer set of paths for the upper bounder, 

    Cube_gold<double> outerPaths_cube(stepsForEvolution,numberStocks,outerPaths);

    MultiDBS_PathGen_Partial_gold<double> generator( 
        numberStocks, 
        factors, 
        stepsForEvolution, 
        pseudo_roots_cube.getDataVector(),
        fixed_drifts_mat.getDataVector()
        );

    Cube_gold<double> variates(stepsForEvolution,factors,outerPaths);
    //    Cube_gold<double> variates_inner(stepsForEvolution,factors,subsimPaths);

    MersenneTwisterUniformRng rng(upperSeed);
    rng.populateCubeWithNormals<CubeFacade<double>,double>(variates.Facade());

    generator.getPaths(outerPaths,
        0,
        0,
        initial_stocks_vec,
        variates.ConstFacade(),
        0, // specifies  how to associate normals in cube with steps in simulation
        outerPaths_cube.Facade());

    Matrix_gold<double> payoffs_outer_mat(stepsForEvolution,outerPaths,-1000.0);
    Matrix_gold<double> payoffs_discounted_outer_mat(stepsForEvolution,outerPaths,-1000.0);

    GenerateMultiDEquityPayoffs<MaxCall>(outerPaths_cube, thePayOff, pathOffsetForOutput,0,0, payoffs_outer_mat.Facade());

    for (int s=0; s < stepsForEvolution; ++s)
    {
        double df = exp(-r*(s+1.0)*T/stepsForEvolution);
        for (int p=0; p < outerPaths; ++p)
            payoffs_discounted_outer_mat(s,p) = payoffs_outer_mat(s,p)*df;
    }  

    Cube_gold<double> basisVariables_outer(stepsForEvolution,numberVariables,outerPaths,0.0);


    GenerateMultiDEquityBasisVariables<MaxCallVariables>(outerPaths_cube, 
        theVariables, 
        pathOffsetForOutput,
        0,
        0,
        basisVariables_outer.Facade());


    Matrix_gold<int> exercise_occurences_outer_mat(stepsForEvolution,outerPaths,0);

    if (useCrossTerms)
    {
        exerciseAllDeterminerEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialCrossGenerator,double>, double>(  
            exercise_occurences_outer_mat.Facade(), // 0 = false, 1 = true
            payoffs_outer_mat.ConstFacade(),
            exMCrossStrategy,
            outerPaths, 
            stepsForEvolution,
            0, //steps to skip
            exerciseIndicatorsbool_vec,
            rate1_cf_vec, 
            rate2_cf_vec, 
            rate3_cf_vec,  
            basisVariables_outer.ConstFacade(),
            evolved_stocks_vec ,
            discounts_vec )   ;
    }
    else
    {
        exerciseAllDeterminerEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialGenerator,double>, double>(  
            exercise_occurences_outer_mat.Facade(), // 0 = false, 1 = true
            payoffs_outer_mat.ConstFacade(),
            exMStrategy,
            outerPaths, 
            stepsForEvolution,
            0, //steps to skip
            exerciseIndicatorsbool_vec,
            rate1_cf_vec, 
            rate2_cf_vec, 
            rate3_cf_vec,  
            basisVariables_outer.ConstFacade(),
            evolved_stocks_vec ,
            discounts_vec )   ;
    }

    // we now have the outer paths and their exercise points
    // we need to get the inner paths expectations
    // we store all the pathwise values since it makes easier to see trends

    std::vector<double> current_stocks_vec(numberStocks);
    std::vector<double> twoPtEst, threePtEst;
    Matrix_gold<double> mcEst;
    int subsimPaths;
    std::vector<double> variates_inner_vec;

    Matrix_gold<double> means_mat(outerPaths,stepsForEvolution,0.0);
    Matrix_gold<double> sds_mat(outerPaths,stepsForEvolution,0.0);

    double meanAB,seAB;
    std::vector<double> workspace_vec;

    for (size_t subsimCounter=0; subsimCounter < subsimPaths_vec.size(); ++subsimCounter)
    {
        subsimPaths = subsimPaths_vec[subsimCounter];
        bool lastSubsimSize_bool = (subsimCounter+1 == subsimPaths_vec.size());

        std::cout << " subsim paths, " << subsimPaths << "\n";

        variates_inner_vec.resize(stepsForEvolution*factors*subsimPaths);


        std::vector<double> basisVariables_inner_vec(stepsForEvolution*numberVariables*subsimPaths,0.0);
        //  Cube_gold<double> basisVariables_inner(stepsForEvolution,numberVariables,subsimPaths,0.0);

        std::vector<double> paths_inner_vec(stepsForEvolution*numberStocks*subsimPaths,0.0);

        //Cube_gold<double> paths_inner(stepsForEvolution,numberStocks,subsimPaths,0.0);

        std::vector<double> payoffs_inner_vec(stepsForEvolution*subsimPaths,0.0);
        //    Matrix_gold<double> payoffs_inner(stepsForEvolution,subsimPaths,0.0);

        std::vector<double> rate1_cf_i_vec(subsimPaths*stepsForEvolution);
        std::vector<double> rate2_cf_i_vec(subsimPaths*stepsForEvolution);
        std::vector<double> rate3_cf_i_vec(subsimPaths*stepsForEvolution);

        Cube_gold<double> paths_inner_values_deflated_cube(outerPaths,stepsForEvolution,subsimPaths);

        std::vector<int> innerPathSizes(outerPaths,subsimPaths);

        for (int p=0; p < outerPaths; ++p)
        {
            if (p % 20 == 0) 
                std::cout << p << " ";
            for (int stepsToSkip=1; stepsToSkip< stepsForEvolution; ++stepsToSkip)
            {

                for (int sto=0; sto<numberStocks;++sto)
                    current_stocks_vec[sto] = outerPaths_cube(stepsToSkip-1,sto,p);

                CubeFacade<double> variates_inner(variates_inner_vec,stepsForEvolution,factors,innerPathSizes[p]);
                CubeConstFacade<double> variates_inner_const(variates_inner_vec,stepsForEvolution,factors,innerPathSizes[p]);

                rng.populateCubeWithNormals<CubeFacade<double>,double>(variates_inner);

                CubeFacade<double> paths_inner(paths_inner_vec,stepsForEvolution,numberStocks,innerPathSizes[p]);

                generator.getPaths(innerPathSizes[p],0,
                    stepsToSkip,
                    current_stocks_vec,
                    variates_inner_const,
                    stepsToSkip, // specifies  how to associate normals in cube with steps in simulation
                    paths_inner);

                //    debugDumpCube(paths_inner.ConstFacade(),"paths inner");

                CubeFacade<double> basisVariables_inner(basisVariables_inner_vec,stepsForEvolution,numberVariables,subsimPaths);


                GenerateMultiDEquityBasisVariables<MaxCallVariables>(paths_inner, 
                    theVariables, 
                    pathOffsetForOutput,
                    stepsToSkip,
                    stepsToSkip,
                    basisVariables_inner);

                // generate payoffs

                MatrixFacade<double> payoffs_inner(payoffs_inner_vec,stepsForEvolution,subsimPaths);


                GenerateMultiDEquityPayoffs<MaxCall>(paths_inner, thePayOff, pathOffsetForOutput,stepsToSkip,stepsToSkip, payoffs_inner);

                // debugDumpMatrix(payoffs_inner.ConstFacade(),"payoffs_inner");


                // generate rates
                for (int s=0; s < stepsForEvolution; ++s)
                {

                    for (int p1=0; p1 < subsimPaths; ++p1)
                    {
                        rate1_cf_i_vec[p1+s*subsimPaths] = basisVariables_inner(s,0,p1);
                        rate2_cf_i_vec[p1+s*subsimPaths] = basisVariables_inner(s,1,p1);
                        rate3_cf_i_vec[p1+s*subsimPaths] = basisVariables_inner(s,2,p1);

                    }
                }

                if (useCrossTerms)
                {

                    cashFlowGeneratorEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialCrossGenerator,double>, double>(
                        pathsPayoffs, // output  
                        pathsPayoffIndices, // output for discounting
                        payoffs_inner,
                        exMCrossStrategy,
                        subsimPaths, 
                        stepsForEvolution,
                        stepsToSkip,
                        exerciseIndicatorsbool_vec,
                        rate1_cf_vec, 
                        rate2_cf_vec, 
                        rate3_cf_vec,  
                        basisVariables_inner,
                        paths_inner_vec ,
                        discounts_vec )  ; 



                }
                else
                {


                    cashFlowGeneratorEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialGenerator,double>, double>(
                        pathsPayoffs, // output  
                        pathsPayoffIndices, // output for discounting
                        payoffs_inner,
                        exMStrategy,
                        subsimPaths, 
                        stepsForEvolution,
                        stepsToSkip,
                        exerciseIndicatorsbool_vec,
                        rate1_cf_vec, 
                        rate2_cf_vec, 
                        rate3_cf_vec,  
                        basisVariables_inner,
                        paths_inner_vec,
                        discounts_vec )  ; 

                }

                //          debugDumpVector(pathsPayoffs,"paths payoffs");

                double x=0.0;
                double xsq=0;
                for (int ip=0; ip<subsimPaths;++ip)
                {
                    double pathValue = pathsPayoffs[ip]*discounts_const_vec[pathsPayoffIndices[ip]];
                    paths_inner_values_deflated_cube(p,stepsToSkip,ip)=pathValue;
                    x+=pathValue;
                    xsq+=pathValue*pathValue;
                }
                double m =( means_mat(p,stepsToSkip-1) = x/subsimPaths);
                sds_mat(p,stepsToSkip-1) =( sqrt(xsq/subsimPaths-m*m))/sqrt(subsimPaths);

            }
        }

        {
            if (dump)
            {
                dumpMatrix(exercise_occurences_outer_mat.ConstFacade(),"exercise_occurences_outer_mat");
                dumpMatrix(payoffs_discounted_outer_mat.ConstFacade(),"payoffs_discounted_outer_mat");
                dumpMatrix(means_mat.ConstFacade(),"means_mat");
                dumpCube(outerPaths_cube.ConstFacade(),"outerPaths_cube");
            }
        }
        ABDualityGapEstimateNonZeroRebate(exercise_occurences_outer_mat,
            payoffs_discounted_outer_mat,
            means_mat,
            sds_mat,
            meanAB,
            seAB,
            exerciseIndicatorsbool_vec,
            workspace_vec);


        std::cout << "mean and se of duality gap are " << meanAB << " " << seAB << "\n";




        int t3=clock();

        {
            double meanU1, se1,meanU2,se2,meanU3,se3;
            std::vector<double> meanU4,se4;
            ABDualityGapEstimateGeneralRebateBiasEstimation(exercise_occurences_outer_mat,
                means_mat,
                sds_mat,
                payoffs_discounted_outer_mat,
                meanU1,
                se1,
                meanU2,
                se2,
                meanU3,
                se3,
                meanU4,
                se4,
                pathsForGaussianEstimation_vec,
                biasseed,
                exerciseIndicatorsbool_vec,
                stepsForEvolution,
                twoPtEst, threePtEst, mcEst);


            std::cout <<outerPaths << "," << subsimPaths <<","<<" Mean and se of duality gap , " << meanU1 << "," << se1<<"," <<meanU2 << "," << se2
                <<"," <<meanU3 << "," << se3<< ",";
            dumpVector(meanU4, "meanu4");
            dumpVector(se4, "se4");


            if (dumpBiases)
            {
                for (int p=0; p< outerPaths; ++p)
                    std::cout << twoPtEst[p] << ",";

                std::cout << ", ,";

                for (int p=0; p< outerPaths; ++p)
                    std::cout << threePtEst[p] << ",";

                std::cout << ", ,";

                for (int p=0; p< outerPaths; ++p)
                    std::cout << mcEst[p] << ",";
            }

            std::cout << "\n";
        }

        Matrix_gold<double> means_partial_mat(means_mat);
        Matrix_gold<double> sds_partial_mat(sds_mat);
        Matrix_gold<double> M_mat(outerPaths,stepsForEvolution,0.0);
        Matrix_gold<double> X_mat(outerPaths,stepsForEvolution,1.0);
        Matrix_gold<double> X_dummy_mat(outerPaths,stepsForEvolution,1.0);
        std::vector<double>  pathwiseMixedVals_vec;                    

        if (lastSubsimSize_bool)
            for (int pathsToUse = minPathsForUpper; pathsToUse <= subsimPaths; ++pathsToUse)
            {
                // first get partial means and sds info
             



                for (int p=0; p < outerPaths; ++p)
                    for (int s=0; s +1 < stepsForEvolution; ++s)
                    {

                        MonteCarloStatisticsSimple stats(1,names);

                        for (int i=0; i <pathsToUse; ++i)
                        {
                            data_vec[0] = paths_inner_values_deflated_cube(p,s+1,i);
                            stats.AddDataVector(data_vec);
                        }
                        std::vector<std::vector<double> > res(stats.GetStatistics());
                        means_partial_mat(p,s) = res[0][0];
                        sds_partial_mat(p,s) = res[1][0];

                    }
                    double meanU1, se1,meanU2,se2,meanU3,se3;
                    std::vector<double> meanU4,se4;

                    std::vector<int> gaussianPaths_null_vec;

                    ABDualityGapEstimateGeneralRebateBiasEstimation(exercise_occurences_outer_mat,
                        means_partial_mat,
                        sds_partial_mat,
                        payoffs_discounted_outer_mat,
                        meanU1,
                        se1,
                        meanU2,
                        se2,
                        meanU3,
                        se3,
                        meanU4,
                        se4,
                        gaussianPaths_null_vec,// int pathsForGaussianEstimation,
                        0,// int seed,
                        exerciseIndicatorsbool_vec,
                        stepsForEvolution,
                        twoPtEst, threePtEst, mcEst);


                    std::cout <<outerPaths << ","<<pathsToUse<<  ", Mean and se of duality gap , " << meanU1 << "," << se1<<"," <<meanU2 << "," << se2
                        <<"," <<meanU3 << "," << se3<< ",";


                    //now lets see the multiplicative and mi

                    



                    ComputeABHedgeVector(exercise_occurences_outer_mat,
                        payoffs_discounted_outer_mat,
                        means_partial_mat,
                        M_mat.Facade());


                    double ABMixedCheckMean,ABMixedCheckSe;

                      MixedDualityGapEstimateNonZeroRebate(
                            payoffs_discounted_outer_mat,
                            M_mat,
                            X_dummy_mat,
                            ABMixedCheckMean,
                            ABMixedCheckSe,
                            exerciseIndicatorsbool_vec,
                            pathwiseMixedVals_vec);

                       std::cout << "," << ABMixedCheckMean <<"," << ABMixedCheckSe <<", now varying theta ,";

                 //      debugDumpMatrix(M_mat.ConstFacade(),"M_mat");

                      // debugDumpVector(pathwiseMixedVals_vec,"pathwiseMixedVals_vec");
                    int Ntheta = 5;
                    for (int i=0; i <= Ntheta; ++i)
                    {
                        double meanMixed,seMixed;

                        double theta = (1.0*i)/Ntheta;

                        ComputeMixedHedgeVector(exercise_occurences_outer_mat,
                            payoffs_discounted_outer_mat,
                            means_partial_mat,secondPassValue, theta, 
                            X_mat.Facade());

                        MixedDualityGapEstimateNonZeroRebate(
                            payoffs_discounted_outer_mat,
                            M_mat,
                            X_mat,
                            meanMixed,
                            seMixed,
                            exerciseIndicatorsbool_vec,
                            pathwiseMixedVals_vec);

                        std::cout  << meanMixed << ",";
         
                        //std::cout <<outerPaths <<  ",Theta," << theta <<", Mean and se of mixed duality gap , " << meanMixed << "," << seMixed << ",";
                    }   
                    
                    ComputeMixedHedgeVector(exercise_occurences_outer_mat,
                            payoffs_discounted_outer_mat,
                            means_partial_mat,secondPassValue, 1.0, 
                            M_mat.Facade());

                    std::cout << ", using multiplicatvie for M ,";

                    for (int i=0; i <= Ntheta; ++i)
                    {
                        double meanMixed,seMixed;

                        double theta = (1.0*i)/Ntheta;

                        ComputeMixedHedgeVector(exercise_occurences_outer_mat,
                            payoffs_discounted_outer_mat,
                            means_partial_mat,secondPassValue, theta, 
                            X_mat.Facade());

                        MixedDualityGapEstimateNonZeroRebate(
                            payoffs_discounted_outer_mat,
                            M_mat,
                            X_mat,
                            meanMixed,
                            seMixed,
                            exerciseIndicatorsbool_vec,
                            pathwiseMixedVals_vec);

                        std::cout  << meanMixed << ",";
         
                        //std::cout <<outerPaths <<  ",Theta," << theta <<", Mean and se of mixed duality gap , " << meanMixed << "," << seMixed << ",";
                    }

                    if (dumpBiases)
                    {
                        std::cout << ",two pt biases ,";

                        for (int p=0; p< outerPaths; ++p)
                            std::cout << twoPtEst[p] << ",";

                        std::cout << ", three pt biases ,";

                        for (int p=0; p< outerPaths; ++p)
                            std::cout << threePtEst[p] << ",,";



                        std::cout << "\n";
                    }
                    else
                        std::cout << "\n";



            }
            int t4=clock();


            double dt1 = (t1-t0)*ticksize;
            double dt2 = (t2-t1)*ticksize;
            double dt3 = (t3-t2)*ticksize;
            double dt4 = (t4-t3)*ticksize;


            std::cout << " times: first pass ,"<<dt1 << ", second pass, " << dt2 << ", AB upper ," << dt3 <<" , AB upper bias est ,"<<dt4 << "\n";
    }

    addPower obj(powerToUse);


    for (int extraPathsIndex=0; extraPathsIndex < extraPaths_vec.size(); ++extraPathsIndex)
    {
        int t4o=clock();

        int extraPaths = extraPaths_vec[extraPathsIndex];

        double totalBias = std::accumulate(threePtEst.begin(),threePtEst.end(),0.0, obj);
        //   double totalBiasSq = std::inner_product(threePtEst.begin(),threePtEst.end(),threePtEst.begin(),0.0);



        std::vector<int> refinedPathNumbers_vec(outerPaths);

        for (int p=0; p < outerPaths; ++p)
            refinedPathNumbers_vec[p] = static_cast<int>(subsimPaths + extraPaths*obj(0.0,threePtEst[p])/totalBias);

        int maxPathsForSubsim = *std::max_element(refinedPathNumbers_vec.begin(),refinedPathNumbers_vec.end());

        variates_inner_vec.resize(stepsForEvolution*factors*maxPathsForSubsim);
        std::vector<double> paths_inner_vec(stepsForEvolution*numberStocks*maxPathsForSubsim);
        std::vector<double> basisVariables_inner_vec(stepsForEvolution*numberVariables*maxPathsForSubsim);
        std::vector<double>  payoffs_inner_vec(stepsForEvolution*maxPathsForSubsim);
        std::vector<double> rate1_cf_i_vec(stepsForEvolution*maxPathsForSubsim);
        std::vector<double> rate2_cf_i_vec(stepsForEvolution*maxPathsForSubsim);
        std::vector<double> rate3_cf_i_vec(stepsForEvolution*maxPathsForSubsim);

        for (int p=0; p < outerPaths; ++p) 
        {
            if (p % 20 == 0) 
                std::cout << p << ", ";
            for (int stepsToSkip=1; stepsToSkip< stepsForEvolution; ++stepsToSkip)
            {

                for (int sto=0; sto<numberStocks;++sto)
                    current_stocks_vec[sto] = outerPaths_cube(stepsToSkip-1,sto,p);


                CubeFacade<double> variates_inner(variates_inner_vec,stepsForEvolution,factors,refinedPathNumbers_vec[p]);
                CubeConstFacade<double> variates_inner_const(variates_inner_vec,stepsForEvolution,factors,refinedPathNumbers_vec[p]);

                rng.populateCubeWithNormals<CubeFacade<double>,double>(variates_inner);

                CubeFacade<double> paths_inner(paths_inner_vec,stepsForEvolution,numberStocks,refinedPathNumbers_vec[p]);

                generator.getPaths(refinedPathNumbers_vec[p],0,
                    stepsToSkip,
                    current_stocks_vec,
                    variates_inner_const,
                    stepsToSkip, // specifies  how to associate normals in cube with steps in simulation
                    paths_inner);

                //    debugDumpCube(paths_inner.ConstFacade(),"paths inner");

                CubeFacade<double> basisVariables_inner(basisVariables_inner_vec,stepsForEvolution,numberVariables,refinedPathNumbers_vec[p]);


                GenerateMultiDEquityBasisVariables<MaxCallVariables>(paths_inner, 
                    theVariables, 
                    pathOffsetForOutput,
                    stepsToSkip,
                    stepsToSkip,
                    basisVariables_inner);

                // generate payoffs

                MatrixFacade<double> payoffs_inner(payoffs_inner_vec,stepsForEvolution,refinedPathNumbers_vec[p]);


                GenerateMultiDEquityPayoffs<MaxCall>(paths_inner, thePayOff, pathOffsetForOutput,stepsToSkip,stepsToSkip, payoffs_inner);

                // debugDumpMatrix(payoffs_inner.ConstFacade(),"payoffs_inner");


                // generate rates
                for (int s=0; s < stepsForEvolution; ++s)
                {

                    for (int p1=0; p1 < refinedPathNumbers_vec[p]; ++p1)
                    {
                        rate1_cf_i_vec[p1+s*refinedPathNumbers_vec[p]] = basisVariables_inner(s,0,p1);
                        rate2_cf_i_vec[p1+s*refinedPathNumbers_vec[p]] = basisVariables_inner(s,1,p1);
                        rate3_cf_i_vec[p1+s*refinedPathNumbers_vec[p]] = basisVariables_inner(s,2,p1);

                    }
                }

                if (useCrossTerms)
                {

                    cashFlowGeneratorEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialCrossGenerator,double>, double>(
                        pathsPayoffs, // output  
                        pathsPayoffIndices, // output for discounting
                        payoffs_inner,
                        exMCrossStrategy,
                        refinedPathNumbers_vec[p], 
                        stepsForEvolution,
                        stepsToSkip,
                        exerciseIndicatorsbool_vec,
                        rate1_cf_vec, 
                        rate2_cf_vec, 
                        rate3_cf_vec,  
                        basisVariables_inner,
                        paths_inner_vec ,
                        discounts_vec )  ; 



                }
                else
                {


                    cashFlowGeneratorEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialGenerator,double>, double>(
                        pathsPayoffs, // output  
                        pathsPayoffIndices, // output for discounting
                        payoffs_inner,
                        exMStrategy,
                        refinedPathNumbers_vec[p], 
                        stepsForEvolution,
                        stepsToSkip,
                        exerciseIndicatorsbool_vec,
                        rate1_cf_vec, 
                        rate2_cf_vec, 
                        rate3_cf_vec,  
                        basisVariables_inner,
                        paths_inner_vec,
                        discounts_vec )  ; 

                }

                //          debugDumpVector(pathsPayoffs,"paths payoffs");

                double x=0.0;
                double xsq=0;
                for (int ip=0; ip<refinedPathNumbers_vec[p];++ip)
                {
                    double pathValue = pathsPayoffs[ip]*discounts_const_vec[pathsPayoffIndices[ip]];
                    //      paths_inner_values_deflated_cube(p,stepsToSkip,ip)=pathValue;
                    x+=pathValue;
                    xsq+=pathValue*pathValue;
                }
                double m =( means_mat(p,stepsToSkip-1) = x/refinedPathNumbers_vec[p]);
                sds_mat(p,stepsToSkip-1) =( sqrt(xsq/refinedPathNumbers_vec[p]-m*m))/sqrt(refinedPathNumbers_vec[p]);

            }
        }
        ABDualityGapEstimateNonZeroRebate(exercise_occurences_outer_mat,
            payoffs_discounted_outer_mat,
            means_mat,
            sds_mat,
            meanAB,
            seAB,
            exerciseIndicatorsbool_vec,
            workspace_vec);

        //  std::vector<double> twoPtEst, threePtEst, mcEst;
        int t5o=clock();

        std::cout << "\nmean and se of duality gap are, " << meanAB << " ," << seAB << "\n";

        int t5=clock();

        double meanU1, se1,meanU2,se2,meanU3,se3;
        std::vector<double> meanU4,se4;
        std::vector<int> gauss_null_vec;

        ABDualityGapEstimateGeneralRebateBiasEstimation(exercise_occurences_outer_mat,
            means_mat,
            sds_mat,
            payoffs_discounted_outer_mat,
            meanU1,
            se1,
            meanU2,
            se2,
            meanU3,
            se3,
            meanU4,
            se4,
            gauss_null_vec,//  pathsForGaussianEstimation,
            0,// int seed,
            exerciseIndicatorsbool_vec,
            stepsForEvolution,
            twoPtEst, threePtEst, mcEst);


        std::cout <<outerPaths <<  ", Mean and se of duality gap , " << meanU1 << "," << se1<<"," <<meanU2 << "," << se2
            <<"," <<meanU3 << "," << se3<< ",," ;

        if (dumpBiases)
        {
            std::cout << ",two pt biases ,";

            for (int p=0; p< outerPaths; ++p)
                std::cout << twoPtEst[p] << ",";

            std::cout << ", three pt biases ,";

            for (int p=0; p< outerPaths; ++p)
                std::cout << threePtEst[p] << ",";

            std::cout << ",MC biases ,";

            for (int p=0; p< outerPaths; ++p)
                std::cout << mcEst[p] << ",";

            std::cout << "\n";
        }
        else
            std::cout << "\n";

        int t6=clock();

        double dt5 = (t5-t4o)*ticksize;
        double dt6 = (t6-t5)*ticksize;


        std::cout << " times: extra paths ,"<<extraPaths << ", AB upper adaptive ," << dt5 <<" , AB upper bias est ,"<<dt6 << "\n";

    }
    // now lets look at the mixed results

    Matrix_gold<double> M_mat(outerPaths,stepsForEvolution,0.0);
    Matrix_gold<double> X_mat(outerPaths,stepsForEvolution,1.0);
    Matrix_gold<double> X_dummy_mat(outerPaths,stepsForEvolution,1.0);

    ComputeMultiplicativeHedgeVector(exercise_occurences_outer_mat,
        payoffs_discounted_outer_mat,
        means_mat,
        X_mat.Facade());


    ComputeABHedgeVector(exercise_occurences_outer_mat,
        payoffs_discounted_outer_mat,
        means_mat,
        M_mat.Facade());

    double meanMixed,seMixed;

    std::vector<double>  pathwiseMixedVals_vec;

    MixedDualityGapEstimateNonZeroRebate(
        payoffs_discounted_outer_mat,
        M_mat,
        X_mat,
        meanMixed,
        seMixed,
        exerciseIndicatorsbool_vec,
        pathwiseMixedVals_vec);

    double meanAB2,seAB2;

    std::vector<double>  pathwiseABVals2_vec;
    MixedDualityGapEstimateNonZeroRebate(
        payoffs_discounted_outer_mat,
        M_mat,
        X_dummy_mat,
        meanAB2,
        seAB2,
        exerciseIndicatorsbool_vec,
        pathwiseABVals2_vec);
   //    debugDumpMatrix(M_mat.ConstFacade(),"M_mat");//

       
      //                 debugDumpVector(pathwiseABVals2_vec,"pathwiseABVals2_vec");

       double meanABM,seABM;

    std::vector<double>  pathwiseMultVals_vec;
    MixedDualityGapEstimateNonZeroRebate(
        payoffs_discounted_outer_mat,
        X_mat,
        X_mat,
        meanABM,
        seABM,
        exerciseIndicatorsbool_vec,
        pathwiseMultVals_vec);

    std::cout <<outerPaths <<  ", Mean and se of mixed duality gap , " << meanMixed << "," << seMixed << "\n";
    std::cout <<outerPaths <<  ", Mean and se of AB2 duality gap , " << meanAB2 << "," << seAB2 << "\n";
     std::cout <<outerPaths <<  ", Mean and se of multiplicative duality gap , " << meanABM << "," << seABM << "\n";
   
    /*
    int Ntheta = 20;
    for (int i=0; i <= Ntheta; ++i)
    {
        double meanMixed,seMixed;

        double theta = (1.0*i)/Ntheta;

        ComputeMixedHedgeVector(exercise_occurences_outer_mat,
            payoffs_discounted_outer_mat,
            means_mat,secondPassValue, theta, 
            X_mat.Facade());

        MixedDualityGapEstimateNonZeroRebate(
            payoffs_discounted_outer_mat,
            M_mat,
            X_mat,
            meanMixed,
            seMixed,
            exerciseIndicatorsbool_vec,
            pathwiseMixedVals_vec);

        std::cout <<outerPaths <<  "," << theta <<", Mean and se of mixed duality gap , " << meanMixed << "," << seMixed << "\n";
    }

    */
    return 0;
}


