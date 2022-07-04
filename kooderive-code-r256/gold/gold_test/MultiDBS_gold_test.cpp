//
//											multidbs_gold_test.cpp
//
//

// (c) Mark Joshi 2015
// released under GPL v 3.0
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



int MultiDEvolverPartialTestMain()
{


    int paths = 16384;

    int numberStocks=5;
    int factors=5;
    int stepsForEvolution=9; 

    int errors =stepsForEvolution;

    double sigma=0.2;
    double rho=0;
    double T=3.0;
    double r =0.05;
    double d =0.1;
    double S0 =100;
    double strike=100.0;
    int upperSeed = 123343;

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

    MultiDBS_PathGen_Partial_gold<double> generator( 
        numberStocks, 
        factors, 
        stepsForEvolution, 
        pseudo_roots_cube.getDataVector(),
        fixed_drifts_mat.getDataVector()
        );

    Cube_gold<double> variates(stepsForEvolution,factors,paths);
    Cube_gold<double> stocks_out_cube(stepsForEvolution,numberStocks,paths);


    MersenneTwisterUniformRng rng(upperSeed);
    rng.populateCubeWithNormals<CubeFacade<double>,double>(variates.Facade());
    std::vector<double> currentStocks_vec(numberStocks);

    for (int stepsToSkip=0; stepsToSkip < stepsForEvolution; ++stepsToSkip)
    {
        for (int i=0; i < numberStocks; ++i)
            currentStocks_vec[i] = S0+i*stepsToSkip+stepsToSkip;

        generator.getPaths(paths,
            0,
            stepsToSkip,
            currentStocks_vec,
            variates.ConstFacade(),
            stepsToSkip, // specifies  how to associate normals in cube with steps in simulation
            stocks_out_cube.Facade());

        double dT = (stepsForEvolution-stepsToSkip)*T/stepsForEvolution;
        double df = exp(-r*dT);
        std::vector<std::string> names(numberStocks);

        MonteCarloStatisticsSimple stats(numberStocks,names);

        std::vector<double> data(numberStocks);

        for (int p=0; p < paths;++p)
        {
            for (int s=0; s<numberStocks; ++s)
                data[s] = df*std::max(stocks_out_cube(stepsForEvolution-1,s,p)-strike,0.0);

            stats.AddDataVector(data);

        }

        std::vector<std::vector<double> > res(stats.GetStatistics());
        bool fail = false;
        for (int s=0; s<numberStocks; ++s)
        {
            double mcPrice = res[0][s];
            double mcSe = res[1][s];

            double BlackScholesCallPrice = BlackScholesCall(currentStocks_vec[s],r,d,dT,sigma,strike);
            double errSize = fabs(mcPrice - BlackScholesCallPrice)/mcSe;

            if (errSize > 4.0)
            {
                std::cout <<" partial generator test failed: " << mcPrice << " " << mcSe << " " << BlackScholesCallPrice << "\n";

                fail =true;
            }
        }

        if (!fail)
            --errors;


    }

    if (errors==0)
        std::cout << " partial generators test passed.\n";

    return errors;
}

int MultiDEvolerTestMain()
{
    int errors =20;

    int paths = 16384*4;

    int indicativePaths =paths; 
    int pathsPerSecondPassBatch=paths;
    int numberSecondPassBatches = 1;
    int duplicate =0;

    int numberStocks=5;
    int factors=5;
    int stepsForEvolution=9; 
    int powerOfTwoForVariates=4;

    double sigma=0.2;
    double rho=0;
    double T=3.0;
    double r =0.05;
    double d =0.1;
    double S0 =100;
    double strike=100;

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
    std::vector<unsigned int> scrambler_vec(factors*numberStocks*stepsForEvolution,0);

    std::vector<double> evolved_stocks_vec;     
    std::vector<double> evolved_log_stocks_vec;     


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

    // test 1
    {
        int olderrors=errors;
        std::vector<std::string> names(numberStocks);

        std::vector<double> strikes(numberStocks); 
        for (int i=0; i<numberStocks; ++i)
        {
            strikes[i] = 90+5.0*i;    
            names[i] = "call price";
        }

        MonteCarloStatisticsSimple stats(numberStocks,names);

        std::vector<double> data_vec(numberStocks);

        double df = exp(-r*T);

        for (int i=0; i < paths; ++i)
        {
            for (int j=0; j < numberStocks; ++j)
            {
                double spot = evolved_stocks_cube(stepsForEvolution-1,j,i);
                data_vec[j] = std::max(spot-strikes[j],0.0)*df;
            }
            stats.AddDataVector(data_vec);
        }
        std::vector<std::vector<double> > res(stats.GetStatistics());

        for (int j=0; j < numberStocks; ++j)
        {
            double MCCall = res[0][j];
            double MCCallse = res[1][j];

            double call=  BlackScholesCall(initial_stocks_vec[j],  r,  d,  T,  sigma,  strikes[j]);

            if ((fabs(call - MCCall)/MCCallse) > 3.0)
                std::cout << "call MCCall mismatch " << call << " " << MCCall << " s.e. " << MCCallse << "\n";
            else --errors;
        }

        if (errors +numberStocks==olderrors)
            std::cout << "multi D MC call pricing test passed\n";
    }
    // now margrabe,
    // test 2
    {
        int preverrors=errors;
        std::vector<std::string> names(numberStocks);

        std::vector<double> strikes(numberStocks); 
        for (int i=0; i<numberStocks; ++i)
        {
            names[i] = "margrabe price";
        }

        MonteCarloStatisticsSimple stats(numberStocks,names);

        std::vector<double> data_vec(numberStocks);

        double df = exp(-r*T);

        for (int i=0; i < paths; ++i)
        {
            for (int j=0; j < numberStocks; ++j)
            {
                double spot1 = evolved_stocks_cube(stepsForEvolution-1,j,i);
                double spot2 = evolved_stocks_cube(stepsForEvolution-1,(j+1) % numberStocks,i);

                data_vec[j] = std::max(spot2-spot1,0.0)*df;
            }
            stats.AddDataVector(data_vec);
        }
        std::vector<std::vector<double> > res(stats.GetStatistics());

        for (int j=0; j < numberStocks; ++j)
        {
            double MCMarb = res[0][j];
            double MCMarbse = res[1][j];

            double marb=  MargrabeBlackScholes(initial_stocks_vec[j],initial_stocks_vec[(j+1) % numberStocks], 
                T,   rho,  sigma,  sigma,d,d);

            if ((fabs(marb - MCMarb)/MCMarbse) > 3.0)
                std::cout << "Margrabe mismatch " << marb << " " << MCMarb  << " s.e. " << MCMarbse << "\n";
            else --errors;
        }

        if (errors ==preverrors-5)
            std::cout << "multi D MC margrabe pricing test passed\n";

    }
    // now maxcall
    // test 3
    {
        int preverrors=errors;
        std::vector<std::string> names(numberStocks);

        std::vector<double> strikes(numberStocks,strike); 
        for (int i=0; i<numberStocks; ++i)
        {
            names[i] = "maxcall price";
        }

        MonteCarloStatisticsSimple stats(numberStocks,names);

        std::vector<double> data_vec(numberStocks);

        double df = exp(-r*T);

        for (int i=0; i < paths; ++i)
        {
            for (int j=0; j < numberStocks; ++j)
            {
                double spot1 = evolved_stocks_cube(stepsForEvolution-1,j,i);
                double spot2 = evolved_stocks_cube(stepsForEvolution-1,(j+1) % numberStocks,i);

                data_vec[j] = std::max(spot2-strikes[j],std::max(spot1-strikes[j],0.0))*df;
            }
            stats.AddDataVector(data_vec);
        }
        std::vector<std::vector<double> > res(stats.GetStatistics());

        for (int j=0; j < numberStocks; ++j)
        {
            double MCMax = res[0][j];
            double MCMaxse = res[1][j];
            double maxC=  BSCallTwoMax(initial_stocks_vec[j],initial_stocks_vec[(j+1) % numberStocks], strikes[j],
                T, r,d,d,    sigma,  sigma,rho);

            if ((fabs(maxC - MCMax)/MCMaxse) > 3.0)
                std::cout << "maxcell mismatch " << maxC << " " << MCMax  << " s.e. " << MCMaxse << "\n";
            else --errors;
        }

        if (errors ==preverrors-5)
            std::cout << "multi D MC max call pricing test passed\n";

    }

    {
        double strike =100.0;
        MaxCall thePayOff(numberStocks,strike);

        int pathOffsetForOutput=0;

        Matrix_gold<double> payoffs(stepsForEvolution,paths+pathOffsetForOutput,0.0);



        GenerateMultiDEquityPayoffs<MaxCall>(evolved_stocks_cube, thePayOff, pathOffsetForOutput,0,0, payoffs.Facade());

        bool errorFound=false;

        for (int p=0; p < paths; ++p)
            for (int s=0; s < stepsForEvolution; ++s)
            {
                double maxis=0.0;

                for (int k=0; k < numberStocks; ++k)
                    maxis=std::max(maxis,evolved_stocks_cube(s,k,p));

                double payoff = std::max(maxis-strike,0.0);
                double testpayoff =  payoffs(s,p+pathOffsetForOutput);

                if (fabs(payoff -testpayoff)>1e-6)
                    errorFound=true;
            }

            if (errorFound)
                std::cout << "error in payoffs multi d equity check\n";
            else
            {
                --errors;
                std::cout << "payoffs generation test passed\n";
            }

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

            bool errorFoundVariables=false;
            std::vector<double> stocks(numberStocks);


            for (int p=0; p < paths; ++p)
                for (int s=0; s < stepsForEvolution; ++s)
                {
                    for (int k=0; k < numberStocks; ++k)
                        stocks[k] = evolved_stocks_cube(s,k,p);

                    std::sort(stocks.begin(),stocks.end());

                    for (int l=0; l < numberStockVariables; ++l)
                    {

                        double testVariable =  basisVariables(s,l,p+pathOffsetForOutput);

                        if (fabs(stocks[numberStocks-1-l] -testVariable)>1e-6)
                            errorFound=true;

                    }
                }
                if (errorFound)
                    std::cout << "error in variables multi d equity check\n";
                else
                {
                    --errors;
                    std::cout << "variables generation test passed\n";
                }

                // now run through the least squares code

                bool useCrossTerms=true;
                bool normalise = false;
                int numberOfExtraRegressions=5;
                int lowerPathCutoff = 5000;
                double lowerFrac = 0.25;
                double upperFrac=0.26;
                double initialSDguess = 2.5;
                double multiplier =1-pow(0.9,1.0/numberOfExtraRegressions);
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

                std::cout << "least squares multiple regression ran!:" << firstPassValue <<"\n";

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

                std::vector<std::string> names(2);


                names[0] = "call price 1";
                names[1] = "call price 2";

                MonteCarloStatisticsSimple stats(2,names);

                std::vector<double> data_vec(2);

                std::vector<double> pathsPayoffs(pathsPerSecondPassBatch);
                std::vector<int> pathsPayoffIndices(pathsPerSecondPassBatch);

                std::vector<double> discounts_const_vec(stepsForEvolution);
                for (int s=0; s < stepsForEvolution; ++s)
                {
                    double df = initialDf/some_numeraireValues_matrix(s,0);
                    discounts_const_vec[s]=df;

                }

                Matrix_gold<int> exercise_occurences_mat(stepsForEvolution,pathsPerSecondPassBatch,0);

                for (int secondBatch=0; secondBatch < numberSecondPassBatches; ++secondBatch)
                {
                    int pathOffset = duplicate*paths+secondBatch*pathsPerSecondPassBatch;

                    int pathOffsetForOutput=0;

                    // generate equity paths

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
                        cashFlowGeneratorEE_gold(
                            flow1_multi_vec,
                            flow2_multi_vec, 
                            nullProduct,
                            exerciseValue,
                            exMCrossStrategy,
                            pathsPerSecondPassBatch, 
                            stepsForEvolution,
                            exerciseIndicatorsbool_vec,
                            rate1_cf_vec, 
                            rate2_cf_vec, 
                            rate3_cf_vec, 
                            basisVariables.ConstFacade(),
                            evolved_stocks_vec, 
                            discounts_vec);

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

                         exerciseAllDeterminerEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialCrossGenerator,double>, double>(  
                             exercise_occurences_mat.Facade(), // 0 = false, 1 = true
                            payoffs.ConstFacade(),
                            exMCrossStrategy,
                            pathsPerSecondPassBatch, 
                            stepsForEvolution,
                            0, //steps to skip
                           exerciseIndicatorsbool_vec,
                           rate1_cf_vec, 
                            rate2_cf_vec, 
                            rate3_cf_vec,  
                            basisVariables.ConstFacade(),
                            evolved_stocks_vec ,
                            discounts_vec )   ;

                    }
                    else
                    {
                        cashFlowGeneratorEE_gold(
                            flow1_multi_vec,
                            flow2_multi_vec, 
                            nullProduct,
                            exerciseValue,
                            exMStrategy,
                            pathsPerSecondPassBatch, 
                            stepsForEvolution,
                            exerciseIndicatorsbool_vec,
                            rate1_cf_vec, 
                            rate2_cf_vec, 
                            rate3_cf_vec, 
                            basisVariables.ConstFacade(),
                            evolved_stocks_vec, 
                            discounts_vec);

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

                        
                         exerciseAllDeterminerEESimple_gold<   LSAMultiExerciseStrategy<quadraticPolynomialGenerator,double>, double>(  
                             exercise_occurences_mat.Facade(), // 0 = false, 1 = true
                            payoffs.ConstFacade(),
                            exMStrategy,
                            pathsPerSecondPassBatch, 
                            stepsForEvolution,
                            0, //steps to skip
                           exerciseIndicatorsbool_vec,
                           rate1_cf_vec, 
                            rate2_cf_vec, 
                            rate3_cf_vec,  
                            basisVariables.ConstFacade(),
                            evolved_stocks_vec ,
                            discounts_vec )   ;

                    }

                    for (int p=0; p < pathsPerSecondPassBatch; ++p)
                    {
                        double pathValue = 0.0;
                        for (int s=0; s < stepsForEvolution; ++s)
                        {
                            double df = initialDf/some_numeraireValues_matrix(s,0);
                            double cf = (flow1_multi_mat(s,p)+flow2_multi_mat(s,p));
                            pathValue += cf*df;
                        }
                        data_vec[0] = pathValue;


                        double pathValue2 = pathsPayoffs[p]*discounts_const_vec[pathsPayoffIndices[p]];
                        data_vec[1] = pathValue2;

                        stats.AddDataVector(data_vec);
                    }
                }

                std::vector<std::vector<double> > res(stats.GetStatistics());

                double secondPassValue = res[0][0];

                std::cout << "Second pass value: " << secondPassValue <<"\n";

                double se = res[1][0];

                std::cout << " standard error: " << se << "\n";

                if (fabs(firstPassValue-secondPassValue)>1E-6)
                    std::cout << " first pass does not agree with second pass LSEstM\n";
                else
                    errors--;

                double secondPassValue2 = res[0][1];

                std::cout << "Second pass value2: " << secondPassValue2 <<"\n";

                double se2 = res[1][1];

                std::cout << " standard error2: " << se2 << "\n";

                if (fabs(secondPassValue2-secondPassValue)>1E-6)
                    std::cout << " second  pass method 2 does not agree with second pass method 1\n" << secondPassValue<<" " << secondPassValue2<<"\n";
                else
                    errors--;

                int mismatches=0;

                for (int p=0; p < paths; ++p)
                {
                    int s=0; 
                    bool notfound =true;
                    while (s< stepsForEvolution && notfound)
                    {
                        if (exercise_occurences_mat(s,p) ==0)
                            ++s;
                        else
                            notfound= false;
                    }

                    if (s != pathsPayoffIndices[p] && !notfound)
                        ++mismatches;




                }

                if (mismatches >0)
                    std::cout << " exerciseAllDeterminerEESimple_gold does not agree.\n";
                else 
                {
                    std::cout << " exerciseAllDeterminerEESimple_gold agrees.\n";
                    --errors;
                }

    }



    return errors;
}


