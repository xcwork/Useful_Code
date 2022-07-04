

//
//
//                                  LMM_Bermudan_gold.cpp
//
//
// (c) Mark Joshi 2014
// This code is released under the GNU public licence version 3

// develop examples of lower and upper bounds for the pricing of cancellable swaps using the gold code
#include <vector>

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


double safesqrt(double x)
{
    return x > 0 ? sqrt(x) : 0.0;
}

void BermudanExample(bool useLogBasis, 
                     bool useCrossTerms,
                     int pathsPerBatch, 
                     int numberOfBatches,
                     int numberOfExtraRegressions, 
                     int duplicate,
                     bool normalise,
                     int pathsPerSecondPassBatch,
                     int numberSecondPassBatches,
                     int choice,
                     int upperPaths,
                     int upperSeed,
                     int pathsPerSubsim,
                     int numberNonCallDates,
                     int gaussianPaths,
                     int innerEstSeed,
                     int minPathsToUse
        )
{

    std::cout  << ",useLogBasis, " << useLogBasis<< ",useCrossTerms," <<
                      useCrossTerms<< ",pathsPerBatch," <<
                      pathsPerBatch<< ",numberOfBatches," <<
                      numberOfBatches<< ",numberOfExtraRegressions," <<
                      numberOfExtraRegressions<< ",duplicate," << 
                      duplicate<< ",normalise," <<
                      normalise<< ",pathsPerSecondPassBatch," <<
                      pathsPerSecondPassBatch<< ",numberSecondPassBatches," <<
                      numberSecondPassBatches<< ",choice," <<
                      choice<< ",upperPaths," <<
                      upperPaths<< ",upperSeed," <<
                      upperSeed<< ",pathsPerSubsim," <<
                      pathsPerSubsim<< ",numberNonCallDates," <<
                      numberNonCallDates<< ",gaussianPaths," <<
                      gaussianPaths << ",innerEstSeed," <<
                      innerEstSeed;

    _controlfp(_EM_UNDERFLOW + _EM_INEXACT, _MCW_EM);
    // generate outer cashflows -- arguably not needed


    bool getRegressions = true;

    std::vector<double> xa(6);
    xa[0] = 0.01;
    xa[1] = 0.005;
    xa[2] = 0.0;
    xa[3] = 0.026;
    xa[4] = 0.032;
    xa[5] = 0.032;



    std::vector<int> rates(6);
    rates[0] = 10+numberNonCallDates;
    rates[1] = 18+numberNonCallDates;
    rates[2] = 38+numberNonCallDates;
    rates[3] = 1+numberNonCallDates;
    rates[4] = 2+numberNonCallDates;
    rates[5] = 6+numberNonCallDates;

    int number_rates = rates[choice];

    // find power of two at least as big as numberOfRates
    int n_poweroftwo =1;
    {
        int power =2;
        while (power < number_rates)
        {
            ++n_poweroftwo;
            power*=2;
        }
    }




    double x =xa[choice];
    double firstForward=0.008+x;
    double rateLength = 0.5;
    double firstDf = 1.0/(1+rateLength*firstForward);

    double forwardIncrement=0.002;
    firstForward += (3-numberNonCallDates)*forwardIncrement;
    double displacement=0.015f;
    double strike =0.04f;

    double beta=2*0.0669f;				  
    double L=0.0f;
    double a= 0.05;
    double b = 0.09;
    double c= 0.44;
    double d=0.2;
    bool useFlatVols = false;

    double firstRateTime =1.5-rateLength*numberNonCallDates;
    if (firstRateTime < 0.5)
        firstRateTime =0.5;

    int numberSteps = number_rates;
    Cube_gold<double> innerPathsValues(upperPaths,numberSteps,pathsPerSubsim);


    Matrix_gold<double> exerciseValues_mat(numberSteps,upperPaths,0.0);
    Matrix_gold<int> exerciseOccurences_mat(numberSteps,upperPaths,0);

    Matrix_gold<double> means_i_mat(upperPaths,numberSteps,0.0);
    Matrix_gold<double> sds_i_mat(upperPaths,numberSteps,0.0);
    std::vector<bool> exerciseIndicatorsbool_vec;



    double cutoffLevel =10.0;

    // dummy calibration data 
    std::vector<double> logRates_vec(number_rates);
    std::vector<double> rates_vec(number_rates);
    std::vector<double> taus_vec(number_rates);
    std::vector<double> displacements_vec(number_rates);
    std::vector<double> rateTimes_vec(number_rates+1);
    std::vector<double> evolutionTimes_vec(number_rates);

    std::vector<int> alive_vec(number_rates);

    rateTimes_vec[0] =firstRateTime;


    for (int i=0; i < number_rates; ++i)
    {
        rates_vec[i] = firstForward+i*forwardIncrement;
        displacements_vec[i] =displacement;
        logRates_vec[i] = log(rates_vec[i]+displacements_vec[i] );
        taus_vec[i] = rateLength;
        rateTimes_vec[i+1]= firstRateTime + rateLength*(i+1);
        evolutionTimes_vec[i] = rateTimes_vec[i];
        alive_vec[i] = i;

    }   



    double maxDisplacement = *std::max_element(displacements_vec.begin(),displacements_vec.end());

    double beta_d = 2*0.0668;
    double L_d = 0.0;
    double vol = 0.20;

    std::vector<double> vols(number_rates,vol);
    int upperStartTime;

    std::vector<double> Ks_vec(number_rates,1.0);

    int factors = std::min(number_rates,5);

    Cube_gold<double> pseudosDouble(useFlatVols ? FlatVolPseudoRootsOfCovariances(rateTimes_vec,
        evolutionTimes_vec,
        vols,
        factors,
        L_d,
        beta_d
        ) : 
    ABCDLBetaPseudoRoots(a,b,c,d,
        evolutionTimes_vec,
        evolutionTimes_vec,
        Ks_vec,
        factors,
        L,
        beta
        ));

    int lowerPathCutoff =1000;


    int totalPaths = numberOfBatches*pathsPerBatch;

    int maxRegressionDepth = numberOfExtraRegressions+1;

    double lowerFrac = 0.25;
    double upperFrac=0.26;
    double initialSDguess = 2.5;
    double multiplier =0.8;
    RegressionSelector_gold_Fraction regressionSelector(lowerFrac, upperFrac, initialSDguess,  multiplier);

    double cutOffLevel =100.0;

    // set the indices of the times at which cash-flows are determined
    std::vector<int> genTimeIndex_vec(numberSteps);
    for (size_t i=0;i< genTimeIndex_vec.size(); ++i)			
        genTimeIndex_vec[i] = static_cast<int>(i);

    int basisVariablesPerStep = basisVariableLog_gold<double>::maxVariablesPerStep();
    std::vector<int> exerciseIndices_vec; // the indices of the exercise times amongst the evolution times
    std::vector<int> exerciseIndicators_vec(numberSteps,0);
    for (int i=numberNonCallDates; i < numberSteps;++i) // 
    {
        exerciseIndicators_vec[i] =1;
    }

    GenerateIndices(exerciseIndicators_vec, exerciseIndices_vec);

    int numberOfExerciseDates = static_cast<int>(exerciseIndices_vec.size());
    std::vector<int> stepToExerciseIndices_vec;

    int firstPositiveStep = findExerciseTimeIndicesFromPaymentTimeIndices(genTimeIndex_vec, 
        exerciseIndices_vec,
        exerciseIndicators_vec, // redundant info but sometimes easier to work with
        stepToExerciseIndices_vec
        );

    // where to deflate cash-flows to when doing backward induction
    std::vector<int> deflation_locations_vec(exerciseIndices_vec);

    int totalNumberOfBasisVariables = basisVariablesPerStep*numberOfExerciseDates*totalPaths;

    {
        std::vector<unsigned int> scrambler_vec(factors*intPower(2,n_poweroftwo),0);

        std::vector<double> evolved_rates_vec(pathsPerBatch*numberSteps*number_rates);       
        std::vector<double> evolved_log_rates_vec(pathsPerBatch*numberSteps*number_rates);   
        std::vector<double> discounts_vec(pathsPerBatch*numberSteps*(number_rates+1));
        std::vector<double> annuities_vec(pathsPerBatch*numberSteps*number_rates);
        std::vector<double> swap_rates_vec(pathsPerBatch*numberSteps*number_rates);
        std::vector<double> rate1_vec(pathsPerBatch*numberSteps);
        std::vector<double> rate2_vec(pathsPerBatch*numberSteps);
        std::vector<double> rate3_vec(pathsPerBatch*numberSteps);
        std::vector<double> rate1_cf_vec(pathsPerBatch*numberSteps);
        std::vector<double> rate2_cf_vec(pathsPerBatch*numberSteps);
        std::vector<double> rate3_cf_vec(pathsPerBatch*numberSteps);

        std::vector<double> spot_measure_values_vec(numberSteps*totalPaths);
        std::vector<double> spot_small_measure_values_vec(numberSteps*pathsPerBatch);
        MatrixConstFacade<double> spot_measure_values_small_constMatrix(spot_small_measure_values_vec,numberSteps,pathsPerBatch);

        CubeConstFacade<double>  forwards_cube(evolved_rates_vec,numberSteps,number_rates,pathsPerBatch);
        CubeConstFacade<double>  discountRatios_cube(discounts_vec,numberSteps,number_rates+1,pathsPerBatch);
        MatrixConstFacade<double>	rate1_matrix(rate1_vec,numberSteps,pathsPerBatch);
        MatrixConstFacade<double>	rate2_matrix(rate2_vec,numberSteps,pathsPerBatch);
        MatrixConstFacade<double>	rate3_matrix(rate3_vec,numberSteps,pathsPerBatch);

        Cube_gold<double> basisFunctionVariables_gold_cube(numberOfExerciseDates,basisVariablesPerStep,totalPaths,0.0);

        std::vector<double> genFlows1_vec(numberSteps*pathsPerBatch);
        std::vector<double> genFlows2_vec(numberSteps*pathsPerBatch);

        MatrixFacade<double> genFlows1_cpu_matrix(&genFlows1_vec[0],numberSteps,pathsPerBatch);
        MatrixConstFacade<double> genFlows1_cpu_constmatrix(&genFlows1_vec[0],numberSteps,pathsPerBatch);

        MatrixFacade<double> genFlows2_cpu_matrix(&genFlows2_vec[0],numberSteps,pathsPerBatch);
        MatrixConstFacade<double> genFlows2_cpu_constmatrix(&genFlows2_vec[0],numberSteps,pathsPerBatch);

        std::vector<double> aggregatedFlows_vec(totalPaths*exerciseIndices_vec.size(),0.0);
        MatrixFacade<double> aggregatedFlows_matrix(&aggregatedFlows_vec[0],static_cast<int>(exerciseIndices_vec.size()),totalPaths);

        std::vector<double> aux_swap_data_vec(2*number_rates+2);

        double payReceive = 1.0f; // i.e. receive fixed
        aux_swap_data_vec[0] = strike;
        aux_swap_data_vec[1] = payReceive;
        std::vector<double> accruals1(number_rates);
        std::vector<double> accruals2(number_rates);

        for (int j=0; j < number_rates; ++j)
        {
            aux_swap_data_vec[j+2] =0.5f;
            accruals1[j] = aux_swap_data_vec[j+2];
            aux_swap_data_vec[number_rates+j+2] =0.5f;
            accruals2[j] =  aux_swap_data_vec[number_rates+j+2];
        }

        std::vector<int> extractor1(numberOfExerciseDates);
        for (int i=0; i < static_cast<int>(extractor1.size()); ++i)
            extractor1[i] = exerciseIndices_vec[i];

        std::vector<int> extractor2(numberOfExerciseDates);
        for (int i=0; i < static_cast<int>(extractor2.size()); ++i)
            extractor2[i] = std::min( exerciseIndices_vec[i]+1,numberSteps-1);

        std::vector<int> extractor3(numberOfExerciseDates);
        for (int i=0; i < static_cast<int>(extractor3.size()); ++i)
            extractor3[i] = numberSteps-1;


        std::vector<int> extractor1_cf(numberSteps);
        for (int i=0; i < static_cast<int>(extractor1_cf.size()); ++i)
            extractor1_cf[i] = i;

        std::vector<int> extractor2_cf(numberSteps);
        for (int i=0; i < static_cast<int>(extractor2_cf.size()); ++i)
            extractor2_cf[i] = std::min( i+1,numberSteps-1);

        std::vector<int> extractor3_cf(numberSteps);
        for (int i=0; i < static_cast<int>(extractor3_cf.size()); ++i)
            extractor3_cf[i] =std::min( i+2,numberSteps-1);

        std::vector<int> integerData_vec(1);
        std::vector<double> floatData_vec(3,maxDisplacement);
        floatData_vec[2] = 0.0;

        std::vector<double> paymentTimes_vec(numberSteps);

        for (size_t i=0;i< paymentTimes_vec.size(); ++i)			
            paymentTimes_vec[i] = rateTimes_vec[i+1];



        std::vector<int> firstIndex_vec, secondIndex_vec;
        std::vector<double> thetas_vec;

        generateCashFlowIndicesAndWeights<double>( firstIndex_vec, 
            secondIndex_vec,
            thetas_vec,
            rateTimes_vec,
            paymentTimes_vec
            );
        std::vector<int> fromGenIndicesToExerciseDeflations_vec(genTimeIndex_vec.size());
        for (size_t i=0; i < genTimeIndex_vec.size();++i)
        {

            int j=stepToExerciseIndices_vec[i];

            fromGenIndicesToExerciseDeflations_vec[i] = j>=0 ? deflation_locations_vec[j] :0 ;
        }

        std::vector<double> discounted_flows1_vec(pathsPerBatch*numberSteps);
        MatrixFacade<double> discounted_flows1_Matrix(&discounted_flows1_vec[0],numberSteps,pathsPerBatch);
        MatrixConstFacade<double> discounted_flows1_constMatrix(&discounted_flows1_vec[0],numberSteps,pathsPerBatch);
        std::vector<double> discounted_flows2_vec(pathsPerBatch*numberSteps);
        MatrixFacade<double> discounted_flows2_Matrix(&discounted_flows2_vec[0],numberSteps,pathsPerBatch);
        MatrixConstFacade<double> discounted_flows2_constMatrix(&discounted_flows2_vec[0],numberSteps,pathsPerBatch);



        LMMEvolutionFullPCSobol_gold<double> LMMevolver(pathsPerBatch, 
            number_rates, 
            factors, 
            numberSteps, 
            n_poweroftwo,
            pseudosDouble.getDataVector(),
            displacements_vec,
            rates_vec, 
            taus_vec,
            cutOffLevel
            );


        //; first pass, we store data
        // design allows storing of data from multiple batches
        // just store what's really needed rather than everyuthing


        for (int batch=0; batch<numberOfBatches; ++batch)
        {
            int pathOffset = batch*pathsPerBatch;

            // first develop LMM paths using PC algorithm
            LMMevolver.getPaths(pathsPerBatch,
                pathOffset,scrambler_vec,
                evolved_rates_vec,
                evolved_log_rates_vec
                );

            // debugDumpVector(evolved_rates_vec,"evolved_rates_vec");

            // compute all the implied ratios from the LMM paths
            discount_ratios_computation_gold( evolved_rates_vec, 
                taus_vec, 
                alive_vec, 
                pathsPerBatch,
                numberSteps, 
                number_rates, 
                discounts_vec  // for output 
                );    
            //        debugDumpMatrix(discounts_vec,"discounts_vec",rateTimes_vec.size(),pathsPerBatch);

            // turn discounts ratios into swap annuities
            coterminal_annuity_ratios_computation_gold(  discounts_vec, 
                taus_vec, 
                alive_vec, 
                pathsPerBatch,
                numberSteps, 
                number_rates, 
                annuities_vec  // for output 
                );

            // use discount ratios and swap annuities to compute swap rates
            coterminal_swap_rates_computation_gold(   discounts_vec, 
                annuities_vec, 
                alive_vec, 
                pathsPerBatch,
                numberSteps, 
                number_rates, 
                swap_rates_vec // for output 
                );

            // get the values of rate1 from forward rates
            forward_rate_extraction_selecting_steps_gold(  evolved_rates_vec, 
                extractor1,     
                exerciseIndices_vec,
                pathsPerBatch,
                number_rates, 
                numberSteps,
                rate1_vec                 
                );
            //  debugDumpVector(rate1_vec,"rate1_vec");

            // get the values of rate2 from swap rates
            forward_rate_extraction_selecting_steps_gold(  swap_rates_vec, 
                extractor2,     
                exerciseIndices_vec,
                pathsPerBatch,
                number_rates,
                numberSteps,
                rate2_vec                 
                );
            //   debugDumpVector(rate2_vec,"rate2_vec");
            // get the values of rates from annuities
            forward_rate_extraction_selecting_steps_gold(  annuities_vec, 
                extractor3,     
                exerciseIndices_vec,
                pathsPerBatch,
                number_rates, 
                numberSteps,
                rate3_vec                 
                );
            //          debugDumpVector(rate3_vec,"rate3_vec");
            // get rate1 for cash flows, this need not be the same as rate1 for basis
            forward_rate_extraction_gold(  evolved_rates_vec, 
                extractor1_cf,                          
                pathsPerBatch,
                numberSteps,
                number_rates, 
                rate1_cf_vec                
                );
            //        debugDumpVector(rate1_cf_vec,"rate1_cf_vec");


            // get rate2 for cash flows, this need not be the same as rate2 for basis
            forward_rate_extraction_gold(  evolved_rates_vec, 
                extractor2_cf,                          
                pathsPerBatch,
                numberSteps,
                number_rates, 
                rate2_cf_vec                
                );

            // get rate3 for cash flows, this need not be the same as rate3 for basis

            forward_rate_extraction_gold(  evolved_rates_vec, 
                extractor3_cf,                          
                pathsPerBatch,
                numberSteps,
                number_rates, 
                rate3_cf_vec                
                );

            adjoinBasisVariablesCaller_gold(useLogBasis,
                integerData_vec, // int data specific to the basis variables
                floatData_vec, // float data specific to the basis variables
                forwards_cube,
                discountRatios_cube,
                rate1_matrix,
                rate2_matrix,
                rate3_matrix,
                pathsPerBatch,
                totalPaths, 
                pathsPerBatch*batch,
                number_rates, 
                exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
                exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                basisFunctionVariables_gold_cube.Facade()// output location
                );



            spot_measure_numeraires_computation_offset_gold(  discounts_vec,
                pathsPerBatch,
                totalPaths,
                pathsPerBatch*batch,
                number_rates,
                spot_measure_values_vec
                );

            spot_measure_numeraires_computation_offset_gold(  discounts_vec,
                pathsPerBatch,
                pathsPerBatch,
                0,
                number_rates,
                spot_small_measure_values_vec
                );


            cashFlowGeneratorCallerSwap_gold<double>(  genFlows1_vec, 
                genFlows2_vec, 
                aux_swap_data_vec, 
                pathsPerBatch, 
                numberSteps,
                rate1_cf_vec, 
                rate2_cf_vec, 
                rate3_cf_vec, 
                evolved_rates_vec, 
                discounts_vec);



            cashFlowDiscountingPartial_gold<double>(
                genTimeIndex_vec, // indices of times at which flows are generated
                firstIndex_vec, // rate time index leq payment date 
                secondIndex_vec, // rate time index > payment date 
                thetas_vec, // interpolation fraction 
                fromGenIndicesToExerciseDeflations_vec, // rate time index to discount to 
                discountRatios_cube,  // all the discount ratios 
                genFlows1_cpu_constmatrix, 
                spot_measure_values_small_constMatrix,
                pathsPerBatch, 
                numberSteps, 
                discounted_flows1_Matrix);


            cashFlowDiscountingPartial_gold<double>(
                genTimeIndex_vec, // indices of times at which flows are generated
                firstIndex_vec, // rate time index leq payment date 
                secondIndex_vec, // rate time index > payment date 
                thetas_vec, // interpolation fraction 
                fromGenIndicesToExerciseDeflations_vec, // rate time index to discount to 
                discountRatios_cube,  // all the discount ratios 
                genFlows2_cpu_constmatrix, 
                spot_measure_values_small_constMatrix,
                pathsPerBatch, 
                numberSteps, 
                discounted_flows2_Matrix);

            AggregateFlows_gold<double>(aggregatedFlows_matrix, // for output, aggregrates are added to existing data
                MatrixConstFacade<double>(discounted_flows1_Matrix),
                stepToExerciseIndices_vec, 
                pathsPerBatch,
                pathsPerBatch*batch );

            AggregateFlows_gold<double>(aggregatedFlows_matrix, // for output, aggregrates are added to existing data
                MatrixConstFacade<double>(discounted_flows2_Matrix),
                stepToExerciseIndices_vec, 
                pathsPerBatch,
                pathsPerBatch*batch );

        } // end of batch loop


        //   debugDumpMatrix(aggregatedFlows_matrix,"aggregated flows matrix");


        int maxBasisVariables=basisVariableExample_gold<double>::maxVariablesPerStep( );

        int numberBasisFunctions =useCrossTerms ? quadraticPolynomialCrossGenerator(maxBasisVariables).numberDataPoints()
            : quadraticPolynomialGenerator(maxBasisVariables).numberDataPoints();




        std::vector<double> productsM_cube_vec;
        std::vector<double> targetsM_mat_vec;

        std::vector<double> ls_coefficients_multi_vec(maxRegressionDepth*exerciseIndices_vec.size()*numberBasisFunctions);
        CubeFacade<double> regression_coefficients_cube(ls_coefficients_multi_vec,numberOfExerciseDates,
            maxRegressionDepth,numberBasisFunctions);

        Matrix_gold<double> lowerCuts_mat(numberOfExerciseDates,maxRegressionDepth,0.0);
        Matrix_gold<double> upperCuts_mat(numberOfExerciseDates,maxRegressionDepth,0.0);

        Cube_gold<double> means_cube_gold(numberOfExerciseDates,numberOfExtraRegressions+1, basisVariableExample_gold<double>::maxVariablesPerStep( ),0.0);
        Cube_gold<double> sds_cube_gold(numberOfExerciseDates,numberOfExtraRegressions+1, basisVariableExample_gold<double>::maxVariablesPerStep( ),0.0);

        std::vector<int> basisVariablesPerIndex(numberOfExerciseDates);
        for (int s=0; s < numberOfExerciseDates; ++s)
            basisVariablesPerIndex[s]=  basisVariableExample_gold<double>::actualNumberOfVariables(exerciseIndices_vec[s], alive_vec[exerciseIndices_vec[s]],number_rates);

        std::vector<double> some_numeraireValues_vec(exerciseIndices_vec.size()*totalPaths);

        MatrixFacade<double> some_numeraireValues_matrix(&some_numeraireValues_vec[0],numberOfExerciseDates,totalPaths);

        spot_measure_numeraires_extraction_gold(   spot_measure_values_vec,
            totalPaths,
            numberSteps,
            numberOfExerciseDates, 
            deflation_locations_vec,
            some_numeraireValues_vec //output
            );
        Matrix_gold<double> exercise_values_zero_matrix(numberOfExerciseDates,totalPaths,0.0);

        double lsestM = generateRegressionCoefficientsViaLSMultiQuadratic_gold(numberOfExerciseDates,
            productsM_cube_vec,
            targetsM_mat_vec,
            regression_coefficients_cube,
            lowerCuts_mat.Facade(),
            upperCuts_mat.Facade(),
            means_cube_gold.Facade(),
            sds_cube_gold.Facade(),
            normalise,
            basisFunctionVariables_gold_cube,
            basisVariablesPerIndex,
            basisVariableExample_gold<double>::maxVariablesPerStep( ),
            aggregatedFlows_matrix, // deflated to current exercise time
            exercise_values_zero_matrix.Facade(), // deflated to current exercise time
            some_numeraireValues_matrix,
            //         deflation_locations_vec,
            totalPaths,
            numberOfExtraRegressions,
            lowerPathCutoff,
            regressionSelector,
            useCrossTerms);

        debugDumpCube(regression_coefficients_cube,"regression_coefficients_cube");

        std::vector<double> batchValuesMGold(numberSecondPassBatches);
        std::vector<double> flow1_vec(pathsPerSecondPassBatch*number_rates);
        std::vector<double> flow2_vec(pathsPerSecondPassBatch*number_rates);

        earlyExerciseNullGold<double> exerciseValue;

        //CubeConstFacade<float> forwards2_cube(&evolved_rates_gpu_vec[0],numberSteps,number_rates,pathsPerSecondPassBatch);


        Swap_gold<double> product( numberSteps, strike,payReceive,  accruals1, accruals2 );

        exerciseIndicatorsbool_vec.resize(exerciseIndicators_vec.size());
        for (size_t i=0; i < exerciseIndicators_vec.size(); ++i)
            exerciseIndicatorsbool_vec[i] = exerciseIndicators_vec[i] != 0;

        std::vector<double> AndersenShifts_vec(numberOfExerciseDates,0.0);


        // multi gold version
        std::vector<double> flow1_multi_vec(pathsPerSecondPassBatch*numberSteps);
        std::vector<double> flow2_multi_vec(pathsPerSecondPassBatch*numberSteps);

        std::vector<double> discountedFlows_M_vec(pathsPerSecondPassBatch*numberSteps);
        std::vector<double> summedDiscountedFlows1_M_vec(pathsPerSecondPassBatch);
        std::vector<double> summedDiscountedFlows2_M_vec(pathsPerSecondPassBatch);

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


        // change containers to work for second pass but only if necessary
        if (pathsPerSecondPassBatch!=pathsPerBatch)
        {
            evolved_rates_vec.resize(pathsPerSecondPassBatch*numberSteps*number_rates);       
            evolved_log_rates_vec.resize(pathsPerSecondPassBatch*numberSteps*number_rates);   
            discounts_vec.resize(pathsPerSecondPassBatch*numberSteps*(number_rates+1));
            annuities_vec.resize(pathsPerSecondPassBatch*numberSteps*number_rates);
            swap_rates_vec.resize(pathsPerSecondPassBatch*numberSteps*number_rates);
            rate1_vec.resize(pathsPerSecondPassBatch*numberSteps);
            rate2_vec.resize(pathsPerSecondPassBatch*numberSteps);
            rate3_vec.resize(pathsPerSecondPassBatch*numberSteps);
            rate1_cf_vec.resize(pathsPerSecondPassBatch*numberSteps);
            rate2_cf_vec.resize(pathsPerSecondPassBatch*numberSteps);
            rate3_cf_vec.resize(pathsPerSecondPassBatch*numberSteps);
        }
        CubeConstFacade<double>  forwards_s_cube(evolved_rates_vec,numberSteps,number_rates,pathsPerSecondPassBatch);
        CubeConstFacade<double>  discountRatios_s_cube(discounts_vec,numberSteps,number_rates+1,pathsPerSecondPassBatch);
        MatrixConstFacade<double>	rate1_s_matrix(rate1_vec,numberSteps,pathsPerSecondPassBatch);
        MatrixConstFacade<double>	rate2_s_matrix(rate2_vec,numberSteps,pathsPerSecondPassBatch);
        MatrixConstFacade<double>	rate3_s_matrix(rate3_vec,numberSteps,pathsPerSecondPassBatch);

        Cube_gold<double> batch_basis_variables_cube(
            numberOfExerciseDates,
            maxBasisVariables,pathsPerSecondPassBatch,0.0);

        std::vector<double> spot_measure_values_s_vec(pathsPerSecondPassBatch*number_rates);



        for (int secondBatch=0; secondBatch < numberSecondPassBatches; ++secondBatch)
        {
            int pathOffset = duplicate*pathsPerBatch*numberOfBatches+secondBatch*pathsPerSecondPassBatch;

            // first develop LMM paths using PC algorithm
            LMMevolver.getPaths(pathsPerSecondPassBatch,
                pathOffset,scrambler_vec,
                evolved_rates_vec,
                evolved_log_rates_vec
                );

            discount_ratios_computation_gold( evolved_rates_vec, 
                taus_vec, 
                alive_vec, 
                pathsPerSecondPassBatch,
                numberSteps, 
                number_rates, 
                discounts_vec  // for output 
                );

            coterminal_annuity_ratios_computation_gold(  discounts_vec, 
                taus_vec, 
                alive_vec, 
                pathsPerSecondPassBatch,
                numberSteps, 
                number_rates, 
                annuities_vec  // for output 
                );

            // use discount ratios and swap annuities to compute swap rates
            coterminal_swap_rates_computation_gold(   discounts_vec, 
                annuities_vec, 
                alive_vec, 
                pathsPerSecondPassBatch,
                numberSteps, 
                number_rates, 
                swap_rates_vec // for output 
                );

            // get the values of rate1 from forward rates
            forward_rate_extraction_selecting_steps_gold(  evolved_rates_vec, 
                extractor1,     
                exerciseIndices_vec,
                pathsPerSecondPassBatch,
                number_rates, 
                numberSteps,
                rate1_vec                 
                );

            // get the values of rate2 from swap rates
            forward_rate_extraction_selecting_steps_gold(  swap_rates_vec, 
                extractor2,     
                exerciseIndices_vec,
                pathsPerSecondPassBatch,
                number_rates, 
                numberSteps,
                rate2_vec                 
                );

            // get the values of rates from annuities
            forward_rate_extraction_selecting_steps_gold(  annuities_vec, 
                extractor3,     
                exerciseIndices_vec,
                pathsPerSecondPassBatch,
                number_rates, 
                numberSteps,
                rate3_vec                 
                );

            // get rate1 for cash flows, this need not be the same as rate1 for basis
            forward_rate_extraction_gold(  evolved_rates_vec, 
                extractor1_cf,                          
                pathsPerSecondPassBatch,
                numberSteps,
                number_rates, 
                rate1_cf_vec                
                );


            // get rate2 for cash flows, this need not be the same as rate2 for basis
            forward_rate_extraction_gold(  evolved_rates_vec, 
                extractor2_cf,                          
                pathsPerSecondPassBatch,
                numberSteps,
                number_rates, 
                rate2_cf_vec                
                );

            // get rate3 for cash flows, this need not be the same as rate3 for basis

            forward_rate_extraction_gold(  evolved_rates_vec, 
                extractor3_cf,                          
                pathsPerSecondPassBatch,
                numberSteps,
                number_rates, 
                rate3_cf_vec                
                );


            adjoinBasisVariablesCaller_gold(useLogBasis,
                integerData_vec, // int data specific to the basis variables
                floatData_vec, // float data specific to the basis variables
                forwards_s_cube,
                discountRatios_s_cube,
                rate1_s_matrix,
                rate2_s_matrix,
                rate3_s_matrix,
                pathsPerSecondPassBatch,
                pathsPerSecondPassBatch, 
                0,
                number_rates, 
                exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
                exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices      
                batch_basis_variables_cube.Facade()// output location
                );


            if (useCrossTerms)
            {
                cashFlowGeneratorEE_gold(
                    flow1_multi_vec,
                    flow2_multi_vec, 
                    product,
                    exerciseValue,
                    exMCrossStrategy,
                    pathsPerSecondPassBatch, 
                    numberSteps,
                    exerciseIndicatorsbool_vec,
                    rate1_cf_vec, 
                    rate2_cf_vec, 
                    rate3_cf_vec, 
                    batch_basis_variables_cube.ConstFacade(),
                    evolved_rates_vec, 
                    discounts_vec);
            }
            else
            {
                cashFlowGeneratorEE_gold(
                    flow1_multi_vec,
                    flow2_multi_vec, 
                    product,
                    exerciseValue,
                    exMStrategy,
                    pathsPerSecondPassBatch, 
                    numberSteps,
                    exerciseIndicatorsbool_vec,
                    rate1_cf_vec, 
                    rate2_cf_vec, 
                    rate3_cf_vec, 
                    batch_basis_variables_cube.ConstFacade(),
                    evolved_rates_vec, 
                    discounts_vec);

            }

            spot_measure_numeraires_computation_offset_gold(  discounts_vec,
                pathsPerSecondPassBatch,
                pathsPerSecondPassBatch,
                0,
                number_rates,
                spot_measure_values_s_vec
                );


            cashFlowDiscounting_gold(firstIndex_vec, 
                secondIndex_vec,
                thetas_vec, 
                discounts_vec, 
                flow1_multi_vec, 
                spot_measure_values_s_vec,
                pathsPerSecondPassBatch, 
                numberSteps, 
                discountedFlows_M_vec, // output
                summedDiscountedFlows1_M_vec // output
                ); 



            cashFlowDiscounting_gold(firstIndex_vec, 
                secondIndex_vec,
                thetas_vec, 
                discounts_vec, 
                flow2_multi_vec, 
                spot_measure_values_s_vec,
                pathsPerSecondPassBatch, 
                numberSteps, 
                discountedFlows_M_vec, // output
                summedDiscountedFlows2_M_vec// output
                ); 


            double batchSumMGold = std::accumulate(summedDiscountedFlows1_M_vec.begin(),summedDiscountedFlows1_M_vec.end(),0.0)
                +std::accumulate(summedDiscountedFlows2_M_vec.begin(),summedDiscountedFlows2_M_vec.end(),0.0);

            batchValuesMGold[secondBatch] =firstDf*batchSumMGold/ pathsPerSecondPassBatch;
        }

        double mean = std::accumulate(batchValuesMGold.begin(),batchValuesMGold.end(),0.0)/batchValuesMGold.size();
        std::cout << " second pass batch values are, \n";  
        for (size_t i=0; i < batchValuesMGold.size(); ++i)
        {
            std::cout << ", " << batchValuesMGold[i];
        }

        std::cout << " \nsecond pass batch mean is,"<<mean << " \n";  
        std::cout << " \nfirst pass estimate is,"<<lsestM*firstDf << " \n";  

        upperStartTime =clock();

        // now suppose we want to find an upper bound

        // first we need to generate some new outer paths

        Cube_gold<double> thirdPassVariates(numberSteps,factors,upperPaths);

        MersenneTwisterUniformRng rng(upperSeed);

        rng.populateCubeWithNormals<CubeFacade<double>,double>(thirdPassVariates.Facade());

        // change containers to work for second pass but only if necessary
        if (pathsPerSecondPassBatch!=upperPaths)
        {
            evolved_rates_vec.resize(upperPaths*numberSteps*number_rates);       
            evolved_log_rates_vec.resize(upperPaths*numberSteps*number_rates);   
            discounts_vec.resize(upperPaths*numberSteps*(number_rates+1));
            annuities_vec.resize(upperPaths*numberSteps*number_rates);
            swap_rates_vec.resize(upperPaths*numberSteps*number_rates);
            rate1_vec.resize(upperPaths*numberSteps);
            rate2_vec.resize(upperPaths*numberSteps);
            rate3_vec.resize(upperPaths*numberSteps);
            rate1_cf_vec.resize(upperPaths*numberSteps);
            rate2_cf_vec.resize(upperPaths*numberSteps);
            rate3_cf_vec.resize(upperPaths*numberSteps);
            flow1_multi_vec.resize(upperPaths*numberSteps);
            flow2_multi_vec.resize(upperPaths*numberSteps);
        }
        CubeConstFacade<double>  forwards_u_cube(evolved_rates_vec,numberSteps,number_rates,upperPaths);
        CubeConstFacade<double>  discountRatios_u_cube(discounts_vec,numberSteps,number_rates+1,upperPaths);
        MatrixConstFacade<double>	rate1_u_matrix(rate1_vec,numberSteps,upperPaths);
        MatrixConstFacade<double>	rate2_u_matrix(rate2_vec,numberSteps,upperPaths);
        MatrixConstFacade<double>	rate3_u_matrix(rate3_vec,numberSteps,upperPaths);
        MatrixConstFacade<double>	rate1_cf_u_matrix(rate1_cf_vec,numberSteps,upperPaths);
        MatrixConstFacade<double>	rate2_cf_u_matrix(rate2_cf_vec,numberSteps,upperPaths);
        MatrixConstFacade<double>	rate3_cf_u_matrix(rate3_cf_vec,numberSteps,upperPaths);

        Cube_gold<double> batch_basis_variables_u_cube(
            numberOfExerciseDates,
            maxBasisVariables,upperPaths,0.0);

        std::vector<double> spot_measure_values_u_vec(upperPaths*numberSteps);
        MatrixConstFacade<double> spot_measure_values_u_mat(spot_measure_values_u_vec,numberSteps,upperPaths);

        LMM_evolver_pc_class_gold<double> upperEvolver(rates_vec,
            taus_vec,
            pseudosDouble.getDataVector(),
            displacements_vec, 
            factors,
            numberSteps, 
            number_rates,
            cutoffLevel);


        upperEvolver.generateRatesFromUncorrelated( thirdPassVariates.getDataVector(),
            upperPaths,
            evolved_rates_vec,
            evolved_log_rates_vec);

        // we now have the outer paths for the upper sim, extract data

        discount_ratios_computation_gold( evolved_rates_vec , taus_vec,   alive_vec, 
            upperPaths, numberSteps,    number_rates,     discounts_vec );
        coterminal_annuity_ratios_computation_gold(  discounts_vec,  taus_vec, alive_vec, upperPaths,
            numberSteps,number_rates, annuities_vec );
        coterminal_swap_rates_computation_gold(   discounts_vec,     annuities_vec,   alive_vec,  upperPaths,  numberSteps,   number_rates, 
            swap_rates_vec  );
        forward_rate_extraction_selecting_steps_gold(  evolved_rates_vec, extractor1, exerciseIndices_vec,upperPaths,number_rates, numberSteps,
            rate1_vec);
        forward_rate_extraction_selecting_steps_gold(  swap_rates_vec,  extractor2,     exerciseIndices_vec,upperPaths,number_rates, numberSteps,
            rate2_vec  );
        forward_rate_extraction_selecting_steps_gold(  annuities_vec,  extractor3,     
            exerciseIndices_vec,upperPaths,number_rates, numberSteps, rate3_vec    );
        forward_rate_extraction_gold(  evolved_rates_vec,    extractor1_cf,   upperPaths,
            numberSteps, number_rates,   rate1_cf_vec   );
        forward_rate_extraction_gold(  evolved_rates_vec,  extractor2_cf, upperPaths,
            numberSteps,  number_rates,    rate2_cf_vec);
        forward_rate_extraction_gold(  evolved_rates_vec, extractor3_cf, upperPaths,
            numberSteps,  number_rates, rate3_cf_vec);
        adjoinBasisVariablesCaller_gold(useLogBasis,integerData_vec, floatData_vec, 
            forwards_u_cube,discountRatios_u_cube,rate1_u_matrix,rate2_u_matrix,rate3_u_matrix,
            upperPaths,upperPaths, 0,number_rates, 
            exerciseIndices_vec, exerciseIndicators_vec, 
            batch_basis_variables_u_cube.Facade()// output location
            );

        Matrix_gold<double> regressedContinuation_mat(numberSteps,upperPaths,0.0);

        if (useCrossTerms)
        {
            cashFlowGeneratorEEToEnd_gold(
                flow1_multi_vec,
                flow2_multi_vec, 
                exerciseValues_mat.getDataVector(),
                exerciseOccurences_mat.getDataVector(),
                product,
                exerciseValue,
                exMCrossStrategy,
                upperPaths, 
                numberSteps,
                exerciseIndicatorsbool_vec,
                rate1_cf_vec, 
                rate2_cf_vec, 
                rate3_cf_vec, 
                batch_basis_variables_u_cube.ConstFacade(),
                evolved_rates_vec, 
                discounts_vec,
                getRegressions,
                regressedContinuation_mat.Facade());
        }
        else
        {
            cashFlowGeneratorEEToEnd_gold(
                flow1_multi_vec,
                flow2_multi_vec, 
                exerciseValues_mat.getDataVector(),
                exerciseOccurences_mat.getDataVector(),
                product,
                exerciseValue,
                exMStrategy,
                upperPaths, 
                numberSteps,
                exerciseIndicatorsbool_vec,
                rate1_cf_vec, 
                rate2_cf_vec, 
                rate3_cf_vec, 
                batch_basis_variables_u_cube.ConstFacade(),
                evolved_rates_vec, 
                discounts_vec,
                getRegressions,
                regressedContinuation_mat.Facade());


        }

   //     if (getRegressions)
     //       debugDumpMatrix(regressedContinuation_mat.ConstFacade(),"regressedContinuation_mat");

        spot_measure_numeraires_computation_offset_gold(  discounts_vec,
            upperPaths,
            upperPaths,
            0,
            number_rates,
            spot_measure_values_u_vec
            );

        std::vector<double> discountedFlows1_o_vec,discountedFlows2_o_vec;

        cashFlowDiscounting_gold(firstIndex_vec, 
            secondIndex_vec,
            thetas_vec, 
            discounts_vec, 
            flow1_multi_vec, 
            spot_measure_values_u_vec,
            upperPaths, 
            numberSteps, 
            discountedFlows1_o_vec, // output
            summedDiscountedFlows1_M_vec // output
            ); 



        cashFlowDiscounting_gold(firstIndex_vec, 
            secondIndex_vec,
            thetas_vec, 
            discounts_vec, 
            flow2_multi_vec, 
            spot_measure_values_u_vec,
            upperPaths, 
            numberSteps, 
            discountedFlows2_o_vec, // output
            summedDiscountedFlows2_M_vec// output
            ); 

        MatrixConstFacade<double> discountedFlows1_o_mat(&discountedFlows1_o_vec[0],numberSteps,upperPaths);
        MatrixConstFacade<double> discountedFlows2_o_mat(&discountedFlows2_o_vec[0],numberSteps,upperPaths);



        // we now have the cash-flows sizes and exercise times for the outer paths

        // we need to set to the vectors to write the data into for each inner simulation
        // we go for a fixed size matrices and cubes
        // the data is simply written into from the distinguished step onwards

        //    Cube_gold<double> innerPaths_cube(numberSteps,number_rates,upperPaths,0.0);

        std::vector<double> evolved_rates_i_vec(pathsPerSubsim*numberSteps*number_rates);  
        CubeFacade<double> evolved_rates_i_cube(evolved_rates_i_vec,numberSteps,number_rates,pathsPerSubsim);

        std::vector<double> evolved_log_rates_i_vec(pathsPerSubsim*numberSteps*number_rates);   
        std::vector<double> discounts_i_vec(pathsPerSubsim*numberSteps*(number_rates+1));
        std::vector<double> annuities_i_vec(pathsPerSubsim*numberSteps*number_rates);
        std::vector<double> swap_rates_i_vec(pathsPerSubsim*numberSteps*number_rates);
        std::vector<double> rate1_i_vec(pathsPerSubsim*numberSteps);
        std::vector<double> rate2_i_vec(pathsPerSubsim*numberSteps);
        std::vector<double> rate3_i_vec(pathsPerSubsim*numberSteps);
        std::vector<double> rate1_cf_i_vec(pathsPerSubsim*numberSteps);
        std::vector<double> rate2_cf_i_vec(pathsPerSubsim*numberSteps);
        std::vector<double> rate3_cf_i_vec(pathsPerSubsim*numberSteps);
        std::vector<double> numeraire_values_i_vec(pathsPerSubsim*numberSteps);
        std::vector<double> flow1_i_vec(pathsPerSubsim*numberSteps);
        std::vector<double> flow2_i_vec(pathsPerSubsim*numberSteps);

        std::vector<double> discountedFlows1_i_vec;
        std::vector<double> discountedFlows2_i_vec;
        std::vector<double> summedDiscountedFlows1_i_vec;
        std::vector<double> summedDiscountedFlows2_i_vec;


        CubeConstFacade<double> discountRatios_i_cube(discounts_i_vec,numberSteps,number_rates+1,pathsPerSubsim);
        MatrixConstFacade<double> rate1_i_matrix(rate1_i_vec,numberSteps,pathsPerSubsim);
        MatrixConstFacade<double> rate2_i_matrix(rate2_i_vec,numberSteps,pathsPerSubsim);
        MatrixConstFacade<double> rate3_i_matrix(rate3_i_vec,numberSteps,pathsPerSubsim);

        Cube_gold<double> subsim_basis_variables_cube(
            numberOfExerciseDates,
            maxBasisVariables,pathsPerSubsim,0.0);

        LMMEvolverPartial_gold<double> innerEvolver(  pseudosDouble,
            numberSteps,
            number_rates,
            factors,
            displacements_vec,
            taus_vec,
            alive_vec,
            rates_vec
            );

        std::vector<double> innerVariates_vec(numberSteps*factors*pathsPerSubsim);
        double* innerVariates_ptr=&innerVariates_vec[0];
        std::vector<double> step_rates_vec(number_rates);

        std::vector<double> flow1_inner_vec(pathsPerSubsim*numberSteps);
        std::vector<double> flow2_inner_vec(pathsPerSubsim*numberSteps);

        //    debugDumpCube(forwards_u_cube,"forwards_u_cube");


        for (int outerPath=0; outerPath<upperPaths; ++outerPath)
        {
            int s;
            int e=0;
            for ( s=1; s < numberSteps;++s) // s represents steps already done, 
            {
                if (exerciseIndicatorsbool_vec[s-1])
                {
                    ++e;

                    CubeFacade<double> variates_cube(innerVariates_ptr,pathsPerSubsim,numberSteps-s,factors);
                    CubeConstFacade<double> variates_const_cube(innerVariates_ptr,pathsPerSubsim,numberSteps-s,factors);

                    rng.populateCubeWithNormals<CubeFacade<double>, double>(variates_cube);

                    for (int r=0; r < number_rates;++r)
                        step_rates_vec[r]=forwards_u_cube(s-1,r,outerPath); 

                    // paths are written into the cube starting at step s
                    innerEvolver.conditionallyGenerate(s,step_rates_vec,variates_const_cube,pathsPerSubsim,0  // offset is zero
                        ,evolved_rates_i_cube,cutOffLevel);


    //                if (outerPath==0)
      //                  debugDumpCube(evolved_rates_i_cube,"evolved_rates_i_cube");

                    // ok got inner paths, now need inner cash-flows

                    // we need to extract rates etc as usual, issue is purely around what step to start at 

                    discount_ratios_computation_gold( evolved_rates_i_vec , taus_vec,   alive_vec, 
                        pathsPerSubsim, numberSteps,    number_rates,     discounts_i_vec ,s);

                    coterminal_annuity_ratios_computation_gold(  discounts_i_vec,  taus_vec, alive_vec, pathsPerSubsim,
                        numberSteps,number_rates, annuities_i_vec ,s );

                    coterminal_swap_rates_computation_gold(   discounts_i_vec,     annuities_i_vec,   alive_vec,  pathsPerSubsim,  numberSteps,  
                        number_rates, swap_rates_i_vec, s  );

                    forward_rate_extraction_selecting_steps_gold(  evolved_rates_i_vec, extractor1, exerciseIndices_vec,
                        pathsPerSubsim,number_rates, numberSteps, rate1_i_vec,e);

      //              if (outerPath==0)
     //                   debugDumpMatrix(rate1_i_matrix,"rate1_i_matrix");

                    forward_rate_extraction_selecting_steps_gold(  swap_rates_i_vec,  extractor2,    
                        exerciseIndices_vec,pathsPerSubsim,number_rates, numberSteps, rate2_i_vec ,e  );

                    forward_rate_extraction_selecting_steps_gold(  annuities_i_vec,  extractor3,     
                        exerciseIndices_vec,pathsPerSubsim,number_rates, numberSteps, rate3_i_vec   ,e );

                    forward_rate_extraction_gold(  evolved_rates_i_vec, extractor1_cf,   pathsPerSubsim,
                        numberSteps, number_rates,   rate1_cf_i_vec  ,s  );

                    forward_rate_extraction_gold(  evolved_rates_i_vec, extractor2_cf, pathsPerSubsim,
                        numberSteps,  number_rates,    rate2_cf_i_vec,s);

                    forward_rate_extraction_gold(  evolved_rates_i_vec, extractor3_cf, pathsPerSubsim,
                        numberSteps,  number_rates, rate3_cf_i_vec,s);

                    adjoinBasisVariablesCaller_gold<double>(useLogBasis,integerData_vec, floatData_vec, 
                        evolved_rates_i_cube,discountRatios_i_cube,rate1_i_matrix,rate2_i_matrix,rate3_i_matrix,
                        pathsPerSubsim,pathsPerSubsim, 0,number_rates, 
                        exerciseIndices_vec, exerciseIndicators_vec, 
                        subsim_basis_variables_cube.Facade(),// output location
                        s);

           //         if (outerPath==0)
        //                debugDumpCube(subsim_basis_variables_cube.ConstFacade(),"subsim_basis_variables_cube");


                    if (useCrossTerms)
                    {
                        cashFlowGeneratorEEConditional_gold(
                            flow1_i_vec, flow2_i_vec,  product,  exerciseValue,  exMCrossStrategy,
                            upperPaths,  numberSteps,exerciseIndicatorsbool_vec, rate1_cf_vec,  rate2_cf_vec, 
                            rate3_cf_vec, subsim_basis_variables_cube.ConstFacade(),
                            evolved_rates_vec, // unconditional data
                            discounts_vec, 
                            outerPath,// conditional data
                            pathsPerSubsim, s, rate1_cf_i_vec,   rate2_cf_i_vec, rate3_cf_i_vec,
                            evolved_rates_i_vec, discounts_i_vec);
                    }
                    else
                    {
                        cashFlowGeneratorEEConditional_gold(
                            flow1_i_vec, flow2_i_vec,  product,  exerciseValue,  exMStrategy,
                            upperPaths,  numberSteps,exerciseIndicatorsbool_vec, rate1_cf_vec,  rate2_cf_vec, 
                            rate3_cf_vec, subsim_basis_variables_cube.ConstFacade(),
                            evolved_rates_vec, // unconditional data
                            discounts_vec, 
                            outerPath,// conditional data
                            pathsPerSubsim, s, rate1_cf_i_vec,   rate2_cf_i_vec, rate3_cf_i_vec,
                            evolved_rates_i_vec, discounts_i_vec);

                    }
                    // ok we have the cash-flows
                    // we need their discounted value 

                    // produces numeraires along the path,
                    // value of numeraire at t_{startStep} is 1
                    spot_measure_numeraires_computation_gold(  discounts_i_vec,
                        pathsPerSubsim,
                        number_rates,
                        numeraire_values_i_vec, //output
                        s
                        );

                    // discounts and divides by numeraire values corresponding to numeraire being worth 1 at 
                    // t_{startStep}
                    cashFlowDiscounting_gold(firstIndex_vec, 
                        secondIndex_vec,
                        thetas_vec, 
                        discounts_i_vec, 
                        flow1_i_vec, 
                        numeraire_values_i_vec,
                        pathsPerSubsim, 
                        numberSteps, 
                        discountedFlows1_i_vec, // output
                        summedDiscountedFlows1_i_vec,// output
                        s
                        ); 

                    cashFlowDiscounting_gold(firstIndex_vec, 
                        secondIndex_vec,
                        thetas_vec, 
                        discounts_i_vec, 
                        flow2_i_vec, 
                        numeraire_values_i_vec,
                        pathsPerSubsim, 
                        numberSteps, 
                        discountedFlows2_i_vec, // output
                        summedDiscountedFlows2_i_vec,// output
                        s
                        ); 

                    double sum=0.0;
                    double sumsq=0.0;
                    double n=  spot_measure_values_u_mat(s,outerPath);

                    double y = discountedFlows1_o_mat(s-1,outerPath) +  discountedFlows2_o_mat(s-1,outerPath);
                    y *= firstDf;

                    for (int p=0; p < pathsPerSubsim;++p)
                    {
                        // we use the outer money market account to discount to time t_0 and then first dF to get to time zero
                        double x= firstDf*(summedDiscountedFlows1_i_vec[p]+summedDiscountedFlows2_i_vec[p])/n;

                        // we also have an additional flow fixed at t_{s-1}
                        x+=y;

                        innerPathsValues(outerPath,s,p) = x;

                        sum += x;
                        sumsq += x*x;
                    }
                    int no= pathsPerSubsim;
                    double mean = sum/no;
                    double sampleVariance = ((sumsq/no - mean*mean)*no/(no-1));
                    double sampleSe = safesqrt(sampleVariance/no);
                    means_i_mat(outerPath,s-1)=mean;
                    sds_i_mat(outerPath,s-1) = sampleSe;
                }
            }
            double y = discountedFlows1_o_mat(s-1,outerPath) +  discountedFlows2_o_mat(s-1,outerPath);
            y *= firstDf;

            means_i_mat(outerPath,s-1)=y;
            sds_i_mat(outerPath,s-1) = 0;
        }
/*        if (getRegressions)
        {
            debugDumpMatrix(means_i_mat.ConstFacade(),"means_i_mat");
            debugDumpMatrix(sds_i_mat.ConstFacade(),"sds_i_mat");
            debugDumpMatrix(exerciseOccurences_mat.ConstFacade(),"exerciseOccurences_mat");
            debugDumpCube(innerPathsValues.ConstFacade(),"innerPathsValues");
        }
        */

        // exerciseIndicators_mat
        // means_i_mat
        // sds_i_mat

    }

    double meanU,se;
    ABDualityGapEstimateZeroRebate( exerciseOccurences_mat,
        means_i_mat,
        sds_i_mat,
        meanU,
        se,
        exerciseIndicatorsbool_vec);

    std::cout << "Mean and se of duality gap , " << meanU << "," << se <<"\n";

    for (int pathsToUse=minPathsToUse; pathsToUse <= pathsPerSubsim; ++pathsToUse)
    {
        for (int outerPath =0; outerPath < upperPaths; ++outerPath)
        {
            for (int stepsDone=1; stepsDone<numberSteps; ++stepsDone)
                if (exerciseIndicatorsbool_vec[stepsDone-1])
                {
                    double sum=0.0;
                    double sumsq=0.0;
                    for (int p=0; p < pathsToUse;++p)
                    {
                        double x= innerPathsValues(outerPath,stepsDone,p);
                        sum += x;
                        sumsq += x*x;
                    }
                    int no = pathsToUse;
                    double mean = sum/no;
                    double sampleVariance = ((sumsq/no - mean*mean)*no/(no-1));
                    double sampleSe = safesqrt(sampleVariance/no);
                    means_i_mat(outerPath,stepsDone-1)=mean;
                    sds_i_mat(outerPath,stepsDone-1) = sampleSe;

                }
        }


        /*     ABDualityGapEstimateZeroRebate( exerciseOccurences_mat,
        means_i_mat,
        sds_i_mat,
        meanU,
        se,
        exerciseIndicatorsbool_vec);
        */         
        double meanU2,se2,meanU3,se3,meanU4,se4;


        ABDualityGapEstimateZeroRebateBiasEstimation( exerciseOccurences_mat,
            means_i_mat,
            sds_i_mat,
            meanU,
            se,
            meanU2,
            se2,
            meanU3,
            se3,
            meanU4,
            se4,
            gaussianPaths,
            innerEstSeed,
            exerciseIndicatorsbool_vec,
            numberOfExerciseDates);


        std::cout <<pathsToUse << ", Mean and se of duality gap , " << meanU << "," << se<<"," <<meanU2 << "," << se2
            <<"," <<meanU3 << "," << se3<< "," <<meanU4 << "," << se4<<"\n";

        int upperTimeTicks = clock()-upperStartTime;

        std::cout << "time spent in upper bound code " << upperTimeTicks/static_cast<double>(CLOCKS_PER_SEC) << "\n";
    }


}

