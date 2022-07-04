//
//
//       Bermudan IRD Pricer.cu
//
//
// (c) Mark Joshi 2011,2012,2013
// This code is released under the GNU public licence version 3

// routine to test the LS Code
#include "bermudanIRDPricer.h"
#include <LS_Basis_main.h>
#include <cutil.h>
#include <gold/LS_Basis_gold.h>
#include <cutil_inline.h>
#include <cutil.h>
#include <gold/Bridge_gold.h>
#include <LMM_evolver_main.h>
#include <multid_path_gen_BS_main.h>
#include <gold/LMM_evolver_gold.h>
#include <early_exercise_value_generator_main.h>
#include <gold/early_exercise_value_generator_gold.h>
#include <cashFlowDiscounting_main.h>
#include <gold/cashFlowDiscounting_gold.h>
#include <gold/ExerciseIndices_gold.h>
#include <cashFlowGeneration_product_main.h>

#include<cashFlowGen_earlyEx_product_main.h>
#include <gold/cashFlowGeneration_product_gold.h>
#include <gold/cashFlowDiscounting_gold.h>
#include <gold/cashFlowAggregation_gold.h>
#include <cashFlowAggregation_main.h>

#include <gold/LS_regression.h>
#include <LS_main.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <InverseCumulatives.h>
#include <gold/Bridge_gold.h>
#include <sobol.h>
#include <Bridge.h>
#include <correlate_drift_main.h>
#include <cudaMacros.h>
#include <gold/MatrixFacade.h> 

#include <gold/early_exercise_value_generator_gold.h>
#include <gold/cashFlowGeneration_earlyEx_product.h>
#include <Utilities.h>
#include <ComparisonCheck.h>

#include <gold/math/fp_utilities.h>
#include <numeric>
#include <gold/math/pseudoSquareRoot.h>
#include <LMM_evolver_full.h>
#include <gold/volstructs_gold.h>
#include <LS_main_cula.h>
#include <RegressionSelectorConcrete.h>
#include <smallFunctions.h>
#include <gold/Timers.h>
#include <LinearSolverConcrete_main.h>
#include <curand.h>

/*
namespace
{
class setLowestBit
{
unsigned int y_;
public:

__host__ __device__ setLowestBit(unsigned int y)
{
y_=y;

}


__host__ __device__ unsigned int operator()(unsigned int x)
{
return x || 1;
}
};

};
*/

double BermudanMultiLSPricerExample(int pathsPerBatch,
                                    int numberOfBatches, 
                                    int numberOfSecondPassBatches, 
                                    bool useCrossTerms,
                                    int regressionDepth,
                                    float sdsForCutOff,
                                    int minPathsForRegression,
                                    float lowerFrac,
                                    float upperFrac,
                                    float multiplier,
                                    bool globallyNormalise,
                                    int duplicate, // if zero use same paths as for first pass, if 1 start at the next paths
                                    bool useLog,
                                    int numberOfRates,
                                    float firstForward,
                                    float forwardIncrement,
                                    float displacement,
                                    float strike,
                                    double beta,
                                    double L,
                                    double a, 
                                    double b,
                                    double c, 
                                    double d,
                                    int numberNonCallCoupons,
                                    double firstRateTime,
                                    double rateLength,
                                    double payReceive_d,
                                    bool useFlatVols,
                                    bool annulNonCallCoupons,
                                    double initialNumeraireValue,
                                    bool globalDiscounting,
                                    bool verbose,
                                    int gpuChoice,
                                    int LMMthreads,
                                    bool scrambleFirst,
                                    bool scrambleSecond
                                    )
{   
    double result=-1.0;

    if (verbose)
    {
        std:: cout << "pathsPerBatch, "<< pathsPerBatch <<",numberOfBatches , " << numberOfBatches <<",numberOfSecondPassBatches, " << numberOfSecondPassBatches<<",useCrossTerms, " << useCrossTerms<<",regressionDepth, " << regressionDepth << ",sdsForCutOff, "

            << sdsForCutOff
            <<", minPathsForRegression, " <<				    minPathsForRegression
            <<", lowerFrac," <<					lowerFrac
            <<", upperFrac, " <<				    upperFrac
            <<", multiplier, " <<					multiplier
            <<", globallyNormalise, " <<				    globallyNormalise
            <<", use log, " <<				    useLog
            <<", numberOfRates, " <<				    numberOfRates
            <<", duplicate, " <<				    duplicate 
            <<",	strike	," <<  strike
            <<",\n	beta, "<<	 beta
            <<",	L, " <<	 L
            <<",	a, " <<	 a
            <<",	b, " <<	 b
            <<",	c, " <<	 c 
            <<",	d, " <<	 d
            <<",	numberNonCallCoupons, " <<	 numberNonCallCoupons
            <<",	firstRateTime, " <<	 firstRateTime
            <<",	rateLength	, " << rateLength
            <<",		payReceive_d, " << payReceive_d 
            <<",		useFlatVols, " << useFlatVols 
            <<", annual, " << annulNonCallCoupons
            <<", global discounting, " << globalDiscounting
            <<", scrambleFirst, " << scrambleFirst
            <<", scrambleSecond, " << scrambleSecond
            << "\n";}


     int numberOfExtraRegressions=regressionDepth-1;

    int n_vectors =  pathsPerBatch;

    int n_poweroftwo =1;
    {
        int r=2;

        while (r < numberOfRates)
        {
            r*=2;
            ++n_poweroftwo;
        }
    }
    //	int n_poweroftwo =5;

    int numberSteps = numberOfRates;
    //int numberSteps = intPower(2,n_poweroftwo);
    int number_rates = numberSteps;
    int factors = min(number_rates,5);

    int tot_dimensions = numberSteps*factors;


    int numberOfSteps = numberSteps;

    int basisVariablesPerStep = basisVariableLog_gold<float>::maxVariablesPerStep();

    bool useShared = false;

    //	bool useLog =true;

    float accrual =static_cast<float>(rateLength);
    float payReceive = static_cast<float>(payReceive_d);

    //	std::cout << " about to set device\n";

    if (gpuChoice == -1)
        gpuChoice = cutGetMaxGflopsDeviceId();

    int t0 = clock();


    cudaSetDevice(gpuChoice);

    size_t N= n_vectors*tot_dimensions; 
    size_t outN = n_vectors*numberSteps*number_rates;

    size_t estimatedFloatsRequired = 8*(2*outN+2*N);

    size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

    //  bool change=false;

    ConfigCheckForGPU checker;

    float excessMem=2.0f;

    while (excessMem >0.0f)
    {
        checker.checkGlobalMem( estimatedMem, excessMem );

        if (excessMem >0.0f)
        {
            n_vectors/=2;
            N= n_vectors*tot_dimensions;
            outN =  n_vectors*numberSteps*number_rates;
            //             change =true;
            estimatedFloatsRequired = 8*(2*outN+2*N);
            estimatedMem = sizeof(float)*estimatedFloatsRequired;

            std::cout<< "halving batch size to maintain memory space in BermudanMultiLSPricerExample \n";

        }
    }

          pathsPerBatch =n_vectors;
      int totalPaths = pathsPerBatch*numberOfBatches;
 

    //	std::cout << " set device\n";
    cudaThreadSynchronize();

    int tpt5=clock();

    //	std::cout << " thread synced\n";

    // set up all device vectors first
    // let's put everything inside a little scope so it all destroys earlier
    {
        std::vector<int> exerciseIndices_vec; // the indices of the exercise times amongst the evolution times
        std::vector<int> exerciseIndicators_vec(numberSteps,0);
        for (int i=numberNonCallCoupons; i < numberSteps;++i) // 
        {
            exerciseIndicators_vec[i] =1;
        }

        //   exerciseIndicators_vec[1] =0;


        GenerateIndices(exerciseIndicators_vec, exerciseIndices_vec);

        int numberOfExerciseDates = static_cast<int>(exerciseIndices_vec.size());

        int totalNumberOfBasisVariables = basisVariablesPerStep*numberOfExerciseDates*totalPaths;

        // across batch data
        thrust::device_vector<float> final_basis_variables_device(totalNumberOfBasisVariables);

        thrust::device_vector<float> spot_measure_values_device(numberOfSteps*totalPaths);




        thrust::device_vector<int> alive_device;   
        int steps_to_test  = numberSteps; // must be less than or equal to numberSteps


        thrust::host_vector<int> exerciseIndices_host(exerciseIndices_vec.begin(),exerciseIndices_vec.end());
        thrust::host_vector<int> exerciseIndicators_host(exerciseIndicators_vec.begin(),exerciseIndicators_vec.end());


        thrust::device_vector<int> exerciseIndices_device(exerciseIndices_host);
        thrust::device_vector<int> exerciseIndicators_device(exerciseIndicators_host);

        int numberExerciseDates =  static_cast<int>(exerciseIndices_vec.size());
        int totalNumberOfExerciseValues =numberExerciseDates*totalPaths;
        thrust::device_vector<float> exercise_values_device(totalNumberOfExerciseValues);




        //	double beta_d = static_cast<double>(beta);
        //	double L_d = static_cast<double>(L);
        //		double vol_d = 0.11;

        std::vector<double> vols_d(number_rates,d);

        std::vector<int> aliveIndex(numberSteps);
        for (int i=0; i < numberSteps; ++i)
            aliveIndex[i] = i;

        thrust::device_vector<float> evolved_rates_device(outN);

        // dummy calibration data 
        std::vector<float> logRates_vec(number_rates);
        std::vector<float> rates_vec(number_rates);
        std::vector<float> taus_vec(number_rates);
        std::vector<float> displacements_vec(number_rates);
        std::vector<float> rateTimes_vec(number_rates+1);
        std::vector<float> evolutionTimes_vec(number_rates);

        std::vector<double> rateTimes_d_vec(number_rates+1);
        std::vector<double> evolutionTimes_d_vec(number_rates);

        rateTimes_vec[0] =static_cast<float>(firstRateTime);
        rateTimes_d_vec[0] =firstRateTime;

        for (int i=0; i < number_rates; ++i)
        {
            rates_vec[i] = firstForward+i*forwardIncrement;
            displacements_vec[i] =displacement;
            logRates_vec[i] = logf(rates_vec[i]+displacements_vec[i] );
            taus_vec[i] = static_cast<float>(accrual);
            rateTimes_d_vec[i+1] = firstRateTime+accrual*(i+1);
            rateTimes_vec[i+1]= static_cast<float>(rateTimes_d_vec[i+1]);
            evolutionTimes_d_vec[i] = rateTimes_d_vec[i]; 
            evolutionTimes_vec[i] =static_cast<float>( evolutionTimes_d_vec[i]);

        }   


        float maxDisplacement = *std::max_element(displacements_vec.begin(),displacements_vec.end());

        thrust::host_vector<float> taus_host(taus_vec.begin(),taus_vec.end());
        thrust::device_vector<float> taus_device(taus_host);

        thrust::host_vector<float> displacements_host(displacements_vec.begin(),displacements_vec.end());
        thrust::device_vector<float> displacements_device(displacements_host);
        std::vector<float> evolved_rates_gpu_vec(evolved_rates_device.size());

        thrust::device_vector<float> discounts_device(n_vectors*steps_to_test*(number_rates+1));

        // cash-flow aggregation data

        thrust::device_vector<float> aggregatedFlows_device(totalPaths*exerciseIndices_vec.size());

        // create new scope so that everything inside dies at end of it

        {
            std::vector<double> multipliers_vec(numberOfRates,1.0);
            std::vector<double> rateStarts_d_vec(rateTimes_d_vec.begin(),rateTimes_d_vec.end()-1);



            /*
            Cube_gold<double> pseudosDouble(FlatVolPseudoRootsOfCovariances(rateTimes_d_vec,
            evolutionTimes_d_vec,
            vols_d,
            factors,
            L_d,
            beta_d
            ));
            */

            Cube_gold<double> pseudosDouble( useFlatVols ?
                FlatVolPseudoRootsOfCovariances(rateTimes_d_vec,
                evolutionTimes_d_vec,
                vols_d,
                factors,
                L,
                beta
                )
                :
            ABCDLBetaPseudoRoots(a,b,c,d,
                evolutionTimes_d_vec,
                rateStarts_d_vec,
                multipliers_vec,
                factors,
                L,
                beta
                ));

            Cube_gold<float> pseudos(CubeTypeConvert<float,double>(pseudosDouble));

            //		debugDumpCube<double>(pseudosDouble.ConstFacade(),"pseudos");


            thrust::device_vector<float> A_device(deviceVecFromCube(pseudos));
            thrust::host_vector<int> alive_host(aliveIndex.begin(),aliveIndex.end());
            alive_device=alive_host;





            std::vector<float> drifts(numberSteps*number_rates);
            MatrixFacade<float> drifts_mat(&drifts[0],numberSteps,number_rates);
            for (int i=0; i < static_cast<int>(drifts.size()); ++i)
            {
                for (int s=0; s < numberSteps; ++s)
                    for (int r=0; r < number_rates; ++r)
                    {  
                        float x = 0.0f;
                        for (int f=0; f< factors; ++f)
                            x += pseudos(s,r,f)*pseudos(s,r,f);

                        drifts_mat(s,r) = -0.5f*x;

                    }
            }


            thrust::host_vector<float> drifts_host(drifts.begin(),drifts.end());
            thrust::device_vector<float> drifts_device(drifts_host);



            thrust::host_vector<float> logRates_host(logRates_vec.begin(),logRates_vec.end());
            thrust::device_vector<float> logRates_device(logRates_host);

            thrust::host_vector<float> rates_host(rates_vec.begin(),rates_vec.end());
            thrust::device_vector<float> rates_device(rates_host);


            thrust::device_vector<float> dev_output(N);    
            thrust::device_vector<float> device_input(N);
            thrust::device_vector<unsigned int> SobolInts_buffer_device(N);

            thrust::device_vector<float> e_buffer_device(factors*n_vectors);
            thrust::device_vector<float> e_pred_buffer_device(factors*n_vectors);

            thrust::device_vector<float> initial_drifts_device(number_rates);

            {

                spotDriftComputer<float> driftComp(taus_vec,  factors,displacements_vec);
                std::vector<float> initial_drifts_vec(number_rates);

                driftComp.getDrifts(pseudos[0], 
                    rates_vec,
                    initial_drifts_vec);

                thrust::host_vector<float> initialDrifts_host(initial_drifts_vec.begin(),initial_drifts_vec.end());
                initial_drifts_device = initialDrifts_host;

            }
            // rates for exercise values
            thrust::device_vector<float>  rate1_device(n_vectors*numberOfExerciseDates); // one rate per exercise date per path
            thrust::device_vector<float>  rate2_device(n_vectors*numberOfExerciseDates);
            thrust::device_vector<float>  rate3_device(n_vectors*numberOfExerciseDates);



            std::vector<float> rate1_vec(n_vectors*numberOfExerciseDates);
            std::vector<float> rate2_vec(n_vectors*numberOfExerciseDates);
            std::vector<float> rate3_vec(n_vectors*numberOfExerciseDates);


            std::vector<int> extractor1(numberOfExerciseDates);
            for (int i=0; i < static_cast<int>(extractor1.size()); ++i)
                extractor1[i] = exerciseIndices_vec[i];

            std::vector<int> extractor2(numberOfExerciseDates);
            for (int i=0; i < static_cast<int>(extractor2.size()); ++i)
                extractor2[i] = std::min( exerciseIndices_vec[i]+1,steps_to_test-1);

            std::vector<int> extractor3(numberOfExerciseDates);
            for (int i=0; i < static_cast<int>(extractor3.size()); ++i)
                extractor3[i] = steps_to_test-1;

            // we want co-terminal swap-rates and their annuities and their swap-rates for Berm test
            thrust::device_vector<float> annuities_device(n_vectors*steps_to_test*number_rates);
            thrust::device_vector<float> swap_rates_device(n_vectors*steps_to_test*number_rates);

            // rates for cash-flow generations
            thrust::device_vector<float>  rate1_cf_device(n_vectors*steps_to_test); // one rate per step per path
            thrust::device_vector<float>  rate2_cf_device(n_vectors*steps_to_test);
            thrust::device_vector<float>  rate3_cf_device(n_vectors*steps_to_test);




            std::vector<int> extractor1_cf(steps_to_test);
            for (int i=0; i < static_cast<int>(extractor1_cf.size()); ++i)
                extractor1_cf[i] = i;

            std::vector<int> extractor2_cf(steps_to_test);
            for (int i=0; i < static_cast<int>(extractor2_cf.size()); ++i)
                extractor2_cf[i] = std::min( i+1,steps_to_test-1);

            std::vector<int> extractor3_cf(steps_to_test);
            for (int i=0; i < static_cast<int>(extractor3_cf.size()); ++i)
                extractor3_cf[i] = std::min( i+2,steps_to_test-1);



            thrust::device_vector<int> integer_Data_device(1); //  data specific to the basis variables
            thrust::device_vector<float> float_Data_device(3,maxDisplacement);
            float_Data_device[2] =0.0f;


            thrust::device_vector<int> integer_Data_evalue_device(1); //  data specific to the exercise value
            thrust::device_vector<float> float_Data_evalue_device(1); 

            float_Data_evalue_device[0] =  strike;

            thrust::device_vector<float> dev_correlated_rates(outN);

            thrust::device_vector<float> evolved_log_rates_device(outN);


            // discounting tests require us to have the target indices set up 
            // and payment times

            std::vector<int> genTimeIndex_vec(numberSteps);
            for (size_t i=0;i< genTimeIndex_vec.size(); ++i)
                genTimeIndex_vec[i] = static_cast<int>(i);

            thrust::device_vector<int> genTimeIndex_device(deviceVecFromStlVec(genTimeIndex_vec));

            std::vector<int> stepToExerciseIndices_vec;

            int firstPositiveStep = findExerciseTimeIndicesFromPaymentTimeIndices(genTimeIndex_vec, 
                exerciseIndices_vec,
                exerciseIndicators_vec, // redundant info but sometimes easier to work with
                stepToExerciseIndices_vec
                );
            thrust::host_vector<int> stepToExerciseIndices_host(stepToExerciseIndices_vec.begin(), stepToExerciseIndices_vec.end());
            thrust::device_vector<int> stepToExerciseIndices_device(stepToExerciseIndices_host);

            std::vector<float> paymentTimes_vec(numberSteps);
            for (size_t i=0;i< paymentTimes_vec.size(); ++i)
                paymentTimes_vec[i] = rateTimes_vec[i+1];

            std::vector<int> firstIndex_vec, secondIndex_vec;
            std::vector<float> thetas_vec;

            generateCashFlowIndicesAndWeights<float>( firstIndex_vec, 
                secondIndex_vec,
                thetas_vec,
                rateTimes_vec,
                paymentTimes_vec
                );

            thrust::device_vector<float> discounted_flows1_device(n_vectors*steps_to_test);

            thrust::device_vector<float> discounted_flows2_device(n_vectors*steps_to_test);


            thrust::device_vector<int> firstIndex_device(deviceVecFromStlVec(firstIndex_vec));
            thrust::device_vector<int> secondIndex_device(deviceVecFromStlVec(secondIndex_vec));
            thrust::device_vector<float> thetas_device(deviceVecFromStlVec(thetas_vec));


            // set up swap product data

            thrust::device_vector<float> genFlows1_device(n_vectors*steps_to_test);
            thrust::device_vector<float> genFlows2_device(genFlows1_device.size());


            std::vector<float> aux_swap_data_vec(2*number_rates+2);


            aux_swap_data_vec[0] = strike;
            aux_swap_data_vec[1] = payReceive;
            std::vector<float> accruals1(number_rates);
            std::vector<float> accruals2(number_rates);

            for (int j=0; j < number_rates; ++j)
            {
                float x=accrual;
                if (j < numberNonCallCoupons && annulNonCallCoupons)
                    x=0.0f;

                aux_swap_data_vec[j+2] =x;
                accruals1[j] = aux_swap_data_vec[j+2];
                aux_swap_data_vec[number_rates+j+2] =x;
                accruals2[j] =  aux_swap_data_vec[number_rates+j+2];
            }

            thrust::device_vector<float> aux_swap_data_device(aux_swap_data_vec.begin(),aux_swap_data_vec.end());
            thrust::device_vector<float> spot_measure_values_small_device(n_vectors*steps_to_test);
            std::vector<int> deflation_locations_vec(exerciseIndices_vec);

            if (globalDiscounting)
                std::fill(deflation_locations_vec.begin(),deflation_locations_vec.end(),0);

            thrust::device_vector<int> deflation_locations_device(deflation_locations_vec.begin(),deflation_locations_vec.end());


            thrust::device_vector<unsigned int> scrambler_device(tot_dimensions,0);
            //	thrust::device_vector<unsigned int> scrambler2_device(tot_dimensions,0);

            std::vector<int> fromGenIndicesToExerciseDeflations_vec(genTimeIndex_vec.size());
            for (size_t i=0; i < genTimeIndex_vec.size();++i)
            {

                int j=stepToExerciseIndices_vec[i];

                fromGenIndicesToExerciseDeflations_vec[i] = j>=0 ? deflation_locations_vec[j] :0 ;
            }
            thrust::host_vector<int> fromGenIndicesToExerciseDeflations_host(fromGenIndicesToExerciseDeflations_vec.begin(),
                fromGenIndicesToExerciseDeflations_vec.end());

            thrust::device_vector<int> fromGenIndicesToExerciseDeflations_device(fromGenIndicesToExerciseDeflations_host);

            unsigned int* scrambler_global = thrust::raw_pointer_cast(&scrambler_device[0]);

            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

            /* Set seed */
            curandSetPseudoRandomGeneratorSeed(gen, 1UL);




            // every is now set up so can do  loops 


            bool useSharedMem = false;
            bool doDiscounts =true;
            bool newBridge =true;
            bool keplerArch = false;
            bool useTextures=true;

            double timeOnEvolutionsFirst = 0.0;
            double timeOnAnnuitiesFirst = 0.0;
            double forwardRateExtractionFirst =0.0;
            double adjoinFirst =0.0;
            double spotMeasureFirst =0.0;
            double cashFlowGenFirst=0.0;
            double	aggregateFirst =0.0;
            double spotNumerairesFirst =0.0;
            double discountingFirst =0.0;

            cudaThreadSynchronize();

            int pathFirstMultiplier = scrambleFirst ? 0 : 1;
            int pathSecondMultiplier = scrambleSecond ? 0 : 1;
            int t1=clock();

            for (int batchCounter=0; batchCounter < numberOfBatches; ++batchCounter)
            {
                int path_offset= batchCounter * pathsPerBatch*pathFirstMultiplier;


                if (scrambleFirst)
                {
                    curandGenerate(gen, scrambler_global, tot_dimensions);

                    //		setLowestBit tmp(1);

                    //			thrust::transform(scrambler_device.begin(),scrambler_device.end(),scrambler2_device.begin(),tmp);
                }

                Timer h1;



                LMMEvolutionRoutineRawSingleKernel(pathsPerBatch, 
                    path_offset, 
                    number_rates, 
                    factors, 
                    number_rates, 
                    n_poweroftwo,
                    scrambler_device, 
                    A_device,
                    drifts_device, 
                    displacements_device,
                    rates_device, 
                    logRates_device, 
                    taus_device, 
                    initial_drifts_device, 
                    aliveIndex, 
                    alive_device, 
                    // buffers
                    SobolInts_buffer_device, 
                    device_input, 
                    dev_output, 
                    thrust::raw_pointer_cast(&dev_correlated_rates[0]), 
                    thrust::raw_pointer_cast(&e_buffer_device[0]),
                    thrust::raw_pointer_cast(&e_pred_buffer_device[0]),
                    thrust::raw_pointer_cast(&evolved_rates_device[0]),
                    thrust::raw_pointer_cast(&evolved_log_rates_device[0]),
                    thrust::raw_pointer_cast(&discounts_device[0]),
                    useSharedMem,
                    doDiscounts,
                    newBridge,
                    keplerArch,
                    LMMthreads
                    );


                cudaThreadSynchronize();

                if (!doDiscounts)
                {
                    discount_ratios_computation_main(  evolved_rates_device, 
                        taus_device, 
                        aliveIndex, 
                        alive_device, 
                        pathsPerBatch,
                        number_rates, 
                        number_rates, 
                        discounts_device,  // for output 
                        true //bool allStepsAtOnce
                        );

                }

                timeOnEvolutionsFirst+=h1.timePassed();

     //           debugDumpCube(evolved_rates_device,"evolved_rates_device",steps_to_test,number_rates,n_vectors);
                // set up rate vectors 

                Timer h2;

                coterminal_annuity_ratios_computation_gpu(  discounts_device, 
                    taus_device, 
                    aliveIndex, 
                    n_vectors,
                    steps_to_test, 
                    number_rates, 
                    annuities_device  // for output 
                    );

                coterminal_swap_rates_computation_main_gpu(  discounts_device, 
                    annuities_device, 
                    aliveIndex, 
                    n_vectors,
                    steps_to_test, 
                    number_rates, 
                    swap_rates_device  // for output 
                    );


                cudaThreadSynchronize();

                timeOnAnnuitiesFirst +=h2.timePassed();

                Timer h3;

                forward_rate_extraction_steps_gpu_main(  evolved_rates_device, 
                    extractor1,    
                    exerciseIndices_vec,
                    n_vectors,
                    number_rates, 
                    rate1_device                
                    );

                forward_rate_extraction_steps_gpu_main(  swap_rates_device, 
                    extractor2,        
                    exerciseIndices_vec,
                    n_vectors,
                    number_rates, 
                    rate2_device                
                    );

                forward_rate_extraction_steps_gpu_main(  annuities_device, 
                    extractor3,    
                    exerciseIndices_vec,
                    n_vectors,
                    number_rates, 
                    rate3_device                
                    );

                cudaThreadSynchronize();


                forward_rate_extraction_gpu_main(  evolved_rates_device, 
                    extractor1_cf,                          
                    n_vectors,
                    steps_to_test,
                    number_rates, 
                    rate1_cf_device                
                    );

                forward_rate_extraction_gpu_main(  evolved_rates_device, 
                    extractor2_cf,                          
                    n_vectors,
                    steps_to_test,
                    number_rates, 
                    rate2_cf_device                
                    );

                forward_rate_extraction_gpu_main(  evolved_rates_device, 
                    extractor3_cf,                          
                    n_vectors,
                    steps_to_test,
                    number_rates, 
                    rate3_cf_device                
                    );

                cudaThreadSynchronize();
                forwardRateExtractionFirst += h3.timePassed();

                Timer h4;

                adjoinBasisVariablesCaller_main(useLog,
                    integer_Data_device, // data specific to the basis variables
                    float_Data_device,
                    evolved_rates_device,
                    discounts_device,
                    rate1_device,
                    rate2_device,
                    rate3_device,
                    pathsPerBatch,
                    totalPaths, 
                    pathsPerBatch*batchCounter,
                    numberOfRates, 
                    static_cast<int>(exerciseIndices_vec.size()),
                    exerciseIndices_device, // the indices of the exercise times amongst the evolution times
                    exerciseIndicators_device, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                    final_basis_variables_device // output location
                    );

                cudaThreadSynchronize();

                adjoinFirst += h4.timePassed();

                Timer h5;

                // spot measure numeraire

                spot_measure_numeraires_computation_gpu_offset_main(  discounts_device,
                    spot_measure_values_device, //output
                    pathsPerBatch,
                    totalPaths,
                    pathsPerBatch*batchCounter,
                    numberOfRates
                    );



                cudaThreadSynchronize();

                spotMeasureFirst += h5.timePassed();

                Timer h6;

                cashFlowGeneratorCallerSwap_main(  genFlows1_device, 
                    genFlows2_device, 
                    aux_swap_data_device, 
                    pathsPerBatch, 
                    numberSteps,
                    rate1_cf_device, 
                    rate2_cf_device, 
                    rate3_cf_device, 
                    evolved_rates_device, 
                    discounts_device);


                cudaThreadSynchronize();

                cashFlowGenFirst += h6.timePassed();

                //			debugDumpMatrix(genFlows1_device,"genFlows1_device",numberSteps,pathsPerBatch);

                //			debugDumpMatrix(genFlows2_device,"genFlows2_device",numberSteps,pathsPerBatch);

                Timer h7;




                spot_measure_numeraires_computation_gpu_offset_main(  discounts_device,
                    spot_measure_values_small_device, //output
                    pathsPerBatch,
                    pathsPerBatch, // only want this batch
                    0, // no offset
                    numberOfRates
                    );

                cudaThreadSynchronize();

                spotNumerairesFirst += h7.timePassed();


                Timer h8;

                cashFlowDiscounting_partial_gpu_main(genTimeIndex_device, // indices of times at which flows are generated
                    firstIndex_device, // rate time index leq payment date 
                    secondIndex_device, // rate time index > payment date 
                    thetas_device, // interpolation fraction 
                    fromGenIndicesToExerciseDeflations_device, 
                    discounts_device, 
                    genFlows1_device, 
                    spot_measure_values_small_device,
                    pathsPerBatch, 
                    numberSteps, 
                    discounted_flows1_device);

                cashFlowDiscounting_partial_gpu_main(genTimeIndex_device, // indivces of times at which flows are generated
                    firstIndex_device, // rate time index leq payment date 
                    secondIndex_device, // rate time index > payment date 
                    thetas_device, // interpolation fraction 
                    fromGenIndicesToExerciseDeflations_device, 
                    discounts_device, 
                    genFlows2_device, 
                    spot_measure_values_small_device,
                    pathsPerBatch, 
                    numberSteps, 
                    discounted_flows2_device);

                cudaThreadSynchronize();


                discountingFirst += h8.timePassed();

                Timer h9;


                AggregateFlows_main(
                    aggregatedFlows_device,// output added to not overwritten
                    totalPaths,
                    static_cast<int>(exerciseIndices_vec.size()), 
                    discounted_flows1_device,
                    pathsPerBatch, 
                    pathsPerBatch*batchCounter,
                    numberSteps, 
                    stepToExerciseIndices_device     ,firstPositiveStep                    );

                AggregateFlows_main(
                    aggregatedFlows_device,// output added to not overwritten
                    totalPaths,
                    static_cast<int>(exerciseIndices_vec.size()), 
                    discounted_flows2_device,
                    pathsPerBatch, 
                    pathsPerBatch*batchCounter,
                    numberSteps, 
                    stepToExerciseIndices_device     ,firstPositiveStep                        );


                cudaThreadSynchronize();

                aggregateFirst += h9.timePassed();


            }//  end of   for (int batchCounter=0; batchCounter < numberOfBatches; ++batchCounter)

            std::cout << " finished loop...\n";


            std::cout << "Evolved rates \n";


            int t2=clock();


            std::vector<int> basisVariablesPerIndex(exerciseIndices_vec.size());
            for (int s=0; s < static_cast<int>(exerciseIndices_vec.size()); ++s)
                basisVariablesPerIndex[s]=   basisVariableExample_gold<float>::actualNumberOfVariables(exerciseIndices_vec[s], aliveIndex[exerciseIndices_vec[s]],number_rates);



            thrust::device_vector<float> some_numeraireValues_device(exerciseIndices_vec.size()*totalPaths);

            Timer spotExtractionTimer;

            spot_measure_numeraires_extraction_main(  spot_measure_values_device,
                some_numeraireValues_device, //output
                totalPaths,
                totalPaths,
                0,
                numberSteps,
                static_cast<int>(exerciseIndices_vec.size()), 
                exerciseIndices_device
                );

            cudaThreadSynchronize();

            double extractTime = spotExtractionTimer.timePassed();

            int maxBasisVariables = basisVariableLog_gold<float>::actualNumberOfVariables(0,0,numberOfRates);



            std::vector<float> variableShiftsToSubtract_vec(numberExerciseDates*(numberOfExtraRegressions+1)*maxBasisVariables,0.0f);
            std::vector<float> variableDivisors_vec(numberExerciseDates*(numberOfExtraRegressions+1)*maxBasisVariables,1.0f);

            CubeFacade<float> variableShiftsToSubtract_cube(variableShiftsToSubtract_vec,numberExerciseDates,numberOfExtraRegressions+1,maxBasisVariables);
            CubeFacade<float> variableDivisors_cube(variableDivisors_vec,numberExerciseDates,numberOfExtraRegressions+1,maxBasisVariables);


            if (globallyNormalise)
            {
                int layerSize = totalPaths*maxBasisVariables;
                // let's rescale our basis variables to improve stability
                for (int i=0; i < numberExerciseDates; ++i)
                {
                    for (int j=0; j < basisVariablesPerIndex[i]; ++j)
                    {

                        thrust::device_vector<float>::iterator start =final_basis_variables_device.begin()+i*layerSize + j*totalPaths;
                        thrust::device_vector<float>::iterator end = start+totalPaths;
                        float sum = thrust::reduce(start,end);
                        float mean = sum/totalPaths;



                        float sumsq = thrust::transform_reduce(start,end,squareof(),0.0f,thrust::plus<float>());


                        float sd = sqrt( sumsq/(totalPaths) - mean*mean);

                        thrust::transform(start,end,start,shiftAndMult(-mean,1.0f/sd));

                        for (int k=0; k < numberOfExtraRegressions+1; ++k)
                        {
                            variableShiftsToSubtract_cube(i,k,j) = mean;
                            variableDivisors_cube(i,k,j) = sd;


                        }



                    }

                }
            }



            int numberBasisFunctions = quadraticPolynomialCrossGenerator(basisVariableLog_gold<float>::maxVariablesPerStep( )).numberDataPoints();

            thrust::host_vector<int> basisVariablesEachStep_host(basisVariablesPerIndex.begin(),basisVariablesPerIndex.end());
            thrust::device_vector<int> basisVariablesEachStep_device(basisVariablesEachStep_host);

            thrust::device_vector<int> basisVariableInt_data_device(numberExerciseDates,0);
            thrust::device_vector<float> basisVariableFloat_data_device(numberExerciseDates,0.0f);
            thrust::device_vector<float> coefficients_device(numberExerciseDates*numberBasisFunctions);
            std::vector<Realv> coefficients_vec(coefficients_device.size());

            thrust::device_vector<float> exercise_values_zero_float_device(numberExerciseDates*totalPaths,0.0f);


            thrust::device_vector<float> lowercuts_device(numberExerciseDates*(numberOfExtraRegressions+1));
            thrust::device_vector<float> uppercuts_device(numberExerciseDates*(numberOfExtraRegressions+1));


            //	RegressionSelectorStandardDeviations regressionSelector(sdsForCutOff);

            RegressionSelectorFraction regressionSelector(lowerFrac, upperFrac,sdsForCutOff, multiplier);

  //          LinearSolverHandCraftedAndCula solve(1,1,0,0); // dummies

       	LinearSolverHandCraftedAndGold solve(1,1,0,0);


            int t2pt5=clock();

            double LSest3  
                = generateRegressionCoefficientsViaLSMultiquadratic_flexi_main(numberExerciseDates,
                basisVariableInt_data_device,
                basisVariableFloat_data_device,
                coefficients_device, // the LS coefficients are placed here 
                coefficients_vec, // the LS coefficients are also placed here 
                lowercuts_device, // the lower cut points for the regressions 
                uppercuts_device, // the upper cut points for the regressions 									
                final_basis_variables_device, // cube of preextracted values for variables of basis funcions
                basisVariablesEachStep_device, //vector of the number of basis variables for each step
                basisVariablesPerIndex, //vector of the number of basis variables for each step should agree with basisVariablesEachStep_global
                maxBasisVariables,
                aggregatedFlows_device, // deflated to current exercise time, matrix of times and paths 
                exercise_values_zero_float_device, // deflated to current exercise time, matrix of times and paths 
                some_numeraireValues_device, // numeraire vals for each, matrix of times and paths 
                //	deflation_locations_vec,
                totalPaths,
                useCrossTerms,
                numberOfExtraRegressions,regressionSelector,
                minPathsForRegression,solve
                );


            int t3=clock();

            double timeInSolver = solve.timeSpent();
            std::cout << "time in solver " << timeInSolver << "\n";
            std::cout << " time in selector " << regressionSelector.timeInHere() << "\n";

            // first passes are done
            // we still need to do the second passes!

            int pathsPerSecondPassBatch = pathsPerBatch; 
            //	int duplicate=1; // if zero use same paths as for first pass, if 1 start at the next paths

            std::vector<float> batchValuesGold(numberOfSecondPassBatches);
            std::vector<float> batchValuesGPU_vec(numberOfSecondPassBatches);
            thrust::device_vector<float> batch_basis_variables_device(pathsPerSecondPassBatch*exerciseIndices_vec.size()*maxBasisVariables);

            thrust::device_vector<float> discountedFlows_dev(pathsPerSecondPassBatch*numberSteps);
            thrust::device_vector<float> summedDiscountedFlows1_dev(pathsPerSecondPassBatch);
            thrust::device_vector<float> summedDiscountedFlows2_dev(pathsPerSecondPassBatch);

            thrust::device_vector<float> spot_measure_values_2_device(pathsPerSecondPassBatch*numberOfRates);



            std::vector<bool> exerciseIndicatorsbool_vec(exerciseIndicators_vec.size());
            for (size_t i=0; i < exerciseIndicators_vec.size(); ++i)
                exerciseIndicatorsbool_vec[i] = exerciseIndicators_vec[i] != 0;




            thrust::host_vector<bool> exerciseIndicatorsbool_host(exerciseIndicatorsbool_vec.begin(),exerciseIndicatorsbool_vec.end());
            thrust::device_vector<bool> exerciseIndicatorsbool_dev(exerciseIndicatorsbool_host);

            thrust::device_vector<float> exerciseValueDataFloatdummy_device(1,0.0f);
            thrust::device_vector<int> exerciseValueDataIntdummy_device(1,0);

            std::vector<float> AndersenShifts_vec(numberExerciseDates,0.0f);


            int floatDataCrossSize;
            int intDataCrossSize;


            LSAMultiExerciseStrategyQuadraticCross_gpu::outputDataVectorSize(floatDataCrossSize,
                intDataCrossSize, 
                numberOfExtraRegressions+1,
                static_cast<int>(exerciseIndices_vec.size()),
                basisVariablesPerIndex);


            thrust::device_vector<float>  exerciseStrategyDataCrossFloat_device( floatDataCrossSize);                                 
            thrust::device_vector<int>  exerciseStrategyDataCrossInt_device( intDataCrossSize);     

            LSAMultiExerciseStrategyQuadraticCross_gpu::outputDataVectors(exerciseStrategyDataCrossFloat_device.begin(), 
                exerciseStrategyDataCrossInt_device.begin(),
                numberOfExtraRegressions+1,
                static_cast<int>(exerciseIndices_vec.size()),
                basisVariablesPerIndex ,
                AndersenShifts_vec,
                variableShiftsToSubtract_vec,
                variableDivisors_vec,
                stlVecFromDevVec(lowercuts_device),
                stlVecFromDevVec(uppercuts_device),
                coefficients_vec);

            int floatDataNonCrossSize;
            int intDataNonCrossSize;


            LSAMultiExerciseStrategyQuadratic_gpu::outputDataVectorSize(floatDataNonCrossSize,
                intDataNonCrossSize, 
                numberOfExtraRegressions+1,
                static_cast<int>(exerciseIndices_vec.size()),
                basisVariablesPerIndex);


            thrust::device_vector<float>  exerciseStrategyDataNonCrossFloat_device( floatDataCrossSize);                                 
            thrust::device_vector<int>  exerciseStrategyDataNonCrossInt_device( intDataCrossSize);     

            LSAMultiExerciseStrategyQuadratic_gpu::outputDataVectors(exerciseStrategyDataNonCrossFloat_device.begin(), 
                exerciseStrategyDataNonCrossInt_device.begin(),
                numberOfExtraRegressions+1,
                static_cast<int>(exerciseIndices_vec.size()),
                basisVariablesPerIndex ,
                AndersenShifts_vec,
                variableShiftsToSubtract_vec,
                variableDivisors_vec,
                stlVecFromDevVec(lowercuts_device),
                stlVecFromDevVec(uppercuts_device),
                coefficients_vec);

            //		debugDumpVector(exerciseStrategyDataFloat_device,"exerciseStrategyDataFloat_device");

            //	debugDumpVector(AndersenShifts_vec,"AndersenShifts_vec");

            //	debugDumpVector(variableShiftsToSubtract_vec,"variableShiftsToSubtract_vec");
            //
            //debugDumpVector(variableDivisors_vec,"variableDivisors_vec");

            //		debugDumpVector(lowercuts_device,"lowercuts_device");
            //
            //		debugDumpVector(uppercuts_device,"uppercuts_device");

            //		debugDumpVector(coefficients_vec,"coefficients_vec");


            // for debugging 
            thrust::device_vector<float> estCValues_device(exerciseIndices_vec.size()*pathsPerSecondPassBatch);

            thrust::device_vector<int>  exerciseIndices_Device(pathsPerSecondPassBatch);

            if (!scrambleSecond && scrambleFirst)
                thrust::fill(scrambler_device.begin(),scrambler_device.end(),0);

        

            for (int k=0; k < numberOfSecondPassBatches; ++k)
            {
                int path_offset= (numberOfBatches * pathsPerBatch*duplicate*pathFirstMultiplier
                    +k*pathsPerSecondPassBatch)*pathSecondMultiplier;

                if (scrambleSecond)
                {
                    curandGenerate(gen, scrambler_global, tot_dimensions);
                    //					setLowestBit tmp(1);			
                    //					thrust::transform(scrambler_device.begin(),scrambler_device.end(),scrambler2_device.begin(),tmp);

                }

                LMMEvolutionRoutineRawSingleKernel(pathsPerBatch, 
                    path_offset, 
                    number_rates, 
                    factors, 
                    number_rates, 
                    n_poweroftwo,
                    scrambler_device, 
                    A_device,
                    drifts_device, 
                    displacements_device,
                    rates_device, 
                    logRates_device, 
                    taus_device, 
                    initial_drifts_device, 
                    aliveIndex, 
                    alive_device, 
                    // buffers
                    SobolInts_buffer_device, 
                    device_input, 
                    dev_output, 
                    thrust::raw_pointer_cast(&dev_correlated_rates[0]), 
                    thrust::raw_pointer_cast(&e_buffer_device[0]),
                    thrust::raw_pointer_cast(&e_pred_buffer_device[0]),
                    thrust::raw_pointer_cast(&evolved_rates_device[0]),
                    thrust::raw_pointer_cast(&evolved_log_rates_device[0]),
                    thrust::raw_pointer_cast(&discounts_device[0]),
                    useSharedMem,
                    doDiscounts,
                    newBridge,
                    keplerArch,
                    LMMthreads
                    );
                cudaThreadSynchronize();

                if (!doDiscounts)
                {
                    bool allStepsAtOnce =true;

                    discount_ratios_computation_main( evolved_rates_device, 
                        taus_device, 
                        aliveIndex, 
                        alive_device,
                        pathsPerSecondPassBatch,
                        steps_to_test, 
                        number_rates, 
                        discounts_device,
                        allStepsAtOnce  
                        );

                    cudaThreadSynchronize();
                }

                // set up rate vectors 

                coterminal_annuity_ratios_computation_gpu(  discounts_device, 
                    taus_device, 
                    aliveIndex, 
                    n_vectors,
                    steps_to_test, 
                    number_rates, 
                    annuities_device  // for output 
                    );

                coterminal_swap_rates_computation_main_gpu(  discounts_device, 
                    annuities_device, 
                    aliveIndex, 
                    n_vectors,
                    steps_to_test, 
                    number_rates, 
                    swap_rates_device  // for output 
                    );

                cudaThreadSynchronize();

                forward_rate_extraction_steps_gpu_main(  evolved_rates_device, 
                    extractor1, 
                    exerciseIndices_vec,
                    n_vectors,
                    number_rates, 
                    rate1_device                
                    );

                forward_rate_extraction_steps_gpu_main(  swap_rates_device, 
                    extractor2,      
                    exerciseIndices_vec,
                    n_vectors,
                    number_rates, 
                    rate2_device                
                    );

                forward_rate_extraction_steps_gpu_main(  annuities_device, 
                    extractor3,       
                    exerciseIndices_vec,
                    n_vectors,
                    number_rates, 
                    rate3_device                
                    );

                cudaThreadSynchronize();



                forward_rate_extraction_gpu_main(  evolved_rates_device, 
                    extractor1_cf,                          
                    n_vectors,
                    numberSteps,
                    number_rates, 
                    rate1_cf_device                
                    );

                forward_rate_extraction_gpu_main(  swap_rates_device, 
                    extractor2_cf,                          
                    n_vectors,
                    numberSteps,
                    number_rates, 
                    rate2_cf_device                
                    );

                forward_rate_extraction_gpu_main(  annuities_device, 
                    extractor3_cf,                          
                    n_vectors,
                    numberSteps,
                    number_rates, 
                    rate3_cf_device                
                    );

                cudaThreadSynchronize();


                adjoinBasisVariablesCaller_main(useLog,
                    integer_Data_device, // data specific to the basis variables
                    float_Data_device,
                    evolved_rates_device,
                    discounts_device,
                    rate1_device,
                    rate2_device,
                    rate3_device,
                    pathsPerSecondPassBatch,
                    pathsPerSecondPassBatch, 
                    0,
                    numberOfRates, 
                    static_cast<int>(exerciseIndices_vec.size()),
                    exerciseIndices_device, // the indices of the exercise times amongst the evolution times
                    exerciseIndicators_device, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                    batch_basis_variables_device // output location
                    );

                //	debugDumpCube(batch_basis_variables_device,"batch_basis_variables_device",exerciseIndices_vec.size(),maxBasisVariables,pathsPerSecondPassBatch);


                spot_measure_numeraires_computation_gpu_offset_main(  discounts_device,
                    spot_measure_values_2_device, //output
                    pathsPerSecondPassBatch,
                    pathsPerSecondPassBatch,
                    0,
                    numberOfRates
                    );




                //		debugDumpVector(exerciseStrategyDataFloat_device,"exerciseStrategyDataFloat_device");
                //
                //				debugDumpVector(exerciseStrategyDataInt_device,"exerciseStrategyDataInt_device");

                //				debugDumpVector(coefficients_vec,"coefficients_vec");

                //				debugDumpVector(lowercuts_device,"lowercuts_device");
                //	
                //				debugDumpVector(uppercuts_device,"uppercuts_device");


                CUT_CHECK_ERR("cashFlowGeneratorEE_main pre-failuer \n");

                if(useCrossTerms)
                {

                    cashFlowGeneratorEE_main<Swap ,earlyExerciseNull ,LSAMultiExerciseStrategyQuadraticCross_gpu >(
                        genFlows1_device,
                        genFlows2_device, 
                        aux_swap_data_device,
                        exerciseValueDataFloatdummy_device,
                        exerciseValueDataIntdummy_device,
                        exerciseStrategyDataCrossFloat_device,
                        floatDataCrossSize,
                        exerciseStrategyDataCrossInt_device,
                        intDataCrossSize,
                        pathsPerSecondPassBatch, 
                        numberSteps,
                        static_cast<int>(exerciseIndices_vec.size()),
                        exerciseIndicatorsbool_dev,
                        rate1_cf_device, 
                        rate2_cf_device, 
                        rate3_cf_device, 
                        batch_basis_variables_device,
                        maxBasisVariables,
                        evolved_rates_device, 
                        discounts_device,
                        estCValues_device, 
                        exerciseIndices_Device);
                }
                else
                {
                    cashFlowGeneratorEE_main<Swap ,earlyExerciseNull ,LSAMultiExerciseStrategyQuadratic_gpu >(
                        genFlows1_device,
                        genFlows2_device, 
                        aux_swap_data_device,
                        exerciseValueDataFloatdummy_device,
                        exerciseValueDataIntdummy_device,
                        exerciseStrategyDataNonCrossFloat_device,
                        floatDataNonCrossSize,
                        exerciseStrategyDataNonCrossInt_device,
                        intDataNonCrossSize,
                        pathsPerSecondPassBatch, 
                        numberSteps,
                        static_cast<int>(exerciseIndices_vec.size()),
                        exerciseIndicatorsbool_dev,
                        rate1_cf_device, 
                        rate2_cf_device, 
                        rate3_cf_device, 
                        batch_basis_variables_device,
                        maxBasisVariables,
                        evolved_rates_device, 
                        discounts_device,
                        estCValues_device, 
                        exerciseIndices_Device);
                }
                CUT_CHECK_ERR("cashFlowGeneratorEE_main failed \n");


                cashFlowDiscounting_gpu_main(firstIndex_device, 
                    secondIndex_device,
                    thetas_device, 
                    discounts_device, 
                    genFlows1_device, 
                    spot_measure_values_2_device,
                    pathsPerSecondPassBatch, 
                    numberSteps, 
                    useTextures,
                    useShared,
                    discountedFlows_dev, // output
                    summedDiscountedFlows1_dev); // output

                cashFlowDiscounting_gpu_main(firstIndex_device, 
                    secondIndex_device,
                    thetas_device, 
                    discounts_device, 
                    genFlows2_device, 
                    spot_measure_values_2_device,
                    pathsPerSecondPassBatch, 
                    numberSteps,
                    useTextures,
                    useShared,
                    discountedFlows_dev, // output
                    summedDiscountedFlows2_dev); // output

                float batchValueGPU = thrust::reduce(summedDiscountedFlows1_dev.begin(),summedDiscountedFlows1_dev.end());
                batchValueGPU += thrust::reduce(summedDiscountedFlows2_dev.begin(),summedDiscountedFlows2_dev.end());
                batchValueGPU /= pathsPerSecondPassBatch;

                batchValuesGPU_vec[k] = static_cast<float>(batchValueGPU*initialNumeraireValue);

                //		debugDumpVector(summedDiscountedFlows1_dev,"summedDiscountedFlows1_dev");
                //	debugDumpVector(summedDiscountedFlows2_dev,"summedDiscountedFlows2_dev");


            }


            float valueGPU = std::accumulate(batchValuesGPU_vec.begin(),batchValuesGPU_vec.end(),0.0f)/numberOfSecondPassBatches;

            result = valueGPU;

            std::cout << " \n\nfirst pass value ,,," << LSest3*initialNumeraireValue << "\n";
            std::cout << " second pass  gpu value ,,," << valueGPU<< "\n";

            std::cout << " number of first pass paths ," << totalPaths << "\n";
            std::cout << " number of second pass paths, " << numberOfSecondPassBatches*pathsPerBatch << "\n";



            std::cout << " second pass  gpu values :\n";

            double sumsq = 0.0;

            for (size_t i=0; i < batchValuesGPU_vec.size(); ++i)
            {
                double x =  batchValuesGPU_vec[i];
                std::cout <<x << ",";
                sumsq+= x*x;
            }
            std::cout << "\n";

            double v = (sumsq/numberOfSecondPassBatches- valueGPU*valueGPU);
            double staderr = sqrt(v/batchValuesGPU_vec.size());

            std::cout << "standard error " << staderr << "\n";

            int t4 = clock();

            double time0 = (tpt5-t0)/static_cast<double>(CLOCKS_PER_SEC);
            double time0pt5 = (t1-tpt5)/static_cast<double>(CLOCKS_PER_SEC);

            double time1 = (t2-t1)/static_cast<double>(CLOCKS_PER_SEC);
            double time2 = (t2pt5-t2)/static_cast<double>(CLOCKS_PER_SEC);
            double time2pt5 = (t3-t2pt5)/static_cast<double>(CLOCKS_PER_SEC);

            double time3 = (t4-t3)/static_cast<double>(CLOCKS_PER_SEC);

            double totalTime = (t4-t1)/static_cast<double>(CLOCKS_PER_SEC);

            std::cout << " time taken for wake up is, " << time0 << "\n";
            std::cout << " time taken for set up is, " << time0pt5 << "\n";

            std::cout << " time taken for first pass paths is, " << time1 << "\n";
            std::cout << " time taken for regression set up is, " << time2 << "\n";
            std::cout << " time taken for regression is, " << time2pt5 << "\n";
            std::cout << " time taken for second pass  is, " << time3 << "\n";

            std::cout << " total time taken is, " << totalTime << "\n";

            if (verbose)
            {
                std::cout << 
                    "extractTime, " << extractTime <<"\n" <<
                    "timeOnEvolutionsFirst, " << timeOnEvolutionsFirst << " , " <<
                    "timeOnAnnuitiesFirst, " <<  timeOnAnnuitiesFirst << " , " <<
                    "forwardRateExtractionFirst, " <<  forwardRateExtractionFirst << " , " << 
                    "adjoinFirst, " << adjoinFirst << " , " << 
                    "spotMeasureFirst, " <<  spotMeasureFirst << " , " << 
                    "cashFlowGenFirst, " <<  cashFlowGenFirst << " , " << 
                    "aggregateFirst, " <<  aggregateFirst  << " , " << 
                    "spotNumerairesFirst, " <<  spotNumerairesFirst  << " , " << 
                    "discountingFirst, " <<  discountingFirst << "\n"; 

            }
        }

    } // outer scope for destruction


    cudaDeviceReset();
    cudaThreadExit();



    return result;

}