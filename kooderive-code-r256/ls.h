
/*
For each batch:

1) generate paths
2) extract basis function variables, or possibly extract rates to feed
into basis function variables 
3) generate cumulative deflated cash-flows between exercise dates,
deflation to exercise date preceding time of determination
a) cash-flow generation
b) cash-flow discounting
c) cash-flow aggregation
4) generate exercise values and discount them to the date of exercise decision

Repeat 1 through 4 until enough paths have been done. Do not store
outputs of 1 across batches. But do do so for 2 to 4. 

5) at final exercise time, for each path 
   a) compute basis function values from basis function variables
   b) compute products of basis fucntion values and continuation values
   c) compute sum products from b) (use doubles and atomics?)
6) compute the basis function weights for predicting the continuation value 
(possibly on CPU)
7) decide exercise decision for each path by computing predicted
continuation value using basis function weights
8) set continutation value at previous time slice to be deflated
cash-flows between the two time slices plus exercise value if exercise
occurs or deflated cash-flows if it does not.



Issues:
A) what do we store under the assumption that the space on the GPU is too
small to store all data from all paths?
B) how do we carry out the deflation?
C) do we store or regenerate basis function values? 

C) if we have N paths with 15 basis function values then this takes
15N floats. This would allow about 1 million paths. alternately,
regenerate
them on the fly, then only need to store the underlying variables. Can
then
do say 3 million paths but takes more time, discounting info 


A) we don't store the paths, we need the cash-flows between exercise
dates, we need the exercise values, we need the basis function
variables, we don't need much else

B) deflation -- we will probably work with discretely compounding MMA
as numeraire, want path indepedence of estimates. 
So 
i) we discount any cash-flow at its determination time, t, 
 to the previous exercise
date, s, where s is max such that s \leq t. 
ii) when writing continuation values backwards we have to discount
them using the realized rates between those two times. 


so we need to store the deflator value between exercise times. If
not every rate time is an exercise time, how do we define this? 
Either as the ratio of the discretely compounded MM at the two
times. Or as the DF of the second time as observed from the first
time. These are not the same. 

if we store the discretely compounded MM for each exercise time then
the first is easy enough. So we will do this. 

For deflated cash-flows and exercise values, we discount them to the time
of determination and then deflate to the previous exercise time. 


We will assume that the evolution times of the simulation are the reset times 
of the forward rates. Anything else is too fiddly. 

We organize all data with pathnumber being the first dimension. 
Other index being the second.  Step number being the third dimension. 

so
pathNumber + numberPaths*otherIndex+numberPaths*numberOther*step

For this material that means number paths is the number of paths across all the batches of
forward rate generation. This is NOT the number of paths in a single batch. 

We will write one kernel for each microstage initially. This will faciliate debugging and testing. 

Cash-flows determined before first exercise date will be ignored in the generation of the exercise strategy.
If you really need them, price them separately.

We therefore have number of exercise dates aggregated deflated cash-flows per path. 

However, we have 2 * number of reset dates cash-flows per path. Note that these flows
occur at fixed times. The routine computing ther magnitudes does not need to know what these times are.  

The deflator needs the time of the cash-flow's occurence, its determination time and the time to discount back to, however. 

numeraireValues_dev will be of size number paths * (numberSteps+1) since we go from 0 to numberSteps inclusive. (
We may not need the numberSteps entry, however.) 

genFlows1_dev, genFlows_dev are not needed outside the batch so size is determined by batch size not total number of paths

*/


// implicit assumption here that steps aand rates are equals.
/* LMM_evolver_gpu.h
Routine is in 
void spot_measure_numeraires_computation_gpu(        float* discount_factors_global,
                                             float* numeraire_values_global, //output
                                             int paths,
                                             int n_rates
                                             )

also in LMM_evolver_main.h                                             
void spot_measure_numeraires_computation_gpu_main(   thrust::device_vector<float>& discount_factors_device,
                                                                                            thrust::device_vector<float>& numeraire_values_device, //output
                                                                                            int paths,
                                                                                            int n_rates
                                                                                            );
*/

   // function to extract
   // basis variables
/*
template<class T>
void adjoinBasisVariables(float* forwards_dev,
                          float* discountRatios_dev,
                          float* rate1,
                          float* rate2,
                          float* rate3,
                          int numberPathsPerBatch,
                          int totalNumberOfPaths, 
                          int numberRates, 
                          int numberExerciseTimes,
                          int* exerciseIndices_dev, // the indices of the exercise times amongst the evolution times
                          int* exerciseIndicators_dev, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                          float* basisFunctionVariables_dev // output location
                          );


// behaviour of T
// we not allow path dependence at this point in time
// however note that rate1, rate2, rate3 could be path-dependent quantities. 
class basisVariableExample
{
public:

basisVariableExample(	int offsetIntegerData, // to access texture containing data specific to the basis variables
		                int offsetFloatData);

void writeBasisVariableValues(float* output_global_location, // includes pathNumber and stepNumber offset 
        int layerSize, // steps*paths
		int numberOfRatesReset,  // the index amongst the rate times 
        int exerciseIndex, // the index amongst the exercise times 
		int numberRates, 
		float rate1, 
		float rate2, 
		float rate3, 
		float* allForwardRates, 
		float* allDiscountRatios,
        int pathsPerBatch,
        int batchLayerSize
		);
};
*/
/*
template<class T> 
void cashFlowGenerator<T>(float* genFlows1_dev, 
                                    float* genFlows2_dev, 
                                    float* aux_data, 
                                    int batchPaths, 
                                    int numberSteps,
                                    float* rates1_dev, 
                                    float* rates2_dev, 
                                    float* rates3_dev, 
                                    float* forwards_dev, 
                                    float* discRatios_dev);
                                    */

// call once for each set of cash flows 
void cashFlowDeflator(float* genFlows_dev, // input 
                      float* genFlows_discounted_dev, // output
                      int batchPaths,
                      int pathsForOutput, // specified separately as we may wish to use the routine in more than one way
                      int totalPaths, 
                      int* firstIndex_dev, //  largest rate time index <= cash-flow occurrence time 
                      int* secondIndex_dev // smallest rate time index > cash-flow occurrence time 
                      float* theta_dev,  // interpolation fraction between two above indices
                      int generationIndex_dev, // index of rate time for the cash-flow generation time 
                      int* targetIndex_dev, // index of rate time for the exercise date to which we wish to deflate
                      float* numeraireValues_dev);

// routine to bundle discounted cash flows between exercise times together 
void cashFlowAggregator(float* genFlows_discounted_1_dev,// batchPaths*numberRates
                        float* genFlows_discounted_2_dev,// batchPaths*numberRates
                        int batchPaths,
                        int totalPaths,
                        int* exerciseIndices_dev, // indices of exercise times amongst reset times
                        int numberExerciseTimes,
                        float* aggregated_flows_dev // totalPaths * numberExerciseTimes
                        );

   /*                     
template<class T> 
void exerciseValueGenerator<T>(float* exerciseValues_dev,  // output, one per exercise time per path
                               float* aux_data, 
                               int pathsForOutput, // typically totalPaths
                               int outputOffset,
                               int batchPaths, 
                               int numberSteps,
                               float* rates1_dev, 
                               float* rates2_dev, 
                               float* rates3_dev, 
                               float* forwards_dev, 
                               float* discRatios_dev);
*/
// use cashFlowDeflator to deflate exerciseValues

// routines above should cover the first part

template<class T> // we may need doubles for this bit
void generateRegressionMatrix(T* regressionMatrix_dev, // E(X_i X_j) where X_k are the basis functions
                              T* targetVector_dev,     // E(X_i continuation val) 
                              int polynomialBasisOrder,
                              float* basisVariableValues_dev,
                              int totalPaths,
                              int numberExerciseDates,
                              int exerciseIndex, 
                              float* continuationValues_dev);

void updateContinuationValues(float* continuationValueCoefficients_dev, 
                              int polynomialBasisOrder,
                              float* basisVariableValues_dev,
                              float* continuationValues_dev,
                              float* exerciseValues_dev,
                              float* numeraireValues_dev);






                          