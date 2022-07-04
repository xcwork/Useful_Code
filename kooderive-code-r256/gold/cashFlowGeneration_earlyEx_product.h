//
//                                  cashFlowGeneration_earlyEx_product.h
//
//


// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef CASH_FLOW_GEN_EARLY_EX_PROD_H
#define CASH_FLOW_GEN_EARLY_EX_PROD_H

#include <gold/cashFlowGeneration_product_gold.h>
#include <gold/cashFlowGeneration_gold.h>
#include <vector>
#include <gold/MatrixFacade.h> 
#include <gold/math/basic_matrix_gold.h>
#include <gold/math/Cube_gold.h>
/*
R is original product to be broken, could be zero or a swap
S is exercise value on breaking, eg a rebate of zero, or the pay-off of a Bermudan swaption
T is exercise strategy eg the LSA strategy
*/

template<class R,class S,class T,class D>
void cashFlowGeneratorEE_gold(
                            std::vector<D>& genFlows1,
                            std::vector<D>& genFlows2, 
                            R product,
                            S exerciseValue,
                            T exerciseStrategy,
                            int paths, 
                            int numberSteps,
                            const std::vector<bool>& isExerciseDate,
                            const std::vector<D>& rates1, 
                            const std::vector<D>& rates2, 
                            const std::vector<D>& rates3, 
                            const CubeConstFacade<D>& strategyVariables,
                            const std::vector<D>& forwards, 
                            const std::vector<D>& discRatios )   
{    
    int numberRates = numberSteps;


    MatrixConstFacade<D> rates1Matrix(&rates1[0],numberSteps,paths);
    MatrixConstFacade<D> rates2Matrix(&rates2[0],numberSteps,paths);
    MatrixConstFacade<D> rates3Matrix(&rates3[0],numberSteps,paths);
    CubeConstFacade<D> forwardsCube(&forwards[0],numberSteps, numberRates, paths);
    CubeConstFacade<D> discountsCube(&discRatios[0],numberSteps, numberRates+1, paths);


    genFlows1.resize(paths*numberSteps);
    genFlows2.resize(paths*numberSteps);

    
    for (int p=0; p < paths; ++p)
    { 
        product.newPath();

        bool done=false;

        int i=0; // maintain i in scope 
        int exerciseIndex=0;
        
        for (; i < numberSteps && !done; ++i)        
        {
            
            D flow1=0.0;        
            D flow2=0.0;
            D rate1 = rates1Matrix(i,p);
            D rate2 = rates2Matrix(i,p);   
            D rate3 = rates3Matrix(i,p); 

            bool exercise=false;

            if (isExerciseDate[i])
            {
				// implicit assumption is that exercise value is discounted to exercise time 
                D exValue= exerciseValue.exerciseValue(i,exerciseIndex,rate1,rate2,rate3,p,paths,forwardsCube,discountsCube);
       
      


                exercise = exerciseStrategy(exValue,p,exerciseIndex,strategyVariables,rate1,rate2,rate3,forwards,discRatios);
           
                if (exercise)
                {
                    flow1 = exValue;
                    done = true;
                }

                ++exerciseIndex;
            }

            if (!exercise)
                done = product.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwardsCube,discountsCube,p);

            int location = i*paths+p;

            genFlows1[location] = flow1;
            genFlows2[location] = flow2;
        
        }  
        for (; i < numberSteps; ++i)        
        {
            int location = i*paths+p;
            genFlows1[location] = 0.0;
            genFlows2[location] = 0.0;

        }

    }
}


/*
R is original product to be broken, could be zero or a swap
S is exercise value on breaking, eg a rebate of zero, or the pay-off of a Bermudan swaption
T is exercise strategy eg the LSA strategy
D type -- ie float or double

purpose of this routine is to generate the cash-flows conditional on a future state
the cash-flows up to some point are always the same and given by the path distinguishedPath
after this step the flows vary
To allow for path-dependence the path is repeated from the start each time even though the first part is always the same,

possible alternate designs are 
1) add a memory state to the product so we can reset to the distinguished step
2) assign the product at the distinguised step to a working copy of it

It seems unlikely that this part of the code will be a bottleneck so it's difficult to get worried.
*/

template<class R,class S,class T,class D>
void cashFlowGeneratorEEConditional_gold(
                            std::vector<D>& genFlows1,
                            std::vector<D>& genFlows2, 
                            R& product,
                            S& exerciseValue,
                            T& exerciseStrategy,
                            int paths, 
                            int numberSteps,
                            const std::vector<bool>& isExerciseDate,
                            const std::vector<D>& rates1, 
                            const std::vector<D>& rates2, 
                            const std::vector<D>& rates3, 
                            const CubeConstFacade<D>& strategyVariables,
                            const std::vector<D>& forwards, 
                            const std::vector<D>& discRatios,
								// conditional data
							int distinguishedPath,
							int subPaths,
							int conditionalStep,
							const std::vector<D>& conditionalRates1, 
                            const std::vector<D>& conditionalRates2, 
                            const std::vector<D>& conditionalRates3, 
                            const std::vector<D>& conditionalForwards, 
                            const std::vector<D>& conditionalDiscRatios)   
{    
    int numberRates = numberSteps;

    MatrixConstFacade<D> rates1Matrix(&rates1[0],numberSteps,paths);
    MatrixConstFacade<D> rates2Matrix(&rates2[0],numberSteps,paths);
    MatrixConstFacade<D> rates3Matrix(&rates3[0],numberSteps,paths);
    CubeConstFacade<D> forwardsCube(&forwards[0],numberSteps, numberRates, paths);
    CubeConstFacade<D> discountsCube(&discRatios[0],numberSteps, numberRates+1, paths);

		
	MatrixConstFacade<D> rates1CondMatrix(&conditionalRates1[0],numberSteps,subPaths);
    MatrixConstFacade<D> rates2CondMatrix(&conditionalRates2[0],numberSteps,subPaths);
    MatrixConstFacade<D> rates3CondMatrix(&conditionalRates3[0],numberSteps,subPaths);
  
	CubeConstFacade<D> forwardsCondCube(&conditionalForwards[0],numberSteps, numberRates, subPaths);
    CubeConstFacade<D> discountsCondCube(&conditionalDiscRatios[0],numberSteps, numberRates+1, subPaths);


    genFlows1.resize(subPaths*numberSteps);
    genFlows2.resize(subPaths*numberSteps);
    MatrixFacade<double> genFlows1_mat(genFlows1,numberSteps,subPaths);
    MatrixFacade<double> genFlows2_mat(genFlows2,numberSteps,subPaths);

    int startExerciseIndex=0;

    for (int k=0; k < conditionalStep; ++k)
        if (isExerciseDate[k])
            ++startExerciseIndex;
    
    for (int p=0; p < subPaths; ++p)
    { 
        product.newPath();

        bool done=false;

        int i=0; // maintain i in scope 
        int exerciseIndex=startExerciseIndex;
        
		D rate1,rate2,rate3;
        for (; i < numberSteps && !done; ++i)        
        {
            
            D flow1=0.0;        
            D flow2=0.0;

			int q= (i >= conditionalStep) ? p : distinguishedPath;
			bool condPart = (i>=conditionalStep);

			if (condPart)
			{
				rate1 = rates1CondMatrix(i,q);
				rate2 = rates2CondMatrix(i,q);   
				rate3 = rates3CondMatrix(i,q); 
			}
			else
			{
				rate1 = rates1Matrix(i,q);
				rate2 = rates2Matrix(i,q);   
				rate3 = rates3Matrix(i,q); 
			}

            bool exercise=false;

            if (isExerciseDate[i] && condPart)
            {
				// implicit assumption is that exercise value is discounted to exercise time 
                D exValue= exerciseValue.exerciseValue(i,exerciseIndex,rate1,rate2,rate3,p,paths,forwardsCondCube,discountsCondCube);
				exercise = exerciseStrategy(exValue,p,exerciseIndex,strategyVariables,rate1,rate2,rate3,conditionalForwards,conditionalDiscRatios);
           
                if (exercise)
                {
                    flow1 = exValue;
                    done = true;
                }

                ++exerciseIndex;
            }

            if (!exercise)
			{
                done = condPart ? product.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwardsCondCube,discountsCondCube,q)
				              : product.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwardsCube,discountsCube,q);
			}
            genFlows1_mat(i,p) = condPart ? flow1 : 0.0;
            genFlows2_mat(i,p) = condPart ? flow2 : 0.0;
        
        }  
        for (; i < numberSteps; ++i)        
        {
            genFlows1_mat(i,p) =  0.0;
            genFlows2_mat(i,p) =  0.0;
        

        }

    }
}

/*
No underlying product
Payoffs inputted
T is exercise strategy eg the LSA strategy
*/

template<class T,class D>
void cashFlowGeneratorEESimple_gold(
                            std::vector<D>& genFlows1, // output  
                            std::vector<int>& generationIndices, // output for discounting
                            const MatrixConstFacade<D>& payoffs,
                            T exerciseStrategy,
                            int paths, 
                            int numberSteps,
                            int stepsToSkip,
                            const std::vector<bool>& isExerciseDate,
                            const std::vector<D>& rates1, 
                            const std::vector<D>& rates2, 
                            const std::vector<D>& rates3, 
                            const CubeConstFacade<D>& strategyVariables,
                            const std::vector<D>& forwards, 
                            const std::vector<D>& discRatios )   
{    
    int numberRates = numberSteps;


    MatrixConstFacade<D> rates1Matrix(&rates1[0],numberSteps,paths);
    MatrixConstFacade<D> rates2Matrix(&rates2[0],numberSteps,paths);
    MatrixConstFacade<D> rates3Matrix(&rates3[0],numberSteps,paths);
    CubeConstFacade<D> forwardsCube(&forwards[0],numberSteps, numberRates, paths);
    CubeConstFacade<D> discountsCube(&discRatios[0],numberSteps, numberRates+1, paths);

    genFlows1.resize(0);

    genFlows1.resize(paths,0.0);
    generationIndices.resize(0);
    generationIndices.resize(paths,0);

    
    for (int p=0; p < paths; ++p)
    { 
   
        bool done=false;

        int i=stepsToSkip; // maintain i in scope 
        int exerciseIndex=stepsToSkip;
        genFlows1[p] =0.0;
        
        for (; i < numberSteps && !done; ++i)        
        {
            
            D flow1=0.0;        
            D rate1 = rates1Matrix(i,p);
            D rate2 = rates2Matrix(i,p);   
            D rate3 = rates3Matrix(i,p); 

            if (isExerciseDate[i])
            {
				// implicit assumption is that exercise value is discounted to exercise time 
                D exValue=payoffs(exerciseIndex,p);
      
                bool exercise = exerciseStrategy(exValue,p,exerciseIndex,strategyVariables,rate1,rate2,rate3,forwards,discRatios);
           
                if (exercise)
                {
                    genFlows1[p] = exValue;
                    generationIndices[p] =i;

                    done = true;
                }

                ++exerciseIndex;
            }   
        }  
    }
}



/*
No underlying product
Payoffs inputted
T is exercise strategy eg the LSA strategy
simply outputs when exercises should occur, including past first exercise
*/

template<class T,class D>
void exerciseAllDeterminerEESimple_gold(  
                            MatrixFacade<int>& exercise_occurences_mat, // 0 = false, 1 = true
                            const MatrixConstFacade<D>& payoffs,
                            T exerciseStrategy,
                            int paths, 
                            int numberSteps,
                            int stepsToSkip,
                            const std::vector<bool>& isExerciseDate,
                            const std::vector<D>& rates1, 
                            const std::vector<D>& rates2, 
                            const std::vector<D>& rates3, 
                            const CubeConstFacade<D>& strategyVariables,
                            const std::vector<D>& forwards, 
                            const std::vector<D>& discRatios )   
{    
    int numberRates = numberSteps;


    MatrixConstFacade<D> rates1Matrix(&rates1[0],numberSteps,paths);
    MatrixConstFacade<D> rates2Matrix(&rates2[0],numberSteps,paths);
    MatrixConstFacade<D> rates3Matrix(&rates3[0],numberSteps,paths);
    CubeConstFacade<D> forwardsCube(&forwards[0],numberSteps, numberRates, paths);
    CubeConstFacade<D> discountsCube(&discRatios[0],numberSteps, numberRates+1, paths);

   
    for (int p=0; p < paths; ++p)
    { 
   
        bool done=false;

        int i=stepsToSkip; // maintain i in scope 
        int exerciseIndex=0;
        
        for (; i < numberSteps && !done; ++i)        
        {
            
            D flow1=0.0;        
            D rate1 = rates1Matrix(i,p);
            D rate2 = rates2Matrix(i,p);   
            D rate3 = rates3Matrix(i,p); 

            if (isExerciseDate[i])
            {
				// implicit assumption is that exercise value is discounted to exercise time 
                D exValue=payoffs(exerciseIndex,p);
      
                bool exercise = exerciseStrategy(exValue,p,exerciseIndex,strategyVariables,rate1,rate2,rate3,forwards,discRatios);
           
                exercise_occurences_mat(exerciseIndex,p) = exercise ? 1 : 0;

                ++exerciseIndex;
            }   
        }  
    }
}






// template because we need a rule for turning basis variables into basis functions
template<class T, class D>
class LSAExerciseStrategy
{
public:
    LSAExerciseStrategy(int numberExerciseTimes_,
        const std::vector<int>& variablesPerStep_,
        const std::vector<float>& basisShifts_,
        const std::vector<float>& basisWeights_,
        const std::vector<T>& functionProducers_);

 
    LSAExerciseStrategy(int numberExerciseTimes_,
        const std::vector<int>& variablesPerStep_,
        const std::vector<double>& basisShifts_,
        const std::vector<double>& basisWeights_,
        const std::vector<T>& functionProducers_);



   bool operator()(D exValue, 
                   int p,
                   int exerciseIndex,
                   const CubeConstFacade<D>& strategyVariables,
                   D rate1,
                   D rate2,
                   D rate3,
                   const std::vector<D>&  v,
                   const std::vector<D>& w) const;
           

private:
    int numberExerciseTimes;
    int maxBasisVariables;
    
    std::vector<int> variablesPerStep;
    std::vector<D> basisShifts; // ie Andersen trigger shifts
    
    std::vector<D> basisWeights;

    std::vector<T> functionProducers;
    int maxFunctions;

    mutable std::vector<D> theseBasisVariables;
    mutable std::vector<D> basisFunctions;
   
};

template<class T, class D>
LSAExerciseStrategy<T,D>::LSAExerciseStrategy(int numberExerciseTimes_,
        const std::vector<int>& variablesPerStep_,
        const std::vector<float>& basisShifts_,
        const std::vector<float>& basisWeights_,
        const std::vector<T>& functionProducers_) : numberExerciseTimes(numberExerciseTimes_),
        maxBasisVariables(*std::max_element(variablesPerStep_.begin(),variablesPerStep_.end())),
        variablesPerStep(variablesPerStep_),  
        basisWeights(basisWeights_.size()),
        functionProducers(functionProducers_),
        theseBasisVariables(maxBasisVariables)
 {
   maxFunctions = functionProducers[0].numberDataPoints();
     for (int i=1; i < numberExerciseTimes_; ++i)
         maxFunctions = std::max(maxFunctions, functionProducers[i].numberDataPoints());

     basisFunctions.resize(maxFunctions);

     basisShifts.resize(basisShifts_.size());

     for (size_t i=0; i < basisShifts.size(); ++i)
         basisShifts[i] = static_cast<D>(basisShifts_[i]);

   
     for (size_t i=0; i < basisWeights.size(); ++i)
             basisWeights[i] = static_cast<D>(basisWeights_[i]);
  
  

 }


template<class T, class D>
LSAExerciseStrategy<T,D>::LSAExerciseStrategy(int numberExerciseTimes_,
        const std::vector<int>& variablesPerStep_,
        const std::vector<double>& basisShifts_,
        const std::vector<double>& basisWeights_,
        const std::vector<T>& functionProducers_) : numberExerciseTimes(numberExerciseTimes_),
        maxBasisVariables(*std::max_element(variablesPerStep_.begin(),variablesPerStep_.end())),
        variablesPerStep(variablesPerStep_),  
        basisWeights(basisWeights_.size()),
        functionProducers(functionProducers_),
        theseBasisVariables(maxBasisVariables)
 {
   maxFunctions = functionProducers[0].numberDataPoints();
     for (int i=1; i < numberExerciseTimes_; ++i)
         maxFunctions = std::max(maxFunctions, functionProducers[i].numberDataPoints());

     basisFunctions.resize(maxFunctions);

     basisShifts.resize(basisShifts_.size());

     for (size_t i=0; i < basisShifts.size(); ++i)
         basisShifts[i] = static_cast<D>(basisShifts_[i]);

   
     for (size_t i=0; i < basisWeights.size(); ++i)
             basisWeights[i] = static_cast<D>(basisWeights_[i]);
  
  

 }

template<class T, class D>
bool LSAExerciseStrategy<T,D>::operator()(D exValue, 
                   int p,
                   int exerciseIndex,
                   const CubeConstFacade<D>& strategyVariables,
                   D rate1,
                   D rate2,
                   D rate3,
                   const std::vector<D>& notused1 ,
                   const std::vector<D>& notused2) const
{
    MatrixConstFacade<D> LS_Coeffs_mat(basisWeights,numberExerciseTimes,maxFunctions);
    for (int i=0; i < variablesPerStep[exerciseIndex]; ++i)
        theseBasisVariables[i] = strategyVariables(exerciseIndex,i,p);

    functionProducers[exerciseIndex].writeData(theseBasisVariables, basisFunctions);

    double estCondValue = basisShifts[exerciseIndex];

    for (int j=0; j <  functionProducers[exerciseIndex].numberDataPoints(); ++j)
         estCondValue += basisFunctions[j]*LS_Coeffs_mat(exerciseIndex,j);

 
    return estCondValue < exValue;

}




// template because we need a rule for turning basis variables into basis functions
template<class T, class D>
class LSAMultiExerciseStrategy
{
public:
    LSAMultiExerciseStrategy(int numberExerciseTimes_,
        const std::vector<int>& variablesPerStep_,
        const std::vector<float>& basisShifts_,
        const std::vector<float>& basisWeights_,
		int regressionDepth_,
		const MatrixConstFacade<float>& lowerCuts_,
		const MatrixConstFacade<float>& upperCuts_,
		const CubeConstFacade<float>& means_cube_,
		const CubeConstFacade<float>& sds_cube_,
        const std::vector<T>& functionProducers_,
        int maxBasisVariables_,
        bool dummybool);

 
LSAMultiExerciseStrategy(int numberExerciseTimes_,
        const std::vector<int>& variablesPerStep_,
        const std::vector<double>& basisShifts_,
        const std::vector<double>& basisWeights_,
		int regressionDepth_,
		const MatrixConstFacade<double>& lowerCuts_,
		const MatrixConstFacade<double>& upperCuts_,
		const CubeConstFacade<double>& means_cube_,
		const CubeConstFacade<double>& sds_cube_,
        const std::vector<T>& functionProducers_,
        int maxBasisVariables_);


   bool operator()(D exValue, 
                   int p,
                   int exerciseIndex,
                   const CubeConstFacade<D>& strategyVariables,
                   D rate1,
                   D rate2,
                   D rate3,
                   const std::vector<D>&  v,
                   const std::vector<D>& w) const;
         
    double estNetCondValue(D exValue, 
                   int p,
                   int exerciseIndex,
                   const CubeConstFacade<D>& strategyVariables,
                   D rate1,
                   D rate2,
                   D rate3,
                   const std::vector<D>& notused1 ,
                   const std::vector<D>& notused2) const;  

private:

	std::vector<D> dummy;

    int numberExerciseTimes;
    int maxBasisVariables;
    
    std::vector<int> variablesPerStep;
    std::vector<D> basisShifts; // ie Andersen trigger shifts
    
    std::vector<D> basisWeights;
	

	int regressionDepth;

	Matrix_gold<D> lowerCuts;
	Matrix_gold<D> upperCuts;

	Cube_gold<D> means_cube;
	Cube_gold<D> sds_cube;


    std::vector<T> functionProducers;
    int maxFunctions;

    mutable std::vector<D> theseBasisVariables;
    mutable std::vector<D> basisFunctions;
   
};

template<class T, class D>
LSAMultiExerciseStrategy<T,D>::LSAMultiExerciseStrategy(int numberExerciseTimes_,
        const std::vector<int>& variablesPerStep_,
        const std::vector<float>& basisShifts_,
        const std::vector<float>& basisWeights_,
		int regressionDepth_,
		const MatrixConstFacade<float>& lowerCuts_,
		const MatrixConstFacade<float>& upperCuts_,
		const CubeConstFacade<float>& means_cube_,
		const CubeConstFacade<float>& sds_cube_,
        const std::vector<T>& functionProducers_,
        int maxBasisVariables_,
        bool floatinput) : 
       numberExerciseTimes(numberExerciseTimes_),
       maxBasisVariables(maxBasisVariables_),// *std::max_element(variablesPerStep_.begin(),variablesPerStep_.end())),
       variablesPerStep(variablesPerStep_),  
       basisWeights(basisWeights_.size()),
		regressionDepth(regressionDepth_),
		lowerCuts(numberExerciseTimes_,regressionDepth_,0.0f),
		upperCuts(numberExerciseTimes_,regressionDepth_,0.0f),
		means_cube(means_cube_.numberLayers(),means_cube_.numberRows(),means_cube_.numberColumns(),0.0f),
		sds_cube(sds_cube_.numberLayers(),sds_cube_.numberRows(),sds_cube_.numberColumns(),0.0f),
        functionProducers(functionProducers_),
       theseBasisVariables(maxBasisVariables)
 {

     

   maxFunctions = functionProducers[0].numberDataPoints();
     for (int i=1; i < numberExerciseTimes_; ++i)
         maxFunctions = std::max(maxFunctions, functionProducers[i].numberDataPoints());

     basisFunctions.resize(maxFunctions);

     basisShifts.resize(basisShifts_.size());

     for (size_t i=0; i < basisShifts.size(); ++i)
         basisShifts[i] = static_cast<D>(basisShifts_[i]);

   
     for (size_t i=0; i < basisWeights.size(); ++i)
             basisWeights[i] = static_cast<D>(basisWeights_[i]);

	 for (int i=0; i < numberExerciseTimes_; ++i)
		 for (int j=0; j < regressionDepth_; ++j)
		 {
			 lowerCuts(i,j)=static_cast<D>(lowerCuts_(i,j));

			 upperCuts(i,j)=static_cast<D>(upperCuts_(i,j));

			  for (int k=0; k < maxBasisVariables; ++k)
			  {
				  means_cube(i,j,k) = static_cast<D>(means_cube_(i,j,k));
			  
				  sds_cube(i,j,k) = static_cast<D>(sds_cube_(i,j,k));
			  }

		 }
  

 }


template<class T, class D>
LSAMultiExerciseStrategy<T,D>::LSAMultiExerciseStrategy(int numberExerciseTimes_,
        const std::vector<int>& variablesPerStep_,
        const std::vector<double>& basisShifts_,
        const std::vector<double>& basisWeights_,
		int regressionDepth_,
		const MatrixConstFacade<double>& lowerCuts_,
		const MatrixConstFacade<double>& upperCuts_,
		const CubeConstFacade<double>& means_cube_,
		const CubeConstFacade<double>& sds_cube_,
        const std::vector<T>& functionProducers_,
        int maxBasisVariables_) : 
numberExerciseTimes(numberExerciseTimes_),
        maxBasisVariables(maxBasisVariables_), //*std::max_element(variablesPerStep_.begin(),variablesPerStep_.end())),
  variablesPerStep(variablesPerStep_),  
  basisWeights(basisWeights_.size()),
		regressionDepth(regressionDepth_),
        		lowerCuts(numberExerciseTimes_,regressionDepth_,static_cast<D>(0.0)),
upperCuts(numberExerciseTimes_,regressionDepth_,static_cast<D>(0.0)) ,
means_cube(means_cube_.numberLayers(),means_cube_.numberRows(),means_cube_.numberColumns(),static_cast<D>(0.0)),
sds_cube(sds_cube_.numberLayers(),sds_cube_.numberRows(),sds_cube_.numberColumns(),static_cast<D>(0.0)),
 functionProducers(functionProducers_),
 theseBasisVariables(maxBasisVariables)
 {
/*     
   maxFunctions = functionProducers[0].numberDataPoints();
     for (int i=1; i < numberExerciseTimes_; ++i)
         maxFunctions = std::max(maxFunctions, functionProducers[i].numberDataPoints());
         */

     T functionProducerMax(maxBasisVariables_);
     maxFunctions = functionProducerMax.numberDataPoints();
     basisFunctions.resize(maxFunctions);

     basisShifts.resize(basisShifts_.size());

     for (size_t i=0; i < basisShifts.size(); ++i)
         basisShifts[i] = static_cast<D>(basisShifts_[i]);

   
     for (size_t i=0; i < basisWeights.size(); ++i)
             basisWeights[i] = static_cast<D>(basisWeights_[i]);

	 for (int i=0; i < numberExerciseTimes_; ++i)
		 for (int j=0; j < regressionDepth_; ++j)
		 {
			 lowerCuts(i,j)=static_cast<D>(lowerCuts_(i,j));

			 upperCuts(i,j)=static_cast<D>(upperCuts_(i,j));

			  for (int k=0; k < maxBasisVariables; ++k)
			  {
				  means_cube(i,j,k) = static_cast<D>(means_cube_(i,j,k));
			  
				  sds_cube(i,j,k) = static_cast<D>(sds_cube_(i,j,k));
			  }

		 }
  
  
 }

template<class T, class D>
double LSAMultiExerciseStrategy<T,D>::estNetCondValue(D exValue, 
                   int p,
                   int exerciseIndex,
                   const CubeConstFacade<D>& strategyVariables,
                   D rate1,
                   D rate2,
                   D rate3,
                   const std::vector<D>& notused1 ,
                   const std::vector<D>& notused2) const
{
	CubeConstFacade<D> basisWeights_cube(basisWeights,numberExerciseTimes,regressionDepth,maxFunctions);

  
 
	int depth=0;
	double netCondValue;

	do
	{
	    double estCondValue = basisShifts[exerciseIndex];
		for (int i=0; i < variablesPerStep[exerciseIndex]; ++i)
			theseBasisVariables[i] = (strategyVariables(exerciseIndex,i,p)-means_cube(exerciseIndex,depth,i))/sds_cube(exerciseIndex,depth,i);

  
		functionProducers[exerciseIndex].writeData(theseBasisVariables, basisFunctions);

 

	    for (int j=0; j <  functionProducers[exerciseIndex].numberDataPoints(); ++j)
        {
            double weight = basisWeights_cube(exerciseIndex,depth,j);
		     estCondValue += basisFunctions[j]*weight;
        }

		netCondValue = estCondValue- exValue;
		++depth;

	}
	while (depth < regressionDepth && netCondValue > lowerCuts(exerciseIndex,depth-1) && netCondValue < upperCuts(exerciseIndex,depth-1));

    return netCondValue;

}

template<class T, class D>
bool LSAMultiExerciseStrategy<T,D>::operator()(D exValue, 
                   int p,
                   int exerciseIndex,
                   const CubeConstFacade<D>& strategyVariables,
                   D rate1,
                   D rate2,
                   D rate3,
                   const std::vector<D>& notused1 ,
                   const std::vector<D>& notused2) const
{
    return estNetCondValue(exValue,p,exerciseIndex,strategyVariables,rate1,rate2,rate3,notused1,notused2)<0.0;
}
/*
R is original product to be broken, could be zero or a swap
S is exercise value on breaking, eg a rebate of zero, or the pay-off of a Bermudan swaption
T is exercise strategy eg the LSA strategy
*/

template<class R,class S,class T,class D>
void cashFlowGeneratorEEToEnd_gold(
                            std::vector<D>& genFlows1,
                            std::vector<D>& genFlows2, 
                            std::vector<D>& exerciseValuesGenerated,
                            std::vector<int>& exerciseOccurences,
                            R product,
                            S exerciseValue,
                            T exerciseStrategy,
                            int paths, 
                            int numberSteps,
                            const std::vector<bool>& isExerciseDate,
                            const std::vector<D>& rates1, 
                            const std::vector<D>& rates2, 
                            const std::vector<D>& rates3, 
                            const CubeConstFacade<D>& strategyVariables,
                            const std::vector<D>& forwards, 
                            const std::vector<D>& discRatios,
                            bool storeContinuations,
                            MatrixFacade<double>& continuations_gold)   
{    
    int numberRates = numberSteps;


    MatrixConstFacade<D> rates1Matrix(&rates1[0],numberSteps,paths);
    MatrixConstFacade<D> rates2Matrix(&rates2[0],numberSteps,paths);
    MatrixConstFacade<D> rates3Matrix(&rates3[0],numberSteps,paths);
    CubeConstFacade<D> forwardsCube(&forwards[0],numberSteps, numberRates, paths);
    CubeConstFacade<D> discountsCube(&discRatios[0],numberSteps, numberRates+1, paths);

    MatrixFacade<D> exerciseValuesGenerated_mat(exerciseValuesGenerated,numberSteps,paths);
    MatrixFacade<int> exerciseOccurences_mat(exerciseOccurences,numberSteps,paths);

    genFlows1.resize(paths*numberSteps);
    genFlows2.resize(paths*numberSteps);

    
    for (int p=0; p < paths; ++p)
    { 
        product.newPath();

        bool done=false;

        int i=0; // maintain i in scope 
        int exerciseIndex=0;
        
        for (; i < numberSteps && !done; ++i)        
        {
            
            D flow1=0.0;        
            D flow2=0.0;
            D rate1 = rates1Matrix(i,p);
            D rate2 = rates2Matrix(i,p);   
            D rate3 = rates3Matrix(i,p); 

            bool exercise=false;

            if (isExerciseDate[i])
            {
				// implicit assumption is that exercise value is discounted to exercise time 
                D exValue= exerciseValue.exerciseValue(i,exerciseIndex,rate1,rate2,rate3,p,paths,forwardsCube,discountsCube);
                
                if (storeContinuations)
                {
                    double x= exerciseStrategy.estNetCondValue(exValue,p,exerciseIndex,strategyVariables,rate1,rate2,rate3,forwards,discRatios);
                    exercise = x  < 0.0;
                    continuations_gold(i,p)=x;
                }
                else
                    exercise = exerciseStrategy(exValue,p,exerciseIndex,strategyVariables,rate1,rate2,rate3,forwards,discRatios);
                
                
                exerciseOccurences_mat(i,p) = exercise ? 1 :0;
                exerciseValuesGenerated_mat(i,p) = exercise ? exValue : 0.0;
      
                ++exerciseIndex;
            }

          
            done = product.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwardsCube,discountsCube,p);

            int location = i*paths+p;

            genFlows1[location] = flow1;
            genFlows2[location] = flow2;
        
        }  
        for (; i < numberSteps; ++i)        
        {
            int location = i*paths+p;
            genFlows1[location] = 0.0;
            genFlows2[location] = 0.0;

        }

    }
}


#endif

