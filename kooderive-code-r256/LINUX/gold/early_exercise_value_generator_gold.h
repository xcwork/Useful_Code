

//
//
//                          early_exercise_value_generator_gold.h
//
//
// (c) Mark Joshi 2011, 2014
// This code is released under the GNU public licence version 3

/*
Routine to generate the exercise values given paths. 
Inputs are batched. 
Outputs are collated. 

Note that the timing of payments is not addressed here.
It is intended that a separate class will take in the exercise values 
and payment schedule and deflate them to exercise decision days. 

*/


#ifndef EARLY_EXERCISE_VALUE_GENERATOR_GOLD_H
#define EARLY_EXERCISE_VALUE_GENERATOR_GOLD_H


#include <gold/pragmas.h> 
#include <gold/MatrixFacade.h>
#include <vector>



template<class D>
class earlyExerciseNullGold
{
public:
    earlyExerciseNullGold(const std::vector<D>& floatingInputs,
        const std::vector<int>& integerInputs, 
        int numberOfRatesAndSteps)
    {}

     earlyExerciseNullGold()
    {}

    D exerciseValue(int stepEvolution,
        int exerciseTimeIndex,
        D rate1,
        D rate2,
        D rate3,
        int pathNumber, // needed to find right bit of forwards and discratios
        int numberOfPaths, // needed to find right bit of forwards and discratios
        const CubeConstFacade<D>& allForwardRates, 
        const CubeConstFacade<D>& allDiscountRatios)
    {   
        return 0.0;
    }


};


template<class D>
class earlyExerciseBermudanPayerSwaptionGold
{
private:
    D strike_;
public:
    earlyExerciseBermudanPayerSwaptionGold(const std::vector<D>& floatingInputs,
        const std::vector<int>& integerInputs, 
        int numberOfRatesAndSteps) : strike_(floatingInputs[0])
    {

    }

    D exerciseValue(int stepEvolution,
        int exerciseTimeIndex,
        D rate1, // interpreted as coterminal swap rate
        D rate2, // interpreted as coterminal annuity
        D rate3, // ignored 
        int pathNumber, // needed to find right bit of forwards and discratios
        int numberOfPaths, // needed to find right bit of forwards and discratios
        const CubeConstFacade<D>& allForwardRates, 
        const CubeConstFacade<D>& allDiscountRatios)
    {   
        D swapVal = (rate1-strike_)*rate2;

        return swapVal > 0.0f ? swapVal : 0.0f;
    }


};


template<class T,class D>
void exercise_value_gold(const std::vector<D>& floatingInputs,
                         const std::vector<int>& integerInputs,// to access texture containing data specific to the  exercise value object
                         int numberStepsAndRates,
                         const CubeConstFacade<D>&  forwards_cube,
                         const CubeConstFacade<D>&  discountRatios_cube,
                         const MatrixConstFacade<D>&  rate1_matrix,
                         const MatrixConstFacade<D>&  rate2_matrix,
                         const MatrixConstFacade<D>&  rate3_matrix,
                         int numberPathsPerBatch,
                         int numberPathsPreviouslyDone, 
                         const std::vector<int>&  exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
                         const std::vector<int>&  exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                         MatrixFacade<D>&  exerciseValues_matrix// output location
                         )
{

    T obj(floatingInputs, 
        integerInputs, 
        numberStepsAndRates);

    int numberExerciseTimes= exerciseIndices_vec.size();



    for (int i =0; i < numberPathsPerBatch; ++i)
        for (int j=0; j < numberExerciseTimes; ++j)
        {
            int step = exerciseIndices_vec[j];


            D rate1 = rate1_matrix(j,i);
            D rate2 = rate2_matrix(j,i);
            D rate3 = rate3_matrix(j,i);

            D eValue = obj.exerciseValue(   
                step,
                j,
                rate1, // interpreted as coterminal swap rate
                rate2, // interpreted as coterminal annuity
                rate3, // ignored 
                i, // needed to find right bit of forwards and discratios
                numberPathsPerBatch, // needed to find right bit of forwards and discratios
                forwards_cube, 
                discountRatios_cube);

            exerciseValues_matrix(j,i+numberPathsPreviouslyDone)=eValue;


        }

}




// non template function using example exercise values  of Bermudan swaption class
// note it is "adjoin" not "adjoint"! 


template<class D>
void adjoinExerciseValues_Bermudan_swaption_gold(const std::vector<D>& floatingInputs,
                                                 const std::vector<int>& integerInputs, 
                                                 int numberStepsAndRates,
                                                 const CubeConstFacade<D>&  forwards_cube,
                                                 const CubeConstFacade<D>&  discountRatios_cube,
                                                 const MatrixConstFacade<D>&  rate1_matrix,
                                                 const MatrixConstFacade<D>&  rate2_matrix,
                                                 const MatrixConstFacade<D>&  rate3_matrix,
                                                 int numberPathsPerBatch,
                                                 int numberPathsPreviouslyDone, 
                                                 const std::vector<int>&  exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
                                                 const std::vector<int>&  exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                                                 MatrixFacade<D>&  exerciseValues_matrix// output location
                                                 )
{
    exercise_value_gold<earlyExerciseBermudanPayerSwaptionGold<D>,D>(floatingInputs,
                         integerInputs,
                         numberStepsAndRates,
                         forwards_cube,
                         discountRatios_cube,
                         rate1_matrix,
                         rate2_matrix,
                         rate3_matrix,
                         numberPathsPerBatch,
                         numberPathsPreviouslyDone, 
                         exerciseIndices_vec, // the indices of the exercise times amongst the evolution times
                         exerciseIndicators_vec, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                         exerciseValues_matrix// output location
                         );
}



#endif
