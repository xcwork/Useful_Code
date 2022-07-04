
//
//
//                                                      cashFlowDiscounting_gold.h
// (c) Mark Joshi 2010,2011,2014
// This code is released under the GNU public licence version 3
//
//
// routines for discounting cash-flows 
// templatized on T to allow variation of float versus double

#ifndef CASH_FLOW_DISCOUNTING_H
#define CASH_FLOW_DISCOUNTING_H

#include <vector>
#include <gold/MatrixFacade.h>
#include <gold/Errors.h>
#include <cmath>

template<class T>
void cashFlowDiscounting_gold(const std::vector<int>& firstIndex, 
                                                                  const std::vector<int>& secondIndex,
                                                                  const std::vector<T>&  thetas, 
                                                                  const std::vector<T>&  discountRatios, 
                                                                  const std::vector<T>&   genFlows, 
                                                                  const std::vector<T>&   numeraireValues,
                                                                  int paths, 
                                                                  int numberSteps, 
                                                                  std::vector<T>&  discountedFlows, // output
                                                                  std::vector<T>&  summedDiscountedFlows,
                                                                  int startStep=0) // output
{
    if (firstIndex.size() != numberSteps)
        GenerateError("Size of firstIndex wrong in cashFlowDiscounting_gold.");

    if (secondIndex.size() != numberSteps)
        GenerateError("Size of secondIndex wrong in cashFlowDiscounting_gold.");
 
    if (thetas.size() != numberSteps)
        GenerateError("Size of theta wrong in cashFlowDiscounting_gold.");

    if (discountRatios.size() != numberSteps*(numberSteps+1)*paths)
        GenerateError("Size of discountRatios wrong in cashFlowDiscounting_gold.");

    if (genFlows.size() !=numberSteps*paths)
        GenerateError("Size of genFlows wrong in cashFlowDiscounting_gold.");

      if (numeraireValues.size() !=numberSteps*paths)
        GenerateError("Size of numeraireValues wrong in cashFlowDiscounting_gold.");
      
      discountedFlows.resize(genFlows.size());
      summedDiscountedFlows.resize(paths);

      CubeConstFacade<T> discounts(&discountRatios[0],numberSteps,numberSteps+1,paths);
      MatrixConstFacade<T> flows(&genFlows[0],numberSteps,paths);
      MatrixConstFacade<T> numeraireVals(&numeraireValues[0],numberSteps,paths);
      MatrixFacade<T> outFlows(&discountedFlows[0],numberSteps,paths);

      for (int p=0; p < paths; ++p)
      {
          T total=0.0;

          for (int s=startStep; s < numberSteps; ++s)
          {
                T cfSize = flows(s,p);
                T numVal = numeraireVals(s,p);

                int fIndex = firstIndex[s];
                int sIndex = secondIndex[s];
                T theta = thetas[s];

                T bRatio = discounts(s,s,p);

                T dRatio1 = discounts(s,fIndex,p)/bRatio;
                T dRatio2 = discounts(s,sIndex,p)/bRatio;

                T W1 = static_cast<T>(1.0)-theta;
      
                T df = pow(dRatio1,W1)*pow(dRatio2,theta);
                cfSize *= df;
                cfSize /= numVal;
                outFlows(s,p) = cfSize;
                total+= cfSize;

          }
            
          summedDiscountedFlows[p] = total;

      }
}


// usual assumption here that rate times and evolution times agree 
// They are discounted to RateTime[s] using the curve at the time of generation
// They are then discounted to a previous time targetIndices[s] using the numeraire.

template<class T>
void cashFlowDiscountingPartial_gold(
                                     const std::vector<int>& genTimeIndices, // indivces of times at which flows are generated
                                     const std::vector<int>& firstIndex, // rate time index leq payment date 
                                     const std::vector<int>& secondIndex, // rate time index > payment date 
                                     const std::vector<T>&  thetas, // interpolation fraction 
                                     const std::vector<int>& targetIndices, // rate time index to discount to 
                                     const CubeConstFacade<T>&  discountRatios,  // all the discount ratios 
                                     const MatrixConstFacade<T>&   genFlows, 
                                     const MatrixConstFacade<T>&   numeraireValues,
                                     int paths, 
                                     int numberSteps, 
                                     MatrixFacade<T>&  discountedFlows)
{
    int numberFlowsPerPath = static_cast<int>(genTimeIndices.size());
    if (firstIndex.size() != numberFlowsPerPath)
        GenerateError("Size of firstIndex wrong in cashFlowDiscountingPartial_gold.");

    if (secondIndex.size() != numberFlowsPerPath)
        GenerateError("Size of secondIndex wrong in cashFlowDiscountingPartial_gold.");
 
    if (thetas.size() != numberFlowsPerPath)
        GenerateError("Size of theta wrong in cashFlowDiscountingPartial_gold.");

    if (discountRatios.numberColumns() != paths)
        GenerateError("Columns of discountRatios wrong in cashFlowDiscountingPartial_gold.");
  
    if (discountRatios.numberRows() !=numberSteps+1)
        GenerateError("Rows of discountRatios wrong in cashFlowDiscountingPartial_gold.");
  
    if (discountRatios.numberLayers() != numberSteps)
        GenerateError("Layers of discountRatios wrong in cashFlowDiscountingPartial_gold.");

    if (genFlows.columns() !=paths)
        GenerateError("Cols of genFlows wrong in cashFlowDiscountingPartial_gold.");
 
    if (genFlows.rows() !=numberSteps)
        GenerateError("Rows of genFlows wrong in cashFlowDiscountingPartial_gold.");
 
    if (numeraireValues.columns() !=paths)
        GenerateError("Cols of numeraireValues wrong in cashFlowDiscountingPartial_gold.");
 
    if (numeraireValues.rows() !=numberSteps)
        GenerateError("Rows of numeraireValues wrong in cashFlowDiscountingPartial_gold.");
    
	if (targetIndices.size() != numberFlowsPerPath)
		GenerateError("We need one target index per flow for each path in cashFlowDiscountingPartial_gold.");
    
      for (int p=0; p < paths; ++p)
      {


          for (int flowNumber=0; flowNumber < numberFlowsPerPath; ++flowNumber)
          {
                T cfSize = genFlows(flowNumber,p);
            
                int fIndex = firstIndex[flowNumber];
                int sIndex = secondIndex[flowNumber];
                double theta = static_cast<double>(thetas[flowNumber]);

                int s= genTimeIndices[flowNumber];

                T bRatio = discountRatios(s,s,p);

                T dRatio1 = discountRatios(s,fIndex,p)/bRatio;
                T dRatio2 = discountRatios(s,sIndex,p)/bRatio;

        //        double W1 = 1.0-theta;

		//		double logDf = W1*log(static_cast<double>(dRatio1))+theta*log(static_cast<double>(dRatio2));
		//		double df = exp(logDf);
      
        //        T df = pow(dRatio1,W1)*pow(dRatio2,theta);
                
                // discount to s using prevailing discount curve
                // at the time of the cash-flow's generation

				double rat = dRatio2/dRatio1;
				#ifdef _DEBUG
if (fp_isnan(cfSize))
		std::cout << "cashFlowDiscountingPartial_gold rat isnan, " <<dRatio1 << "," << dRatio2  << ","<< p << ","
		<< flowNumber<< ", "<< "\n";
#endif

				double df = dRatio1*exp((std::log((double)rat))*theta);

                double dfcfSize = cfSize*df;

                // now use numeraire values to discount to exercise time 
      
                double numRatio = numeraireValues(targetIndices[flowNumber],p)/numeraireValues(s,p);
				double df2cfSize = numRatio*dfcfSize;

#ifdef _DEBUG
	if (fp_isnan(df2cfSize))
		std::cout << "cashFlowDiscountingPartial_gold isnan, " << cfSize << ","<< p << ","
		<< flowNumber<< ", "<<dRatio1 << "," << dRatio2  <<"," << theta << ","<< df <<"," << numRatio << "," <<df2cfSize <<","<< numeraireValues(s,p)<< "\n";
#endif

                discountedFlows(flowNumber,p) = static_cast<T>(df2cfSize);
         

          }
            
       

      }
}



// this turns a vector of payment times into indices of rate times and interpolation/extrapolation weights

template<class T> 
void generateCashFlowIndicesAndWeights(std::vector<int>& firstIndex, 
                                                                            std::vector<int>& secondIndex,
                                                                            std::vector<T>&  thetas, 
                                                                            const std::vector<T>& rateTimes,
                                                                            const std::vector<T>& paymentTimes
                                                                            )
{
    int  numberTimes = static_cast<int>(paymentTimes.size());

    firstIndex.resize(numberTimes);
    secondIndex.resize(numberTimes);
    thetas.resize(numberTimes);

    for (int i=0; i < numberTimes; ++i)
    {
        T time = paymentTimes[i];

        if (time < rateTimes[0])
        {
            firstIndex[i] = 0;
            secondIndex[i] = 1;
        }
        else
            if (time > rateTimes[numberTimes-1])
            {
                   firstIndex[i] = numberTimes-1;
                  secondIndex[i] = numberTimes;
            }
            else
            {
                int j=1;

                while (time > rateTimes[j])  // the above implies that this must eventually be false
                    ++j;

                 firstIndex[i]=j-1;
                 secondIndex[i]=j;
                
            }

        T theta = (time- rateTimes[firstIndex[i]])/(rateTimes[secondIndex[i]]-rateTimes[firstIndex[i]]);
        thetas[i] = theta;
    }
}

// We will often want to discount exercise values and cash-flows first to 
// their time of generation using the discount curve at their time of 
// generation, and then to the last exercise date <= time of generation.
// This routine gives the index amongst exercise times of that exercise date. 
// For generation indices before first exercise date the index -1
// The return value fo the function is the first non-negative index. 
int findExerciseTimeIndicesFromPaymentTimeIndices(const std::vector<int>& genTimeIndex, 
                                                   const std::vector<int>& exerciseIndices,
                                                   const std::vector<int>& exerciseIndicators, // redundant info but sometimes easier to work with
                                                   std::vector<int>& stepToExerciseIndices
                                                   );


#endif
