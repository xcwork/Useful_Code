
///                                    KooQl main.cpp

#include <ql/qldefines.hpp>
#include <ql/version.hpp>
#ifdef BOOST_MSVC
#  include <ql/auto_link.hpp>
#endif
#include <ql/models/marketmodels/all.hpp>
#include <ql/models/marketmodels/products/multistep/multisteptarn.hpp>
#include <ql/methods/montecarlo/genericlsregression.hpp>
#include <ql/legacy/libormarketmodels/lmlinexpcorrmodel.hpp>
#include <ql/legacy/libormarketmodels/lmextlinexpvolmodel.hpp>
#include <ql/time/schedule.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/simpledaycounter.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/blackcalculator.hpp>
#include <ql/utilities/dataformatters.hpp>
#include <ql/math/integrals/segmentintegral.hpp>
#include <ql/math/statistics/convergencestatistics.hpp>
#include <ql/termstructures/volatility/abcd.hpp>
#include <ql/termstructures/volatility/abcdcalibration.hpp>
#include <ql/math/functional.hpp>
#include <ql/math/optimization/simplex.hpp>
#include <ql/quotes/simplequote.hpp>
#include <sstream>
#include <iostream>
#include <ctime>
#include "marketmodelevolvercuda.h"
#include "BrownianGeneratorSobolCUDA.h"
#include <float.h>
#include "Utilities.h"
#include "TARNCUDAPricer.h"

using namespace QuantLib;



int Caplets(Size numberRates ,
            Size paths,
    int pathBatchSize ,
    int pathOffset,
    int mersenneSeed ,
    bool scramble ,
    Size numberOfFactors,
    bool useGPU
)
{
     // first set up the product
    // we are pricing a set of caplets
    
    // how many caplets 
 

    // the accrual for each caplet
    Real accrual = 0.5;
    // here we take all caplets to have the same accrual 
    // but the code can handle different for each
    std::vector<Real> accruals(numberRates,accrual);

    // the reset time of the first caplet
    Real firstTime = 0.5;

    // the rate times specify the start and end times of rate underlying each caplet
    // the end of one is  the start of the next
    std::vector<Real> rateTimes(numberRates+1);
    for (Size i=0; i < rateTimes.size(); ++i)
        rateTimes[i] = firstTime + i*accrual;

    // set up the vector of the times at which the caplets pay-off
    std::vector<Real> paymentTimes(numberRates);
    for (Size i=0; i < paymentTimes.size(); ++i)
        paymentTimes[i] = firstTime + (i+1)*accrual;

    // set the strikes of the caplets
    // we use the same class with a different pay-off for floorlets and caplets
    // or indeed any derivative on a single rate
    Real fixedRate = 0.05;
    std::vector<boost::shared_ptr<Payoff> > optionletPayoffs(numberRates);
   
    for (Size i=0; i<numberRates; ++i)
    {
        optionletPayoffs[i] = boost::shared_ptr<Payoff>(new
                                                                    PlainVanillaPayoff(Option::Call, fixedRate));
    }


// we created the data to go into the product, now create the product itself
     MultiStepOptionlets    product( rateTimes,
                                                                        accruals,
                                                                        paymentTimes,
                                                                          optionletPayoffs );

// we need an object that keeps everything doing the same thing
// this specifies what the product expects to be passed
// what times to evolve to 
// what rates to evolve
    EvolutionDescription evolution = product.evolution();


    // parameters for models


   
    std::cout << " paths, " << paths << "\n";


    // set up a calibration, this would typically be done by using a calibrator

    Real rateLevel =0.06;
    Real rateTop = 0.06;
    Real rateStep = (rateTop - rateLevel)/numberRates;

    Real initialNumeraireValue = 0.95;

    Real volLevel = 0.1;
    Real longTermCorr = 0.5;
    Real beta = 0.2;
    Real gamma = 1.0;


    Spread displacementLevel =0.00;

    // set up vectors 
    std::vector<Rate> initialRates(numberRates,rateLevel);

    for (Size i=0; i < numberRates; ++i)
        initialRates[i] = rateLevel + rateStep*i;

    std::vector<Volatility> volatilities(numberRates, volLevel);
    std::vector<Spread> displacements(numberRates, displacementLevel);

    ExponentialForwardCorrelation correlations(rateTimes,volLevel, beta,gamma);

    FlatVol  calibration(
        volatilities,
        boost::shared_ptr<PiecewiseConstantCorrelation>(new  ExponentialForwardCorrelation(correlations)),
        evolution,
        numberOfFactors,
        initialRates,
        displacements);

    boost::shared_ptr<MarketModel> marketModel(new FlatVol(calibration));

    // calibration is now set up

    std::vector<Real> todaysDiscounts(numberRates+1);
    todaysDiscounts[0] = initialNumeraireValue;
    for (Size i=0; i < numberRates; ++i)
        todaysDiscounts[i+1] = todaysDiscounts[i]/(1+accruals[i]*initialRates[i]);

    //unsigned int control_word;
   // int err;
  //  err = _controlfp_s(&control_word, 0, 0);

   // err = _controlfp_s(&control_word,_EM_UNDERFLOW + _EM_INEXACT,_MCW_EM);


    // the evolver will actually evolve the rates 
    // 
    MarketModelEvolverLMMPC_CUDA  evolver(marketModel, mersenneSeed, pathBatchSize,pathOffset,scramble,useGPU);

    
// the accounting engine takes in a shared pointer 
// so we create a copy of the evolver 
// and turn it into a shared_ptr to the base class
 
   evolver.generatePathsIfNeeded();

#ifdef _DEBUG
   evolver.dumpCache();
#endif


    boost::shared_ptr<MarketModelEvolver> evolverPtr(new MarketModelEvolverLMMPC_CUDA(evolver));


    // we need the initial numeraire value, simply to multiply the
    // expectation by at the end 
     // the accounter also takes in a clone ptr of the product
    // clone ptr ensures that different ptrs to the product
    // don't confuse each other 
    AccountingEngine accounter(evolverPtr,
        Clone<MarketModelMultiProduct>(product),
        initialNumeraireValue);

    int t2= clock();

    SequenceStatisticsInc stats;

    accounter.multiplePathValues (stats,paths);

    int t3 = clock();

    std::vector<Real> modelValues(numberRates);

    for (Size i=0; i < numberRates; ++i)
    {


             modelValues[i] =blackFormula(Option::Call, fixedRate,
                                                                    initialRates[i],
                                                                    volatilities[i]*std::sqrt(rateTimes[i]),
                                                                    todaysDiscounts[i+1]*accruals[i],
                                                                    displacements[i]);

   
        
    }

    std::vector<Real> means(stats.mean());

    std::cout << "caplets\n means ,    standard errors\n";

    for (Size i=0; i < means.size(); ++i)
        std::cout << means[i] << ", " << stats.errorEstimate()[i]<< ", " <<modelValues[i] <<  "\n";


    std::cout << " time to price, " << (t3-t2)/static_cast<Real>(CLOCKS_PER_SEC)<< ", seconds.\n";

MarketModelEvolver* ptr = &(*evolverPtr);

    std::cout << " total time in CUDAPart " << dynamic_cast<MarketModelEvolverLMMPC_CUDA*>(ptr)->totalTimeInCudaPart() << "\n";

    //char c;
   // std::cin >>c;

    return 0;
}


int Caplets2(Size numberRates ,
            Size paths,
    int pathBatchSize ,
    int pathOffset ,
    int mersenneSeed ,
    bool scramble 
    ,    Size numberOfFactors)
{
     // first set up the product
    // we are pricing a set of caplets
    
    // how many caplets 
   

    // the accrual for each caplet
    Real accrual = 0.5;
    // here we take all caplets to have the same accrual 
    // but the code can handle different for each
    std::vector<Real> accruals(numberRates,accrual);

    // the reset time of the first caplet
    Real firstTime = 0.5;

    // the rate times specify the start and end times of rate underlying each caplet
    // the end of one is  the start of the next
    std::vector<Real> rateTimes(numberRates+1);
    for (Size i=0; i < rateTimes.size(); ++i)
        rateTimes[i] = firstTime + i*accrual;

    // set up the vector of the times at which the caplets pay-off
    std::vector<Real> paymentTimes(numberRates);
    for (Size i=0; i < paymentTimes.size(); ++i)
        paymentTimes[i] = firstTime + (i+1)*accrual;

    // set the strikes of the caplets
    // we use the same class with a different pay-off for floorlets and caplets
    // or indeed any derivative on a single rate
    Real fixedRate = 0.05;
    std::vector<boost::shared_ptr<Payoff> > optionletPayoffs(numberRates);
   
    for (Size i=0; i<numberRates; ++i)
    {
        optionletPayoffs[i] = boost::shared_ptr<Payoff>(new
                                                                    PlainVanillaPayoff(Option::Call, fixedRate));
    }


// we created the data to go into the product, now create the product itself
     MultiStepOptionlets    product( rateTimes,
                                                                        accruals,
                                                                        paymentTimes,
                                                                          optionletPayoffs );

// we need an object that keeps everything doing the same thing
// this specifies what the product expects to be passed
// what times to evolve to 
// what rates to evolve
    EvolutionDescription evolution = product.evolution();
    std::vector<Size> numeraires(moneyMarketMeasure(evolution));

    // parameters for models
   
    std::cout << " paths, " << paths << "\n";


    // set up a calibration, this would typically be done by using a calibrator

    Real rateLevel =0.06;
    Real rateTop = 0.06;
    Real rateStep = (rateTop - rateLevel)/numberRates;

    Real initialNumeraireValue = 0.95;

    Real volLevel = 0.1;
    Real longTermCorr = 0.5;
    Real beta = 0.2;
    Real gamma = 1.0;
    

    Spread displacementLevel =0.00;

    // set up vectors 
    std::vector<Rate> initialRates(numberRates,rateLevel);

    for (Size i=0; i < numberRates; ++i)
        initialRates[i] = rateLevel + rateStep*i;

    std::vector<Volatility> volatilities(numberRates, volLevel);
    std::vector<Spread> displacements(numberRates, displacementLevel);

    ExponentialForwardCorrelation correlations(rateTimes,volLevel, beta,gamma);

    FlatVol  calibration(
        volatilities,
        boost::shared_ptr<PiecewiseConstantCorrelation>(new  ExponentialForwardCorrelation(correlations)),
        evolution,
        numberOfFactors,
        initialRates,
        displacements);

    boost::shared_ptr<MarketModel> marketModel(new FlatVol(calibration));

    // calibration is now set up

    std::vector<Real> todaysDiscounts(numberRates+1);
    todaysDiscounts[0] = initialNumeraireValue;
    for (Size i=0; i < numberRates; ++i)
        todaysDiscounts[i+1] = todaysDiscounts[i]/(1+accruals[i]*initialRates[i]);

    //unsigned int control_word;
   // int err;
  //  err = _controlfp_s(&control_word, 0, 0);

   // err = _controlfp_s(&control_word,_EM_UNDERFLOW + _EM_INEXACT,_MCW_EM);




      SobolCudaBrownianGeneratorFactory bgFactory( pathBatchSize,
                           pathOffset,
                           mersenneSeed,
                           scramble);

   LogNormalFwdRatePc   evolver(marketModel,
                                                            bgFactory,
                                                            numeraires);

// the accounting engine takes in a shared pointer 
// so we create a copy of the evolver 
// and turn it into a shared_ptr to the base class
 
    boost::shared_ptr<MarketModelEvolver> evolverPtr(new LogNormalFwdRatePc(evolver));
/*
#ifdef _DEBUG
    for (Size i=0; i < paths; ++i)
    {
        std::cout << "\n" << i << "\n";
        evolver.startNewPath();
        for (Size s=0; s < numberRates ; ++s)
        {
            evolver.advanceStep();
            for (Size r=0;r < numberRates; ++r)
            {
                Real rateVal= evolver.currentState().forwardRate(r);
                std::cout << rateVal << ",";
            }
             std::cout << "\n";
        }
       
    }

#endif

*/
    // we need the initial numeraire value, simply to multiply the
    // expectation by at the end 
     // the accounter also takes in a clone ptr of the product
    // clone ptr ensures that different ptrs to the product
    // don't confuse each other 
    AccountingEngine accounter(evolverPtr,
        Clone<MarketModelMultiProduct>(product),
        initialNumeraireValue);

    int t2= clock();

    SequenceStatisticsInc stats;

    accounter.multiplePathValues (stats,paths);

    int t3 = clock();

    std::vector<Real> modelValues(numberRates);

    for (Size i=0; i < numberRates; ++i)
    {


             modelValues[i] =blackFormula(Option::Call, fixedRate,
                                                                    initialRates[i],
                                                                    volatilities[i]*std::sqrt(rateTimes[i]),
                                                                    todaysDiscounts[i+1]*accruals[i],
                                                                    displacements[i]);

   
        
    }

    std::vector<Real> means(stats.mean());

    std::cout << "caplets\n means ,    standard errors\n";

    for (Size i=0; i < means.size(); ++i)
        std::cout << means[i] << ", " << stats.errorEstimate()[i]<< " ," <<modelValues[i] <<  "\n";


    std::cout << " time to price, " << (t3-t2)/static_cast<Real>(CLOCKS_PER_SEC)<< ", seconds.\n";



    //char c;
   // std::cin >>c;

    return 0;
}


Real TARNAllOnGpu(Size numberRates ,
            Size paths,
    int pathBatchSize ,
    int pathOffset,
    int mersenneSeed ,
    bool scramble ,
    Size numberOfFactors,
    int numberOfThreads,
    Real paymentDelay,
    bool multiBatch,
    bool useShared,
    bool mergeDiscounts,
    bool newBridge,
    const std::vector<int>& deviceIndex,
	bool fermiArch,
	bool sharedForDiscounting
)
{
     // first set up the product

    // the accrual for each rate
    Real accrual = 0.5;
    // here we take all caplets to have the same accrual 
    // but the code can handle different for each
    std::vector<Real> accruals(numberRates,accrual);

    // the reset time of the first rate
    Real firstTime = 0.5;

    // the rate times specify the start and end times of rate underlying each caplet
    // the end of one is  the start of the next
    std::vector<Real> rateTimes(numberRates+1);
    for (Size i=0; i < rateTimes.size(); ++i)
        rateTimes[i] = firstTime + i*accrual;

    // set up the vector of the times at which the coupons pay-off
    std::vector<Real> paymentTimes(numberRates);
    for (Size i=0; i < paymentTimes.size(); ++i)
        paymentTimes[i] = firstTime + (i+1)*accrual+paymentDelay;


    // create dummy product 
    // set the strikes of the caplets
    // we use the same class with a different pay-off for floorlets and caplets
    // or indeed any derivative on a single rate
    Real fixedRate = 0.05;
    std::vector<boost::shared_ptr<Payoff> > optionletPayoffs(numberRates);
   
    for (Size i=0; i<numberRates; ++i)
    {
        optionletPayoffs[i] = boost::shared_ptr<Payoff>(new
                                                                    PlainVanillaPayoff(Option::Call, fixedRate));
    }


// we created the data to go into the product, now create the product itself
     MultiStepOptionlets    product( rateTimes,
                                                                        accruals,
                                                                        paymentTimes,
                                                                          optionletPayoffs );

// we need an object that keeps everything doing the same thing
// this specifies what the product expects to be passed
// what times to evolve to 
// what rates to evolve
    EvolutionDescription evolution = product.evolution();

    ///////////////

    // now set up data for TARN

     std::vector<float> firstCashFlowsTimes(stlVecCastStlVec<float,double>(paymentTimes));
     std::vector<float> secondCashFlowsTimes(firstCashFlowsTimes);
     std::vector<float> auxData_TARN;

     float totalCoupon_ = 0.15f;
     auxData_TARN.push_back(totalCoupon_);
     
     float strike_ = 0.1f;
     auxData_TARN.push_back(strike_);

     float multiplier_ =2.0f;
     auxData_TARN.push_back(multiplier_);

     // two legs
     for (Size i=0; i < accruals.size(); ++i)
            auxData_TARN.push_back(static_cast<float>(accruals[i]));

     for (Size i=0; i < accruals.size(); ++i)
            auxData_TARN.push_back(static_cast<float>(accruals[i]));


    // parameters for models


   
    std::cout << " paths, " << paths << "\n";


    // set up a calibration, this would typically be done by using a calibrator

    Real rateLevel =0.04;
    Real rateTop = 0.06;
    Real rateStep = (rateTop - rateLevel)/numberRates;

    Real initialNumeraireValue = 0.95;

    Real volLevel = 0.1;
    Real longTermCorr = 0.5;
    Real beta = 0.2;
    Real gamma = 1.0;


    Spread displacementLevel =0.00;

    // set up vectors 
    std::vector<Rate> initialRates(numberRates,rateLevel);

    for (Size i=0; i < numberRates; ++i)
        initialRates[i] = rateLevel + rateStep*i;

    std::vector<Volatility> volatilities(numberRates, volLevel);
    std::vector<Spread> displacements(numberRates, displacementLevel);

    ExponentialForwardCorrelation correlations(rateTimes,volLevel, beta,gamma);

    FlatVol  calibration(
        volatilities,
        boost::shared_ptr<PiecewiseConstantCorrelation>(new  ExponentialForwardCorrelation(correlations)),
        evolution,
        numberOfFactors,
        initialRates,
        displacements);

    boost::shared_ptr<MarketModel> marketModel(new FlatVol(calibration));

    // calibration is now set up

    std::vector<Real> todaysDiscounts(numberRates+1);
    todaysDiscounts[0] = initialNumeraireValue;
    for (Size i=0; i < numberRates; ++i)
        todaysDiscounts[i+1] = todaysDiscounts[i]/(1+accruals[i]*initialRates[i]);

    // we set up an evolver as this does all the relevant data preprocessing 
    // 
    bool useGPU = true;
    MarketModelEvolverLMMPC_CUDA  evolver(marketModel, mersenneSeed, pathBatchSize,pathOffset,scramble,useGPU);

    int batches = paths / pathBatchSize;
    if (paths % pathBatchSize >0)
        ++batches;

// we are doing randomized QMC
    std::vector<float> batchValues_vec;
    int t2= clock();
    TARNPricerRoutineMultiThread( evolver,
                                         batches,
                                         auxData_TARN,
                                         firstCashFlowsTimes, 
                                         secondCashFlowsTimes,
                                         numberOfThreads,
                                         batchValues_vec,
                                         multiBatch,
                                         useShared,
                                         mergeDiscounts,
                                         newBridge,
                                         deviceIndex,
										 fermiArch,
										 sharedForDiscounting
                        );


    


    int t3 = clock();

    std::vector<double> v(1);

    SequenceStatistics stats;
   
    for (Size i=0; i < batchValues_vec.size(); ++i)
    {
        v[0] = batchValues_vec[i]*initialNumeraireValue;
        std::cout << i << ","<<  v[0] << ","; 
        stats.add(v);
    }

     std::vector<Real> means(stats.mean());

    std::cout << "\nTARN \n mean ,    standard error\n";

    for (Size i=0; i < means.size(); ++i)
        std::cout << means[i] << ", " << stats.errorEstimate()[i] <<  ",";

    Real timeTaken = (t3-t2)/static_cast<Real>(CLOCKS_PER_SEC);

    std::cout << " \ntime to price, " << timeTaken << ", seconds.\n";


    return timeTaken;
}

int TARNQL(Size numberRates ,
                      Size paths,  
                      int mersenneSeed,
                      int pathBatchSize,
                      int pathOffset,
                      bool useGPU, 
                      bool scramble,
                      Size numberOfFactors,
                      Real paymentDelay
)
{
     // first set up the product

    // the accrual for each rate
    Real accrual = 0.5;
    // here we take all caplets to have the same accrual 
    // but the code can handle different for each
    std::vector<Real> accruals(numberRates,accrual);

    // the reset time of the first rate
    Real firstTime = 0.5;

    // the rate times specify the start and end times of rate underlying each caplet
    // the end of one is  the start of the next
    std::vector<Real> rateTimes(numberRates+1);
    for (Size i=0; i < rateTimes.size(); ++i)
        rateTimes[i] = firstTime + i*accrual;

    // set up the vector of the times at which the coupons pay-off
    std::vector<Real> paymentTimes(numberRates);
    for (Size i=0; i < paymentTimes.size(); ++i)
        paymentTimes[i] = firstTime + (i+1)*accrual+paymentDelay;

    std::vector<Real> floatingPaymentTimes(paymentTimes);



     float totalCoupon_ = 0.15f;
     
     float strike_ = 0.1f;

     float multiplier_ =2.0f;


     std::vector<Real> strikes(numberRates,strike_);
     std::vector<Real> multipliers(numberRates,multiplier_);
     std::vector<Real> floatingSpreads(numberRates,0.0);


         MultiStepTarn tarn( rateTimes,
                         accruals,
                         accruals,                         
                         paymentTimes,                         
                         floatingPaymentTimes,
                         totalCoupon_,
                         strikes,
                        multipliers,
                        floatingSpreads);

     EvolutionDescription evolution(tarn.evolution());

    // parameters for models
    std::cout << "QL paths, " << paths << "\n";


    // set up a calibration, this would typically be done by using a calibrator

    Real rateLevel =0.04;
    Real rateTop = 0.06;
    Real rateStep = (rateTop - rateLevel)/numberRates;

    Real initialNumeraireValue = 0.95;

    Real volLevel = 0.1;
    Real longTermCorr = 0.5;
    Real beta = 0.2;
    Real gamma = 1.0;


    Spread displacementLevel =0.00;

    // set up vectors 
    std::vector<Rate> initialRates(numberRates,rateLevel);

    for (Size i=0; i < numberRates; ++i)
        initialRates[i] = rateLevel + rateStep*i;

    std::vector<Volatility> volatilities(numberRates, volLevel);
    std::vector<Spread> displacements(numberRates, displacementLevel);

    ExponentialForwardCorrelation correlations(rateTimes,volLevel, beta,gamma);

    FlatVol  calibration(
        volatilities,
        boost::shared_ptr<PiecewiseConstantCorrelation>(new  ExponentialForwardCorrelation(correlations)),
        evolution,
        numberOfFactors,
        initialRates,
        displacements);

    boost::shared_ptr<MarketModel> marketModel(new FlatVol(calibration));

    // calibration is now set up

    std::vector<Real> todaysDiscounts(numberRates+1);
    todaysDiscounts[0] = initialNumeraireValue;
    for (Size i=0; i < numberRates; ++i)
        todaysDiscounts[i+1] = todaysDiscounts[i]/(1+accruals[i]*initialRates[i]);

    // we set up an evolver as this does all the relevant data preprocessing  


   
    MarketModelEvolverLMMPC_CUDA  evolver(marketModel, mersenneSeed, pathBatchSize,pathOffset,scramble,useGPU);

   
    boost::shared_ptr<MarketModelEvolver> evolverPtr(new MarketModelEvolverLMMPC_CUDA(evolver));

    // we need the initial numeraire value, simply to multiply the
    // expectation by at the end 
     // the accounter also takes in a clone ptr of the product
    // clone ptr ensures that different ptrs to the product
    // don't confuse each other 
    AccountingEngine accounter(evolverPtr,
        Clone<MarketModelMultiProduct>(tarn),
        initialNumeraireValue);


    int t2 = clock();


    std::vector<double> v(1);

    SequenceStatisticsInc stats;

    accounter.multiplePathValues(stats,paths);
    int t3 = clock();


     std::vector<Real> means(stats.mean());

    std::cout << "\nTARN \n mean ,    standard error\n";

    for (Size i=0; i < means.size(); ++i)
        std::cout << means[i] << ", " << stats.errorEstimate()[i] <<  ",";


    std::cout << " \ntime to price, " << (t3-t2)/static_cast<Real>(CLOCKS_PER_SEC)<< ", seconds.\n";


    return 0;
}




int main()
{
    
    unsigned int control_word;
    int err;
    err = _controlfp_s(&control_word, 0, 0); 
    err = _controlfp_s(&control_word,_EM_UNDERFLOW + _EM_INEXACT,_MCW_EM);  
    Size numberRates =32;
    int pathBatchSize = 65536-1;
    Size paths=2*pathBatchSize;
    int pathOffset =0;
    int mersenneSeed = 3243;
    bool scramble = false;
    Size numberOfFactors =std::min<Size>(numberRates,5);
    bool useGPU=true;
    int numberOfThreads=1;
	bool fermiaArch = false;

    std::cout << " number of threads " << numberOfThreads << "\n";

    std::vector<int> deviceIndex(numberOfThreads);

	for (int i=0; i < numberOfThreads; ++i)
	    deviceIndex[i] =i;

    Real paymentDelay = 0.25;
    bool multiBatch = true;
    bool useShared=true;
    bool mergeDiscounts=true;
    bool newBridge=true;
	bool sharedForDiscounting=true;

    int loops = 1;
    Real time =0.0;
    Real timesq =0.0;
    for (int l=0; l < loops; ++l)
    {

        Real t = TARNAllOnGpu(numberRates ,paths, pathBatchSize , pathOffset , mersenneSeed ,scramble 
            ,numberOfFactors,numberOfThreads,paymentDelay,multiBatch,useShared,mergeDiscounts,newBridge,deviceIndex,fermiaArch, sharedForDiscounting);
        time += t;
        timesq +=t*t;
    }

    if (loops > 1)
    {
        Real av = time/loops;
        Real Esq = timesq/loops;
        Real sde = sqrt((Esq - av*av)/loops);

        std::cout << "Mean time " << av << " st err " << sde << "\n";
    }
    std::cout << "for " << paths << " total paths.\n";

 /*
    TARNQL(numberRates ,
                   paths,  
                     mersenneSeed,
                     pathBatchSize,
                     pathOffset,
                     useGPU, 
                     scramble,
                     numberOfFactors,paymentDelay);
           */
 //   Caplets(numberRates ,paths, pathBatchSize , pathOffset , mersenneSeed ,scramble ,numberOfFactors,useGPU);
    //Caplets(numberRates ,paths, pathBatchSize , pathOffset , mersenneSeed ,scramble ,numberOfFactors,!useGPU);
  //  Caplets2(numberRates ,paths, pathBatchSize , pathOffset , mersenneSeed ,scramble,numberOfFactors);

  //  char c;
 //   std::cin >> c;
    return 0;
}

