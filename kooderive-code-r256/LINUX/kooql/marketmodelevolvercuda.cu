#include "marketmodelevolvercuda.h"
#include <ql/models/marketmodels/marketmodel.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <gold/MatrixFacade.h>
#include <ql/models/marketmodels/driftcomputation/lmmdriftcalculator.hpp>
#include "ConvertsQLToThrust.h"
#include <gold/LMM_evolver_full_gold.h>
#include <LMM_evolver_full.h>
#include <iostream>
#include <cmath>
#include <limits>


MarketModelEvolverLMMPC_CUDA::MarketModelEvolverLMMPC_CUDA(const boost::shared_ptr<MarketModel>& marketModel,
                                                           int MersenneSeed, // seed is for scrambling only
                                                           int pathBatchSize, // how many paths to generate at a time, this is constrainted by the memory of your GPU
                                                           int pathOffset,
                                                           bool scramble,
                                                           bool useGPU
                                                           )
                                                           : marketModel_(marketModel), 
                                                           pathBatchSize_(pathBatchSize),
                                                           pathOffset_(pathOffset),
                                                           totalTimeInCudaPart_(0.0),
                                                           numeraires_(marketModel->numberOfSteps()),
                                                           initialStep_(0),
                                                           fixedDrifts_(marketModel->numberOfSteps()),
                                                           rng_(MersenneSeed),
                                                           scramble_(scramble),
                                                           currentPath_(pathBatchSize),
                                                           numberOfRates_(marketModel->numberOfRates()),
                                                           numberOfFactors_(marketModel->numberOfFactors()),
                                                           curveState_(marketModel->evolution().rateTimes()),
                                                           currentStep_(0),
                                                           forwards_(marketModel->initialRates()),
                                                           displacements_(marketModel->displacements()),
                                                           initialLogForwards_(numberOfRates_),
                                                           initialDrifts_(numberOfRates_), 
                                                           alive_(marketModel->numberOfSteps()),
                                                           currentForwards_(numberOfRates_),
                                                           stepsForEvolution_(marketModel->numberOfSteps()),
                                                           pseudoRoots_host_(stepsForEvolution_*numberOfRates_*numberOfFactors_),
                                                           fixedDrifts_host_(stepsForEvolution_*numberOfRates_),
                                                           displacements_host_(numberOfRates_),
                                                           initial_rates_host_(numberOfRates_),
                                                           initial_log_rates_host_(numberOfRates_),
                                                           taus_host_(numberOfRates_),
                                                           initial_drifts_host_(numberOfRates_),
                                                           alive_host_(stepsForEvolution_),
                                                           scrambler_host_(stepsForEvolution_*numberOfFactors_),
                                                           evolved_rates_host_(pathBatchSize_*stepsForEvolution_*numberOfRates_),
                                                           useGPU_(useGPU)
{

    numeraires_ = moneyMarketMeasure(marketModel->evolution());

    marketModel->evolution().firstAliveRate();

    curveState_.setOnForwardRates(forwards_);

    std::vector<Matrix> pseudos;

    for (int i=0; i < stepsForEvolution_; ++i)
    {
        alive_[i] = marketModel->evolution().firstAliveRate()[i];
        pseudos.push_back(marketModel->pseudoRoot(i));

        MatrixFacade<float> fixedDrifts(&fixedDrifts_host_[0],stepsForEvolution_,numberOfRates_);

        for (Size r=0; r < numberOfRates_; ++r)
        {
            Real variance =0.0;
            for (Size f=0; f < numberOfFactors_; ++f)
                variance += pseudos[i][r][f]*pseudos[i][r][f];

            fixedDrifts(i,r) =static_cast<float>(-0.5*variance);

      
        }

    }
    
    cubeFromMatrixVector<float>(pseudos,pseudoRoots_host_);

    for (Size r=0; r < numberOfRates_; ++r)
        initialLogForwards_[r] = log(forwards_[r]);

    LMMDriftCalculator driftCal(pseudos[0],
        displacements_,
        marketModel->evolution().rateTaus(),
        numeraires_[0], 
        alive_[0]);
    std::vector<Real> initialDrifts(numberOfRates_);

    driftCal.computeReduced(forwards_,
        initialDrifts);

    for (Size i=0; i < numberOfRates_; ++i)
    {
        displacements_host_[i] = static_cast<float>(marketModel->displacements()[i]);
        initial_rates_host_[i] =  static_cast<float>(forwards_[i]);
        initial_log_rates_host_[i] = static_cast<float>(initialLogForwards_[i]);
        taus_host_[i] = static_cast<float>(marketModel->evolution().rateTaus()[i]);
        initial_drifts_host_[i] = static_cast<float>(initialDrifts[i]);
        alive_host_[i] = alive_[i];

    }

    int j=1;
    powerOfTwoForVariates_=0;

    while (j < stepsForEvolution_)
    {
        j*=2;
        ++powerOfTwoForVariates_;
    }

    for (int i=0; i < static_cast<int>(scrambler_host_.size()); ++i)
                scrambler_host_[i] = 0;

    if (!useGPU)
    {
    pseudoRoots_vec_.resize(pseudoRoots_host_.size());
    std::copy(pseudoRoots_host_.begin(),pseudoRoots_host_.end(),pseudoRoots_vec_.begin() );

     fixedDrifts_vec_.resize(fixedDrifts_host_.size());
     std::copy(fixedDrifts_host_.begin(),fixedDrifts_host_.end(),fixedDrifts_vec_.begin() );

      displacements_vec_.resize(displacements_host_.size());
      std::copy(displacements_host_.begin(),displacements_host_.end(),displacements_vec_.begin() );
 
       initial_rates_vec_.resize(initial_rates_host_.size());
       std::copy(initial_rates_host_.begin(),initial_rates_host_.end(),initial_rates_vec_.begin() );

       initial_log_rates_vec_.resize(initial_log_rates_host_.size());
       std::copy(initial_log_rates_host_.begin(),initial_log_rates_host_.end(),initial_log_rates_vec_.begin() );
    
       taus_vec_.resize(taus_host_.size());
       std::copy(taus_host_.begin(),taus_host_.end(),taus_vec_.begin() );
  
       initial_drifts_vec_.resize(initial_drifts_host_.size());
       std::copy(initial_drifts_host_.begin(),initial_drifts_host_.end(),initial_drifts_vec_.begin() );

       alive_vec_.resize(alive_host_.size());
       std::copy(alive_host_.begin(),alive_host_.end(),alive_vec_.begin() );


    }
            
    scrambler_vec_.resize(scrambler_host_.size());
    
}

bool MarketModelEvolverLMMPC_CUDA::generatePathsIfNeeded() const
{
    if (currentPath_ != pathBatchSize_) // nothing to do 
        return false;
   
    if (scramble_)
        for (int i=0; i < static_cast<int>(scrambler_host_.size()); ++i)
                scrambler_vec_[i] = scrambler_host_[i] = static_cast<unsigned int>(rng_.nextInt32());
   
    
    if (useGPU_)
    {
         bool getLogs=false;

        float innerTime;
    
        totalTimeInCudaPart_+= LMMEvolutionMainRoutine(pathBatchSize_, 
                                                        pathOffset_, 
                                                       numberOfRates_, 
                                                       numberOfFactors_, 
                                                       stepsForEvolution_, 
                                                       powerOfTwoForVariates_,
                                                       scrambler_host_, 
                                                       pseudoRoots_host_,
                                                       fixedDrifts_host_, 
                                                       displacements_host_,
                                                       initial_rates_host_, 
                                                       initial_log_rates_host_, 
                                                       taus_host_, 
                                                       initial_drifts_host_, 
                                                       alive_, 
                                                       alive_host_, 
                                                       evolved_rates_host_, // for output
                                                       getLogs,
                                                       evolved_log_rates_host_,
                                                       innerTime
                                                        );
    }
    else
    {
        LMMEvolutionRoutineGold<float>( pathBatchSize_, 
                                                        pathOffset_, 
                                                        numberOfRates_, 
                                                        numberOfFactors_, 
                                                       stepsForEvolution_, 
                                                       powerOfTwoForVariates_,
                                                        scrambler_vec_, 
                                                       pseudoRoots_vec_,
                                                       fixedDrifts_vec_, 
                                                       displacements_vec_,
                                                       initial_rates_vec_, 
                                                       initial_log_rates_vec_, 
                                                       taus_vec_, 
                                               //        initial_drifts_vec_, 
                                              //         alive_, 
                                              //         alive_vec_,
                                                       evolved_rates_vec_,
                                                       evolved_log_rates_vec_
                                                        );

        std::copy(evolved_rates_vec_.begin(),evolved_rates_vec_.end(),evolved_rates_host_.begin());
        std::cout << "\ngold routine used\n";
    }

    currentPath_ =0;

    if (!scramble_)
        pathOffset_ += pathBatchSize_;

    return true;

}

void MarketModelEvolverLMMPC_CUDA::dumpCache(int numberPaths) const
{
   CubeFacade<float>  tmp(&evolved_rates_host_[0],stepsForEvolution_,numberOfRates_,pathBatchSize_);
   
   
   for (int i=0; i < std::min(pathBatchSize_,numberPaths); ++i)
    {
        std::cout << "\n\n" << i << "\n";
        for (int j=0; j< stepsForEvolution_; ++j)
        {
            for (Size k=0; k < numberOfRates_; ++k)
                std::cout << tmp(j,k,i) << ",";
            std::cout << "\n";
        }
    }

}

const std::vector<Size>& MarketModelEvolverLMMPC_CUDA::numeraires() const
{
    return numeraires_;
}

Real MarketModelEvolverLMMPC_CUDA::startNewPath()
{
    generatePathsIfNeeded();
    pathToUse_ = currentPath_;
    ++currentPath_;
    currentStep_ =0;



    return 1.0;
}

Real MarketModelEvolverLMMPC_CUDA::advanceStep()
{
  CubeFacade<float>  tmp(&evolved_rates_host_[0],stepsForEvolution_,numberOfRates_,pathBatchSize_);
  

    for (Size i=0; i < numberOfRates_; ++i)
        currentForwards_[i] = static_cast<Rate>(tmp(currentStep_,i,pathToUse_));
    const std::vector<Size>& alives = marketModel_->evolution().firstAliveRate();

    Size aliveIndex = alives[currentStep_];
 
    curveState_.setOnForwardRates(currentForwards_,aliveIndex);
    
    ++currentStep_;
    
    return 1.0;
}

Size MarketModelEvolverLMMPC_CUDA::currentStep() const
{
    return currentStep_;
}

const CurveState& MarketModelEvolverLMMPC_CUDA::currentState() const
{
    return curveState_;
}

void MarketModelEvolverLMMPC_CUDA::setInitialState(const CurveState&)
{
    throw("not yet done");
}

