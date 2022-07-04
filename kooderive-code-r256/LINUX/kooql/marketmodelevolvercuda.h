//
//
//                                                                  MarketModelEvolverCuda.h
//
// (c) Mark Joshi 2010


#ifndef MARKET_MODEL_EVOLVER_CUDA_H
#define MARKET_MODEL_EVOLVER_CUDA_H

#include <ql/models/marketmodels/marketmodel.hpp>
#include <thrust/host_vector.h>
#include <ql\models\marketmodels/evolver.hpp>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>
#include <gold/MatrixFacade.h>

using namespace QuantLib;

class MarketModelEvolverLMMPC_CUDA : public MarketModelEvolver
{

public:
    MarketModelEvolverLMMPC_CUDA(const boost::shared_ptr<MarketModel>&,
        int MersenneSeed, // seed is for scrambling only
        int pathBatchSize, // how many paths to generate at a time, this is constrained by the memory of your GPU
        int pathOffset,
        bool scramble, // switch scrambling off
        bool useGPU);

    const std::vector<Size>& numeraires() const;
    Real startNewPath();
    Real advanceStep();
    Size currentStep() const;
    const CurveState& currentState() const;
    void setInitialState(const CurveState&);

    bool generatePathsIfNeeded() const;

    void dumpCache(int numberPaths =10) const;

    double totalTimeInCudaPart()
    {
        return totalTimeInCudaPart_;
    }


// accessors, useful for code reuse 
    const thrust::host_vector<float>& GetpseudoRoots_host() const
    {    return  pseudoRoots_host_; }
    
    const thrust::host_vector<float>& GetfixedDrifts_host() const
    {    return  fixedDrifts_host_;     }

    const thrust::host_vector<float>& Getdisplacements_host() const
    {   return  displacements_host_; }
    
    const thrust::host_vector<float>& Getinitial_rates_host() const
    {   return  initial_rates_host_;  }
    
    const thrust::host_vector<float>& Getinitial_log_rates_host() const
    {   return  initial_log_rates_host_; }

    const thrust::host_vector<float>& Gettaus_host() const
    {   return  taus_host_;   }

    const thrust::host_vector<float>& Getinitial_drifts_host() const
    {  return  initial_drifts_host_;  }

    const thrust::host_vector<int>& Getalive_host() const 
    {        return alive_host_;   }

    MersenneTwisterUniformRng GetScramblingRNG() const
    {
        return rng_;
    }   

    const std::vector<int>& Getalive() const
    {
        return alive_;
    }
    
    int GetBatchSize() const
    { return pathBatchSize_;
    }

    int GetpathOffset() const
    {
        return pathOffset_;
    }
        
    bool Getscramble() const
    {   
        return scramble_;
    }

    int GetpowerOfTwoForVariates() const
    {
            return   powerOfTwoForVariates_;
    }

    boost::shared_ptr<MarketModel> GetmarketModel() const
    {
            return marketModel_;
    }

private:
    //     void setForwards(const std::vector<Real>& forwards);
    // inputs
    boost::shared_ptr<MarketModel> marketModel_;
    int pathBatchSize_;
    mutable int pathOffset_;

    mutable double totalTimeInCudaPart_;

    // fixed variables
    std::vector<Size> numeraires_;
    Size initialStep_;

    std::vector<std::vector<Real> > fixedDrifts_;
    bool scramble_;

    // evolving variables

    mutable MersenneTwisterUniformRng rng_;
    mutable int currentPath_;

    // working variables
    Size numberOfRates_, numberOfFactors_;
    LMMCurveState curveState_;
    Size currentStep_;
    int pathToUse_;
    std::vector<Rate> forwards_, displacements_,  initialLogForwards_;
    std::vector<Real> initialDrifts_;
    std::vector<int> alive_;
    std::vector<Rate> currentForwards_;

    // data passing to gpu

    // fixed data 
    int stepsForEvolution_; 
    int powerOfTwoForVariates_;

    thrust::host_vector<float> pseudoRoots_host_;
    thrust::host_vector<float> fixedDrifts_host_;
    thrust::host_vector<float> displacements_host_;
    thrust::host_vector<float> initial_rates_host_;
    thrust::host_vector<float> initial_log_rates_host_; 
    thrust::host_vector<float> taus_host_;
    thrust::host_vector<float> initial_drifts_host_; 
    thrust::host_vector<int> alive_host_;

    // changing gpu data

    mutable thrust::host_vector<unsigned int> scrambler_host_;

    // data passed from gpu
    mutable thrust::host_vector<float> evolved_rates_host_; // for output
    mutable thrust::host_vector<float> evolved_log_rates_host_;  // for output 

    //       CubeFacade<float> evolved_rates_cube_;

    // for cpu version
    bool useGPU_;

    std::vector<float> pseudoRoots_vec_;
    std::vector<float> fixedDrifts_vec_;
    std::vector<float> displacements_vec_;
    std::vector<float> initial_rates_vec_;
    std::vector<float> initial_log_rates_vec_; 
    std::vector<float> taus_vec_;
    std::vector<float> initial_drifts_vec_;
    std::vector<int> alive_vec_;
    mutable std::vector<float> evolved_rates_vec_;
    mutable std::vector<float> evolved_log_rates_vec_;
    mutable std::vector<unsigned int> scrambler_vec_;


};


#endif
