
//
//
//                                                                  MarketModelEvolverCuda.h
//
// (c) Mark Joshi 2010


#ifndef BROWNIAN_GENERATOR_SOBOL_CUDA_H
#define BROWNIAN_GENERATOR_SOBOL_CUDA_H

#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/models/marketmodels/browniangenerator.hpp>
using namespace QuantLib;


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



    class SobolCudaBrownianGenerator : public BrownianGenerator
    {
      public:
      
        SobolCudaBrownianGenerator(
                           Size factors,
                           Size steps,
                           int pathBatchSize,
                           int pathOffset,
                           unsigned long scramblingSeed,
                           bool doScrambling);

        Real nextPath();
        Real nextStep(std::vector<Real>&);

        Size numberOfFactors() const;
        Size numberOfSteps() const;

        void GeneratePathsIfNeeded() const;
      private:
        Size factors_, steps_;
        int nextPowerOfTwo_;
        int powerOfTwoForVariates_;
        int pathBatchSize_;
        mutable int pathOffset_;
        Size currentStep_;


        mutable thrust::host_vector<float> bridgeVariates_host_;
        mutable thrust::host_vector<int> scramblers_; 
        MersenneTwisterUniformRng rng_;
        bool doScrambling_;
        mutable int currentPath_;

    };

    class SobolCudaBrownianGeneratorFactory : public BrownianGeneratorFactory {
      public:
        SobolCudaBrownianGeneratorFactory(int pathBatchSize,
                           int pathOffset,
                           unsigned long scramblingSeed,
                           bool doScrambling);
        boost::shared_ptr<BrownianGenerator> create(Size factors,
                                                    Size steps) const;
      private:
          int pathBatchSize_;
          int pathOffset_;
          unsigned long scramblingSeed_;
          bool doScrambling_;
     
    };




#endif
