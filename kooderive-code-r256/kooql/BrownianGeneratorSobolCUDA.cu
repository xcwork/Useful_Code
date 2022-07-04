#include "BrownianGeneratorSobolCUDA.h"
#include <Brownian_generator_full.h>
#include <gold/MatrixFacade.h> 

SobolCudaBrownianGenerator::SobolCudaBrownianGenerator(
                           Size factors,
                           Size steps,
                           int pathBatchSize,
                           int pathOffset,
                           unsigned long scramblingSeed,
                           bool doScrambling)
                           :
                        factors_(factors),
                        steps_(steps),
                        pathBatchSize_(pathBatchSize),
                        pathOffset_(pathOffset),
                        scramblers_(steps*factors),
                        rng_(scramblingSeed),
                        doScrambling_(doScrambling) ,
                        currentPath_(pathBatchSize)
{
    nextPowerOfTwo_ =1;
    powerOfTwoForVariates_=0;

    while (nextPowerOfTwo_< static_cast<int>(steps))
    {
        ++powerOfTwoForVariates_;
        nextPowerOfTwo_*=2;
    }

    scramblers_.resize(nextPowerOfTwo_*factors_);
    bridgeVariates_host_.resize(nextPowerOfTwo_*pathBatchSize_*factors_);

}

void SobolCudaBrownianGenerator::GeneratePathsIfNeeded() const
{
       if (currentPath_ != pathBatchSize_)
            return;

       if (doScrambling_)
          for (int i=0; i < static_cast<int>(scramblers_.size()); ++i)
                 scramblers_[i] = static_cast<unsigned int>(rng_.nextInt32());

       float innerTime=-1.0;

       BrownianGenerationMainRoutine(pathBatchSize_, 
                                                       pathOffset_, 
                                                       factors_, 
                                                       steps_, 
                                                       powerOfTwoForVariates_,
                                                       scramblers_, 
                                                       bridgeVariates_host_,
                                                       innerTime
                                                        );

       if (!doScrambling_)
           pathOffset_ += pathBatchSize_;


       currentPath_=-1;
        

}

Real SobolCudaBrownianGenerator::nextPath()
{
        GeneratePathsIfNeeded();
        ++currentPath_;

        
        currentStep_ =0;

        return 1.0;
}

Real SobolCudaBrownianGenerator::nextStep(std::vector<Real>& variates)
{
        CubeConstFacade<float> variatesFacade(&bridgeVariates_host_[0],steps_,pathBatchSize_,factors_);

        for (Size i=0; i < factors_; ++i)
            variates[i] = variatesFacade(currentStep_,currentPath_,i);

        /*
#ifdef _DEBUG
          
        std::cout << currentPath_ << "," << currentStep_  <<",";
        
        for (Size i=0; i < factors_; ++i)    
            std::cout << variates[i] << ",";
        
        std::cout << "\n";

#endif
*/
        ++currentStep_;

        return 1.0;
}

Size  SobolCudaBrownianGenerator::numberOfFactors() const
{
    return factors_;
}

Size SobolCudaBrownianGenerator::numberOfSteps() const
{
    return steps_;
}   


SobolCudaBrownianGeneratorFactory::SobolCudaBrownianGeneratorFactory(int pathBatchSize,
                           int pathOffset,
                           unsigned long scramblingSeed,
                           bool doScrambling)
                           :pathBatchSize_(pathBatchSize),
                           pathOffset_(pathOffset),
                           scramblingSeed_(scramblingSeed),
                           doScrambling_(doScrambling) 
{
}
        
boost::shared_ptr<BrownianGenerator> SobolCudaBrownianGeneratorFactory::create(Size factors,
                                                    Size steps) const
{
    return boost::shared_ptr<BrownianGenerator>(new SobolCudaBrownianGenerator(
                            factors,
                            steps,
                            pathBatchSize_,
                            pathOffset_,
                            scramblingSeed_,
                            doScrambling_));
}
