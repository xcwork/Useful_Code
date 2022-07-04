
#include <gold/Max_estimation_MC.h>
#include <algorithm>
#include <numeric>
MaxEstimatorForAB::MaxEstimatorForAB(unsigned long seed, int numberpoints) :rng_(seed), numberPoints_(numberpoints)
{
}

void MaxEstimatorForAB::GetEstimate(int paths, 
                                    const std::vector<double>& means,
                                    const std::vector<double>& sds, 
                                    const std::vector<bool>& exerciseOccurred,
                                    double& biasMean,
                                    double& biasSe)
{
    if (means.size() != numberPoints_)
        GenerateError("means misized in MaxEstimatorForAB");

    if (sds.size() != numberPoints_)
        GenerateError("means misized in MaxEstimatorForAB");
 
    if (exerciseOccurred.size() != numberPoints_)
        GenerateError("means misized in MaxEstimatorForAB");

    double maxM = *std::max_element(means.begin(),means.end());

    double sum=0.0;
    double sumsq=0.0;
    inverseCumulativeGold<double> inverter;
    for (int p=0; p < paths; ++p)
    {
        double g = inverter(rng_.next())*sds[0];
        double m= means[0] + (exerciseOccurred[0] ? 0.0 : g);
    
        double numerairesErr = exerciseOccurred[0] ? g : 0.0;

        for (int i=1; i < numberPoints_; ++i)
        {
            double thisErr = inverter(rng_.next())*sds[i];

            double thisP = means[i] + (exerciseOccurred[i] ? 0.0 : thisErr) - numerairesErr;
            numerairesErr += exerciseOccurred[i] ? thisErr : 0.0;

            m = std::max(thisP,m);
        }
        
        m-=maxM;

        sum +=m;
        sumsq +=m*m;

    }
    biasMean = sum/paths;

    double biasEx2 = sumsq/paths;
    double variance = biasEx2- biasMean*biasMean;
    biasSe = sqrt(variance/paths);

 
}
