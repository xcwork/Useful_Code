//
//
//                          Andersen_Broadie_gold.h
//
//

#include <gold/Andersen_Broadie_gold.h>
#include <algorithm>
#include <gold/BSFormulas_gold.h>
#include <gold/math/Normals_gold.h>
#include <gold/math/basic_matrix_gold.h>
#include <gold/Max_estimation_MC.h>
namespace
{
    double squareof(double x)
    {
        return x*x;
    }
}

void findTwoLargest(const std::vector<double>& v, std::vector<std::pair<double,int> >& wsp, int&i, int &j)
{
    wsp.resize(v.size());
    {
        for (int i=0; i < v.size(); ++i)
        {
            wsp[i].first = v[i];
            wsp[i].second =i;
        }
    }
    std::sort(wsp.begin(),wsp.end());
    int first = wsp.rbegin()->second;
    int second = (wsp.rbegin()+1)->second;
    i = std::min(first,second);
    j = std::max(first,second);
}

void findThreeLargest(const std::vector<double>& v, std::vector<std::pair<double,int> >& wsp, int&i, int &j, int& k)
{
    wsp.resize(v.size());
    {
        for (int i=0; i < v.size(); ++i)
        {
            wsp[i].first = v[i];
            wsp[i].second =i;
        }
    }
    std::sort(wsp.begin(),wsp.end());
    int first = wsp.rbegin()->second;
    int second = (wsp.rbegin()+1)->second;
    int third = (wsp.rbegin()+2)->second;

    i =std::min( std::min(first,second),third);
    k = std::max(std::max(first,second),third);
    j = first+second+third-i-k; //i.e. the one that's not the other two
}

void ABDualityGapEstimateZeroRebate(const MatrixConstFacade<int>& exerciseIndicators_mat,
                                    const MatrixConstFacade<double>& means_i_mat,
                                    const MatrixConstFacade<double>& sds_i_mat,
                                    double& mean,
                                    double& se,

                                    const std::vector<bool>& isExeriseDate_vec)
{
    int steps = means_i_mat.columns();
    int paths = means_i_mat.rows();

    if (exerciseIndicators_mat.rows()!= steps)
        GenerateError("incompatible data in ABDualityGapEstimateZeroRebate");

    if (exerciseIndicators_mat.columns()!= paths)
        GenerateError("incompatible data in ABDualityGapEstimateZeroRebate");
    std::vector<double> pathwiseVals_vec(paths);

    double total=0.0;
    double totalSq=0.0;
    for (int p=0; p < paths; ++p)
    {
        double pathwiseMax =0.0;
        double numeraireBalance=0.0;

        for (int s=0;s < steps; ++s)
        {
            if (isExeriseDate_vec[s])
            {
                double x;
                if (exerciseIndicators_mat(s,p) == 1) // it was an exercise point
                {
                    x = -   numeraireBalance;
                    double continuationVal = means_i_mat(p,s);
                    numeraireBalance -= continuationVal;
                }
                else
                {
                    x =  -means_i_mat(p,s) -numeraireBalance;
                }

                pathwiseMax = std::max(pathwiseMax,x);
            }
        }

        // now we are at the very end, there are no cash-flows left
        //so all we have is the numeraire balance
        // if we exercised well, this is positive and so has no effect
        pathwiseMax = std::max(pathwiseMax,-numeraireBalance);

        pathwiseVals_vec[p]=pathwiseMax;
        total+=pathwiseMax;
        totalSq+=pathwiseMax*pathwiseMax;

    }

    mean = total/paths;
    double var = totalSq/paths - mean*mean;
    se = sqrt(var/paths);
    debugDumpVector(pathwiseVals_vec,"pathwiseMax");
}


void ABDualityGapEstimateZeroRebateBiasEstimation(const MatrixConstFacade<int>& exerciseIndicators_mat,
                                                  const MatrixConstFacade<double>& means_i_mat,
                                                  const MatrixConstFacade<double>& sds_i_mat,
                                                  double& mean,
                                                  double& se,
                                                  double& mean2,
                                                  double& se2,
                                                  double& mean3,
                                                  double& se3,
                                                  double& mean4,
                                                  double& se4,
                                                  int pathsForGaussianEstimation,
                                                  int seed,
                                                  const std::vector<bool>& isExeriseDate_vec,
                                                  int numberExerciseDates)
{
    int steps = means_i_mat.columns();
    int paths = means_i_mat.rows();

    if (exerciseIndicators_mat.rows()!= steps)
        GenerateError("incompatible data in ABDualityGapEstimateZeroRebate");

    if (exerciseIndicators_mat.columns()!= paths)
        GenerateError("incompatible data in ABDualityGapEstimateZeroRebate");
    std::vector<double> pathwiseVals_vec(paths);
    std::vector<double> pathwiseVals2_vec(paths);
    std::vector<double> pathwiseVals3_vec(paths);
    std::vector<double> pathwiseVals4_vec(paths);

    std::vector<double> corrections2_vec(paths);
    std::vector<double> corrections3_vec(paths);
    std::vector<double> corrections4_vec(paths);



    std::vector<double> pathData(numberExerciseDates+1);
    std::vector<double> numeraireBalances(numberExerciseDates+1);
    std::vector<double> thisPathSds(numberExerciseDates+1,0.0);
    std::vector<bool> thisPathExercises(numberExerciseDates+1,true);
    Matrix_gold<double> weights_matrix(3,3,0.0);
    weights_matrix(0,1) = -1.0;
    weights_matrix(0,2) = -1.0;

    MaxEstimatorForAB ABestimator(seed,numberExerciseDates+1);


    std::vector<std::pair<double,int> > wsp,wsp2;

    double total=0.0,total2=0.0,total3=0.0,total4=0.0;
    double totalSq=0.0,totalSq2=0.0,totalSq3=0.0,totalSq4=0.0;
    for (int p=0; p < paths; ++p)
    {
        double pathwiseMax =0.0;
        double pathwiseMax2;
        double pathwiseMax3;
        double pathwiseMax4=0.0;

        double numeraireBalance=0.0;

        int exerciseNumber = 0;
        for (int s=0;s < steps; ++s)
        {
            if (isExeriseDate_vec[s])
            {
                double x;
                thisPathSds[exerciseNumber] = sds_i_mat(p,s);
                thisPathExercises[exerciseNumber] = (exerciseIndicators_mat(s,p) == 1);

                if (  thisPathExercises[exerciseNumber]) // it was an exercise point
                {
                    x = -   numeraireBalance;
                    double continuationVal = means_i_mat(p,s);
                    numeraireBalance -= continuationVal;
                }
                else
                {
                    x =  -means_i_mat(p,s) -numeraireBalance;
                }

                pathwiseMax = std::max(pathwiseMax,x);


                pathData[exerciseNumber] =x;
                numeraireBalances[exerciseNumber] = numeraireBalance;

                ++exerciseNumber;
            }
        }

        // now we are at the very end, there are no cash-flows left
        //so all we have is the numeraire balance
        // if we exercised well, this is positive and so has no effect
        pathwiseMax = std::max(pathwiseMax,-numeraireBalance);
        *pathData.rbegin() = -numeraireBalance;
        *numeraireBalances.rbegin() = numeraireBalance;

        debugDumpVector(thisPathExercises,"thisPathExercises");
        debugDumpVector(pathData,"pathData");
        debugDumpVector(thisPathSds,"thisPathSds");


        pathwiseVals_vec[p]=pathwiseMax;

        //     if (pathwiseMax >0)

        // two asset correction
        {
            int firstIndex, secondIndex;

            findTwoLargest(pathData,wsp,firstIndex,secondIndex);

            double sqErrorInFirst=0.0;
            double sqNumeraireError =0.0;
            double sqErrorInSecond =0.0;


            if (!  thisPathExercises[firstIndex])
                sqErrorInFirst = squareof(thisPathSds[firstIndex]); // we didn't exercise so there is noise in first value

            if (!  thisPathExercises[secondIndex])
                sqErrorInSecond = squareof(thisPathSds[secondIndex]);// we didn't exercise so there is noise in second value

            for (int ind = firstIndex; ind < secondIndex; ++ind) // note we include first index in the numeraire calc but not second
            {
                if ( thisPathExercises[ind])
                    sqNumeraireError+=  squareof(thisPathSds[ind]);// we didn't exercise so there is noise in second value

            }

            double sqError=sqErrorInFirst+sqNumeraireError+sqErrorInSecond;

            double sigma = sqrt(sqError);

            double strike = fabs(pathData[firstIndex] - pathData[secondIndex]);

            Realv correction =sigma > 0.0 ? NormalBlackFromSd(0.0, //Realv f0, 
                strike, 
                sigma,//sd
                1.0 //Realv Annuity
                ) : 0.0;

            pathwiseMax2 =pathwiseMax- correction;
            pathwiseVals2_vec[p] = pathwiseMax2;
            corrections2_vec[p] = correction;
        }
        // three asset correction
        {
            int firstIndex, secondIndex,thirdIndex;

            findThreeLargest(pathData,wsp2,firstIndex,secondIndex,thirdIndex);
            bool isFirstEx = thisPathExercises[firstIndex];
            bool isSecondEx = thisPathExercises[secondIndex];
            bool isThirdEx =  thisPathExercises[thirdIndex];

            Realv mux = pathData[firstIndex];
            Realv muy = pathData[secondIndex];
            Realv muz = pathData[thirdIndex];

            Realv vx = squareof(thisPathSds[firstIndex]);
            Realv vy = squareof(thisPathSds[secondIndex]);
            Realv vz = squareof(thisPathSds[thirdIndex]);

            Realv corr=0.0;


            if (isSecondEx)
            {
                weights_matrix(1,1) = 0.0;
                weights_matrix(1,2) =-1.0;
            }
            else
            {
                weights_matrix(1,1) = 1.0;
                weights_matrix(1,2) = 0.0;
            }

            if (isThirdEx)
            {
                weights_matrix(2,2) =0.0;
            }
            else
                weights_matrix(2,2) =1.0;


            Realv vbetween1and2 =0.0;
            for (int preEx=firstIndex+1; preEx < secondIndex; ++preEx)
                if (thisPathExercises[preEx])
                    vbetween1and2 += squareof(thisPathSds[preEx]);

            Realv vbetween2and3 =0.0;
            for (int betweenEx=secondIndex+1; betweenEx < thirdIndex; ++betweenEx)
                if (thisPathExercises[betweenEx])
                    vbetween2and3 += squareof(thisPathSds[betweenEx]);


            Realv mu1,mu2,c;
            Realv V1,V2,cov;

            c = mux;
            mu1 = muy;
            mu2 = muz;
            V1 = vx+squareof( weights_matrix(1,1))*vy+vbetween1and2;
            V2 = vx+squareof( weights_matrix(1,2))*vy+squareof( weights_matrix(2,2))*vz+vbetween1and2+vbetween2and3;
            cov = vx+vbetween1and2;

            Realv maxfor3 = std::max(mux,std::max(muy,muz));

            Realv correction3 =  ExpectedMaximumOfTwoGausssiansAndAConstant(mu1,  mu2, c, V1, cov,V2)-maxfor3;
            pathwiseMax3 = pathwiseMax - correction3;

            pathwiseVals3_vec[p] = pathwiseMax3;

            corrections3_vec[p] = correction3;


        }
        if (pathsForGaussianEstimation>0)
        {
            double thisBias,thisSe;
            ABestimator.GetEstimate(pathsForGaussianEstimation,pathData,thisPathSds,thisPathExercises,thisBias,thisSe);

            pathwiseMax4 = pathwiseMax - thisBias;

            pathwiseVals4_vec[p] = pathwiseMax4;

            corrections4_vec[p] = thisBias;

        }

        total+=pathwiseMax;
        totalSq+=pathwiseMax*pathwiseMax;

        total2+=pathwiseMax2;
        totalSq2+=pathwiseMax2*pathwiseMax2;
      
        total3+=pathwiseMax3;
        totalSq3+=pathwiseMax3*pathwiseMax3;
  
        total4+=pathwiseMax4;
        totalSq4+=pathwiseMax4*pathwiseMax4;



    } 

    mean = total/paths;
    double var = totalSq/paths - mean*mean;
    se = sqrt(var/paths);
    
    mean2 = total2/paths;
    double var2 = totalSq2/paths - mean2*mean2;
    se2 = sqrt(var2/paths);

    mean3 = total3/paths;
    double var3 = totalSq3/paths - mean3*mean3;
    se3 = sqrt(var3/paths);

    mean4 = total4/paths;
    double var4 = totalSq4/paths - mean4*mean4;
    se4 = sqrt(var4/paths);


    debugDumpVector(pathwiseVals_vec,"pathwiseMax");
    debugDumpVector(pathwiseVals2_vec,"pathwiseMax2");
    debugDumpVector(corrections2_vec,"corrections2_vec");

    debugDumpVector(pathwiseVals3_vec,"pathwiseMax3");
    debugDumpVector(corrections3_vec,"corrections3_vec");

    debugDumpVector(pathwiseVals4_vec,"pathwiseMax4");
    debugDumpVector(corrections4_vec,"corrections4_vec");
}

