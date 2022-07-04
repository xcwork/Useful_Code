//
//
//                  LMM_evolver_partial_gold.h
//
//
// (c) Mark Joshi 2014
// This code is released under the GNU public licence version 3

#ifndef LMM_EVOLVER_PARTIAL_GOLD_H
#define LMM_EVOLVER_PARTIAL_GOLD_H

#include <vector>
#include <gold/MatrixFacade.h>
#include <gold/math/cube_gold.h>
#include <gold/math/basic_matrix_gold.h>
#include <gold/LMM_evolver_gold.h>

/*
templatized to allow choice between doubles and floats
*/
template<class T>
class LMMEvolverPartial_gold
{
public:
    LMMEvolverPartial_gold(const Cube_gold<T>& pseudoRoots_vec,
        int numberSteps,
        int numberRates,
        int numberFactors,
        const std::vector<T>& displacements_vec,
        const std::vector<T>& taus_vec,
        const std::vector<int>& aliveIndex_vec,
        const std::vector<T>& initialRates_vec
        );

    void conditionallyGenerate(int stepNumber,  
        const std::vector<T>& currentRates_vec,
        const CubeConstFacade<T>& variates_cube,  //variates_cube(p,s-stepNumber,f);
        int paths,
        int offset, // subtracted from step from doing output
        CubeFacade<T>& paths_out_cube,
        double cutoffLevel) const;




private:

    Cube_gold<T> pseudoRoots_cube_;
    int numberSteps_;
    int numberRates_;
    int numberFactors_;
    std::vector<T> displacements_vec_;
    std::vector<T> taus_vec_;
    std::vector<int> aliveIndex_vec_;

    std::vector<T> initialRates_vec_;
    std::vector<T> initialLogRates_vec_;

    std::vector<T> initialDrifts_vec_;

    Matrix_gold<T> fixedDrifts_mat_;
    spotDriftComputer<T> driftComputer_;

    mutable std::vector<T> stepInitialRates_vec_;
    mutable std::vector<T> stepInitialLogRates_vec_;

    mutable std::vector<T> stepInitialDrifts_vec_;

    mutable std::vector<T> currentRates_vec_;
    mutable std::vector<T> currentLogRates_vec_;
    mutable std::vector<T> predictorDrifts_vec_;
    mutable std::vector<T> correctedDrifts_vec_;


};


template<class T>
LMMEvolverPartial_gold<T>::LMMEvolverPartial_gold(const Cube_gold<T>& pseudoRoots_cube,
                                                  int numberSteps,
                                                  int numberRates,
                                                  int numberFactors,
                                                  const std::vector<T>& displacements_vec,
                                                  const std::vector<T>& taus_vec,
                                                  const std::vector<int>& aliveIndex_vec,
                                                  const std::vector<T>& initialRates_vec
                                                  )
                                                  :
pseudoRoots_cube_(pseudoRoots_cube),
    numberSteps_(numberSteps),
    numberRates_(numberRates),
    numberFactors_(numberFactors),
    displacements_vec_(displacements_vec),
    taus_vec_(taus_vec),
    aliveIndex_vec_(aliveIndex_vec),
    initialRates_vec_(initialRates_vec),
    initialLogRates_vec_(numberRates),
    initialDrifts_vec_(numberRates),
    fixedDrifts_mat_(numberRates,numberSteps,0.0),
    driftComputer_(taus_vec,numberFactors,displacements_vec),
    stepInitialRates_vec_(numberRates),
    stepInitialLogRates_vec_(numberRates),
    stepInitialDrifts_vec_(numberRates),
    currentRates_vec_(numberRates),
    currentLogRates_vec_(numberRates),
    predictorDrifts_vec_(numberRates),
    correctedDrifts_vec_(numberRates)
{

    if (pseudoRoots_cube_.layers() != numberSteps_)
        GenerateError("pseudoRoots_cube_ and numberSteps_ mismatch in LMMEvolverPartial_gold");
    if (pseudoRoots_cube_.rows() != numberRates_)
        GenerateError("pseudoRoots_cube_ and numberRates_ mismatch in LMMEvolverPartial_gold");
    if (pseudoRoots_cube_.columns() != numberFactors_)
        GenerateError("pseudoRoots_cube_ and numberFactors_ mismatch in LMMEvolverPartial_gold");
    if (aliveIndex_vec_.size()!= numberSteps_)
        GenerateError("alive index and numberSteps_ mismatch in LMMEvolverPartial_gold");
    if (displacements_vec_.size()!=numberRates_)
        GenerateError("displacements_vec_ size mismatch in LMMEvolverPartial_gold");
    if (taus_vec_.size()!=numberRates_)
        GenerateError("taus_vec_ size mismatch in LMMEvolverPartial_gold");
    if (initialRates_vec_.size()!=numberRates_)
        GenerateError("initialRates_vec_ size mismatch in LMMEvolverPartial_gold");

    for (int s=0; s < numberSteps_; ++s)
        for (int r=0; r < numberRates_; ++r)
        {
            T x=0.0f;

            for (int f=0; f < numberFactors_; ++f)
            {
                T a = pseudoRoots_cube_(s,r,f);
                x+=a*a;
            }
            fixedDrifts_mat_(s,r)= -0.5*x;

        }

        driftComputer_.getDrifts(pseudoRoots_cube_[0],initialRates_vec_,initialDrifts_vec_);


        for (int r=0; r < numberRates_; ++r)
            initialLogRates_vec_[r] = initialRates_vec_[r];

}

template<class T>
void LMMEvolverPartial_gold<T>::conditionallyGenerate(int stepNumber,  // number of steps already done
                                                      const std::vector<T>& currentRates_vec,
                                                      const CubeConstFacade<T>& variates_cube,
                                                      int paths,
                                                      int offset,
                                                      CubeFacade<T>& paths_out_cube,
                                                      double cutoffLevel) const
{
    stepInitialRates_vec_ = currentRates_vec;
    int alive = aliveIndex_vec_[stepNumber];

    for (int r=alive; r < numberRates_; ++r)
        stepInitialLogRates_vec_[r] = log( stepInitialRates_vec_[r]+displacements_vec_[r]);

    driftComputer_.getDrifts(pseudoRoots_cube_[stepNumber],stepInitialRates_vec_,alive,stepInitialDrifts_vec_);

    for (int p=0; p <paths; ++p)
    {
        bool notFirstStep=false;

        currentRates_vec_=stepInitialRates_vec_; // this may be superfluous
        currentLogRates_vec_=stepInitialLogRates_vec_;


        for (int s = stepNumber; s< numberSteps_; ++s)
        {
            int aliveIndex = aliveIndex_vec_[s];

            if (notFirstStep)
            {
                driftComputer_.getDrifts(pseudoRoots_cube_[s],currentRates_vec_,aliveIndex,predictorDrifts_vec_);
            }
            else
            {
                predictorDrifts_vec_ = stepInitialDrifts_vec_;
                notFirstStep=true;
            }

            for (int r=aliveIndex; r< numberRates_; ++r)
            {
                T increment =0.0f;
                for (int f=0; f< numberFactors_; ++f)
                {
                    T z=variates_cube(p,s-stepNumber,f);
                    increment += pseudoRoots_cube_(s,r,f)*z;
                }
                currentLogRates_vec_[r]+=predictorDrifts_vec_[r]+fixedDrifts_mat_(s,r)+increment;
                if (currentLogRates_vec_[r] >cutoffLevel)
                    currentLogRates_vec_[r] =cutoffLevel;
                currentRates_vec_[r] =exp(currentLogRates_vec_[r])-displacements_vec_[r];
            }

            // Euler part of stepping is done, now do correction
            driftComputer_.getDrifts(pseudoRoots_cube_[s],currentRates_vec_,aliveIndex,correctedDrifts_vec_);

            for (int r=aliveIndex; r< numberRates_; ++r)
            {
                currentLogRates_vec_[r]+=0.5*(correctedDrifts_vec_[r]-predictorDrifts_vec_[r]);
                if (currentLogRates_vec_[r] >cutoffLevel)
                    currentLogRates_vec_[r] =cutoffLevel;

                currentRates_vec_[r] =exp(currentLogRates_vec_[r])-displacements_vec_[r];
            }
            for (int r=0; r < numberRates_; ++r)
                paths_out_cube(s+offset,r,p)= currentRates_vec_[r]; 

        }// end step loop
    }// end path loop
}// end method


#endif
