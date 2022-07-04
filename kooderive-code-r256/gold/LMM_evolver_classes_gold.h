//
//
//
//                                           LMM_evolver_classes_gold.h
//
//

#ifndef LMM_EVOLVER_CLASSES_GOLD_H

#define LMM_EVOLVER_CLASSES_GOLD_H

#include <gold/LMM_evolver_gold.h>
#include <gold/LMM_evolver_full_gold.h>


// template implemementations are at the end of the file


// if you want to use a PC evolver several times without having to reset up workspace etc
template<class D>
class LMM_evolver_pc_class_gold  
{
public:
    LMM_evolver_pc_class_gold(const std::vector<D>& initial_Rates_vec,
        const std::vector<D>& taus_vec,
        const std::vector<D>& pseudo_roots_vec,
        const std::vector<D>& displacements_vec, 
        int factors,
        int steps, 
        int rates,
        D cutoffLevel);


    void generateRates(const std::vector<D>& correlatedBrownianIncrements_vec,
        int paths,
        std::vector<D>& evolved_rates_vec,
        std::vector<D>& evolved_log_rates_vec) const;

    void generateRatesFromUncorrelated(const std::vector<D>& brownianIncrements_vec,
        int paths,
        std::vector<D>& evolved_rates_vec,
        std::vector<D>& evolved_log_rates_vec) const;


private:

    std::vector<D> initial_Rates_vec_;
    std::vector<D> initial_log_Rates_vec_;

    std::vector<D> taus_vec_;
    std::vector<D> pseudo_roots_vec_;
    std::vector<D> displacements_vec_;
    int factors_;
    int steps_;
    int rates_;
    D cutoffLevel_;
    spotDriftComputer<D> driftComputer_;
    mutable std::vector<D> these_Drifts_vec_; //(rates);
    mutable std::vector<D> these_Drifts_predicted_vec_; //;(rates);
    mutable std::vector<D> current_rates_vec_; //(rates);
   mutable  std::vector<D> current_log_rates_vec_; //;(rates);

    mutable std::vector<D> correlated_brownian_vec_;
    std::vector<D> fixedDrifts_vec_;

};

template<class D>
class LMMEvolutionFullPCSobol_gold
{
public:
    LMMEvolutionFullPCSobol_gold(int indicativePaths, 
        int rates, 
        int factors, 
        int stepsForEvolution, 
        int powerOfTwoForVariates,
        const std::vector<D>& pseudoRoots_vec,
        const std::vector<D>& displacements_vec,
        const std::vector<D>& initial_rates_vec, 
        const std::vector<D>& taus_vec,
        D cutOffLevel
        );

    D getPaths(int paths,int pathOffset, const std::vector<unsigned int>& scrambler_vec,  
        std::vector<D>& evolved_rates_vec,
        std::vector<D>& evolved_log_rates_vec);



private:
    int lastPaths_;
    int rates_;
    int factors_;
    int stepsForEvolution_; 
    int powerOfTwoForVariates_;

    std::vector<D> pseudoRoots_vec_;
    std::vector<D> fixedDrifts_vec_; 
    std::vector<D> displacements_vec_;
    std::vector<D> initial_rates_vec_;
    std::vector<D> initial_log_rates_vec_; 
    std::vector<D> taus_vec_;
    D cutOffLevel_;

    // stuff to set up
    int tot_dimensions_;
    int NintVariates_;
    std::vector<unsigned int> SobolInts_buffer_vec_;//(NintVariates); 
    std::vector<unsigned int> SobolInts_buffer_scrambled_vec_;//(NintVariates); 

    std::vector<D>  quasiRandoms_buffer_vec_;//(NintVariates); 

    std::vector<D> bridgeVariates_vec_;//(NintVariates); 
    std::vector<D> correlatedVariates_vec_;//(paths*rates*stepsForEvolution);

    std::vector<unsigned int> directions_vec_;//(tot_dimensions*n_directions*factors);

 //   BrownianBridgeMultiDim<D>::ordering allocator_; //( BrownianBridgeMultiDim::triangular);

    BrownianBridgeMultiDim<D> BB_;//(powerOfTwoForVariates, factors, allocator);
    std::vector<D> variates_; //(tot_dimensions*factors);
    std::vector<D> bridgedVariates_; //(tot_dimensions*factors);


    std::vector<D> initial_log_Rates_vec_; //(initial_Rates_vec);

    spotDriftComputer<D> driftComputer_;// (taus_vec,factors,displacements_vec);

    std::vector<D> these_Drifts_vec_; //(rates);
    std::vector<D> these_Drifts_predicted_vec_; //(rates);
    std::vector<D> current_rates_vec_; // (rates);
    std::vector<D> current_log_rates_vec_; //(rates);
 std::vector<D> pathVariates_vec_; //(rates);
  std::vector<D> bridgedPathVariates_vec_; //(rates);
};

// template implemementations

template<class D>
LMMEvolutionFullPCSobol_gold<D>::LMMEvolutionFullPCSobol_gold(int indicativePaths, 
                                                              int rates, 
                                                              int factors, 
                                                              int stepsForEvolution, 
                                                              int powerOfTwoForVariates,
                                                              const std::vector<D>& pseudoRoots_vec,
                                                              const std::vector<D>& displacements_vec,
                                                              const std::vector<D>& initial_rates_vec, 
                                                              const std::vector<D>& taus_vec,
                                                              D cutOffLevel
                                                              )
                                                              :lastPaths_(indicativePaths),
                                                              rates_(rates), 
                                                              factors_(factors), 
                                                              stepsForEvolution_(stepsForEvolution), 
                                                              powerOfTwoForVariates_(powerOfTwoForVariates),
                                                              pseudoRoots_vec_(pseudoRoots_vec),
                                                              fixedDrifts_vec_(rates*stepsForEvolution), 
                                                              displacements_vec_(displacements_vec),
                                                              initial_rates_vec_(initial_rates_vec),
                                                              initial_log_rates_vec_(initial_rates_vec), 
                                                              taus_vec_(taus_vec),
                                                              cutOffLevel_(cutOffLevel),
                                                              tot_dimensions_(intPower(2,powerOfTwoForVariates)),
                                                              NintVariates_(tot_dimensions_*indicativePaths*factors),
                                                              SobolInts_buffer_vec_(NintVariates_), 
                                                              SobolInts_buffer_scrambled_vec_(NintVariates_), 
                                                              quasiRandoms_buffer_vec_(NintVariates_),
                                                              bridgeVariates_vec_(NintVariates_),
                                                              correlatedVariates_vec_(indicativePaths*rates*stepsForEvolution),
                                                              directions_vec_(tot_dimensions_*n_directions*factors),
                                                      //        allocator_( BrownianBridgeMultiDim::triangular),
                                                              BB_(powerOfTwoForVariates_, factors, BrownianBridgeMultiDim<D>::triangular),
                                                              variates_(tot_dimensions_*factors),
                                                              bridgedVariates_(tot_dimensions_*factors),
                                                              initial_log_Rates_vec_(rates),
                                                              driftComputer_(taus_vec,factors,displacements_vec),
                                                              these_Drifts_vec_(rates),
                                                              these_Drifts_predicted_vec_(rates),
                                                              current_rates_vec_(rates),
                                                              current_log_rates_vec_(rates),
                                                              pathVariates_vec_(tot_dimensions_*factors),
                                                              bridgedPathVariates_vec_(tot_dimensions_*factors)

{

    initSobolDirectionVectors(tot_dimensions_*factors_, &directions_vec_[0]);

    for (int i=0; i < rates; ++i)
        initial_log_Rates_vec_[i] = log(initial_rates_vec_[i]+displacements_vec_[i]);

    CubeConstFacade<D> pseudo_roots_cube(&pseudoRoots_vec_[0],stepsForEvolution_,rates,factors);
    MatrixFacade<D> fixedDrifts_mat(&fixedDrifts_vec_[0],stepsForEvolution_,rates);
    for (int s=0; s < stepsForEvolution_; ++s)
    {
        for (int r=0; r< rates_; ++r)
        {
            D v=0.0;
            for (int f=0; f < factors;++f)
            {
                D x= pseudo_roots_cube(s,r,f);
                v+= x*x;
            }
            fixedDrifts_mat(s,r) = -0.5*v;
        }
    }


}
template<class D>
D LMMEvolutionFullPCSobol_gold<D>::getPaths(int paths, 
                                            int pathOffset,
                                            const std::vector<unsigned int>& scrambler_vec,  
                                            std::vector<D>& evolved_rates_vec,
                                            std::vector<D>& evolved_log_rates_vec) 
{
    int t0=clock();

    if (paths!=lastPaths_)
    {
        int NintVariates = tot_dimensions_*paths*factors_;



        SobolInts_buffer_vec_.resize(NintVariates); 
        SobolInts_buffer_scrambled_vec_.resize(NintVariates); 

        quasiRandoms_buffer_vec_.resize(NintVariates); 

        bridgeVariates_vec_.resize(NintVariates); 
        correlatedVariates_vec_.resize(paths*rates_*stepsForEvolution_);
        lastPaths_=paths;
    }



    LMMEvolutionRoutineUsingWorkspacesGold<D>( paths, 
        pathOffset, 
        rates_, 
        factors_, 
        stepsForEvolution_, 
        powerOfTwoForVariates_,
        scrambler_vec, 
        pseudoRoots_vec_,
        fixedDrifts_vec_, 
        displacements_vec_,
        initial_rates_vec_, 
        initial_log_rates_vec_, 
        taus_vec_, 
        //    const std::vector<D>& initial_drifts_vec, 
        //   const std::vector<int>& aliveIndices, 
        //     const  std::vector<int>& alive_vec,
        cutOffLevel_,
        evolved_rates_vec,
        evolved_log_rates_vec,
        // workspace
        SobolInts_buffer_vec_,
        SobolInts_buffer_scrambled_vec_,
        quasiRandoms_buffer_vec_,
        bridgeVariates_vec_,
        correlatedVariates_vec_,
        directions_vec_,
        BB_,
        tot_dimensions_,
        initial_log_Rates_vec_ // must equal logs of initial_Rates_vec
        ,    these_Drifts_vec_   //(rates);
        ,  these_Drifts_predicted_vec_ //(rates);
        ,   current_rates_vec_//(rates);
        ,   current_log_rates_vec_//(rates);
        ,    driftComputer_
        , pathVariates_vec_
        , bridgedPathVariates_vec_
        );






    int t1=clock();

    return (t1-t0)/static_cast<D>(CLOCKS_PER_SEC);


}










template<class D>
LMM_evolver_pc_class_gold<D>::LMM_evolver_pc_class_gold(const std::vector<D>& initial_Rates_vec,
                                                        const std::vector<D>& taus_vec,
                                                        const std::vector<D>& pseudo_roots_vec,
                                                        const std::vector<D>& displacements_vec, 
                                                        int factors,
                                                        int steps, 
                                                        int rates,
                                                        D cutoffLevel) :initial_Rates_vec_(initial_Rates_vec), initial_log_Rates_vec_(rates),
                                                        taus_vec_(taus_vec),
                                                        pseudo_roots_vec_(pseudo_roots_vec),
                                                        displacements_vec_(displacements_vec),
                                                        factors_(factors),
                                                        steps_(steps),
                                                        rates_(rates),
                                                        cutoffLevel_(cutoffLevel),
                                                        driftComputer_(taus_vec,factors,displacements_vec),
                                                        these_Drifts_vec_(rates),
                                                        these_Drifts_predicted_vec_(rates),
                                                        current_rates_vec_(rates),
                                                        current_log_rates_vec_(rates),
                                                        fixedDrifts_vec_(steps*rates)
{
    if (initial_Rates_vec.size()    != rates)
        GenerateError("rates array missized on entry to LMM_evolver_pc_class_gold.");

    if (taus_vec.size() != rates)
        GenerateError("taus array missized on entry to LMM_evolver_pc_class_gold.");

    if (displacements_vec.size() != rates)
        GenerateError("taus array missized on entry to LMM_evolver_pc_class_gold.");


    for (int i=0; i < rates; ++i)
        initial_log_Rates_vec_[i] = log(initial_Rates_vec_[i]+displacements_vec[i]);

    MatrixFacade<double> fixedDrifts_mat(fixedDrifts_vec_,steps,rates);
    CubeConstFacade<double> pseudo_roots_cube(pseudo_roots_vec_,steps,rates,factors);
    for (int s=0; s<steps; ++s)
        for (int r=0; r < rates; ++r)
        {
            double x=0.0;
            for (int f=0; f < factors; ++f)
            {
                double y = pseudo_roots_cube(s,r,f);
                x+=y*y;
            }
            fixedDrifts_mat(s,r) = -0.5*x;
        }

}

template<class D>
void
    LMM_evolver_pc_class_gold<D>::generateRates(const std::vector<D>& correlatedBrownianIncrements_vec,
    int paths,
    std::vector<D>& evolved_rates_vec,
    std::vector<D>& evolved_log_rates_vec) const
{
    evolved_rates_vec.resize(paths*rates_*steps_); 
    evolved_log_rates_vec.resize(paths*rates_*steps_); 

    CubeFacade<D> outputRates_cube(&evolved_rates_vec[0],steps_,rates_,paths);
    CubeFacade<D> outputLogRates_cube(&evolved_log_rates_vec[0],steps_,rates_,paths);

    CubeConstFacade<D> correlatedBrownianIncrements_cube(&correlatedBrownianIncrements_vec[0],steps_,rates_,paths);
    CubeConstFacade<D> pseudo_roots_cube(&pseudo_roots_vec[0],steps_,rates_,factors);


    LMM_evolver_pc_with_workspace_gold<D>( initial_Rates_vec_,
        taus_vec_,
        correlatedBrownianIncrements_vec_, //  AZ  + mu_fixed
        pseudo_roots_vec_,
        displacements_vec_, 
        paths,
        factors_,
        steps_, 
        rates_, 
        evolved_rates_vec_,
        evolved_log_rates_vec_,
        cutoffLevel_, 
        // precomputed and wsp 
        initial_log_Rates_vec_ // must equal logs of initial_Rates_vec
        ,    these_Drifts_vec_   //(rates);
        ,  these_Drifts_predicted_vec_ //(rates);
        ,    current_rates_vec_//(rates);
        ,    current_log_rates_vec_//(rates);
        ,    driftComputer_//(taus_vec,factors,displacements_vec);
        );
}


template<class D>
void
    LMM_evolver_pc_class_gold<D>::generateRatesFromUncorrelated(const std::vector<D>& brownianIncrements_vec,
    int paths,
    std::vector<D>& evolved_rates_vec,
    std::vector<D>& evolved_log_rates_vec) const
{
    evolved_rates_vec.resize(paths*rates_*steps_); 
    evolved_log_rates_vec.resize(paths*rates_*steps_); 
    correlated_brownian_vec_.resize(paths*rates_*steps_);

    CubeFacade<D> outputRates_cube(&evolved_rates_vec[0],steps_,rates_,paths);
    CubeFacade<D> outputLogRates_cube(&evolved_log_rates_vec[0],steps_,rates_,paths);

    CubeFacade<D> correlatedBrownianIncrements_cube(&correlated_brownian_vec_[0],steps_,rates_,paths);
    CubeConstFacade<D> pseudo_roots_cube(&pseudo_roots_vec_[0],steps_,rates_,factors_);
    CubeConstFacade<D> brownianIncrements_cube(&brownianIncrements_vec[0],steps_,factors_,paths);

    MatrixConstFacade<D> fixedDrifts_mat(fixedDrifts_vec_,steps_,rates_);


  
    for (int p=0; p < paths; ++p)
    {
        for (int r=0; r < rates_; ++r)
        {
            double x=0.0;
            for (int s=0; s < steps_; ++s)
            {
                x=fixedDrifts_mat(s,r);
                for (int f=0; f < factors_; ++f)
                {

                    x+=pseudo_roots_cube(s,r,f)*brownianIncrements_cube(s,f,p);
                }
                correlatedBrownianIncrements_cube(s,r,p) = x;
            }
        }
    }


    LMM_evolver_pc_with_workspace_gold<D>( initial_Rates_vec_,
        taus_vec_,
        correlated_brownian_vec_, //  AZ  + mu_fixed
        pseudo_roots_vec_,
        displacements_vec_, 
        paths,
        factors_,
        steps_, 
        rates_, 
        evolved_rates_vec,
        evolved_log_rates_vec,
        cutoffLevel_, 
        // precomputed and wsp 
        initial_log_Rates_vec_ // must equal logs of initial_Rates_vec
        ,    these_Drifts_vec_   //(rates);
        ,  these_Drifts_predicted_vec_ //(rates);
        ,    current_rates_vec_//(rates);
        ,    current_log_rates_vec_//(rates);
        ,    driftComputer_//(taus_vec,factors,displacements_vec);
        );
}
#endif

