
#ifndef LMM_EVOLVER_FULL_GOLD_H
#define LMM_EVOLVER_FULL_GOLD_H
// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3
#include <vector> 

#include <gold/sobol_gold.h>
#include <gold/scramble_gold.h>
#include <gold/InverseCumulatives_gold.h>
#include <gold/Bridge_gold.h>
#include <gold/MatrixFacade.h> 
#include <gold/Correlate_drift_gold.h>
#include <gold/LMM_evolver_gold.h>
#include <ctime>


// this routine assumes all the workspace stuff is set up correctly
// it's supposed to be called from the non workspace or class version
template<class D>
void LMMEvolutionRoutineUsingWorkspacesGold(int paths, 
                                                       int pathOffset, 
                                                       int rates, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       const std::vector<unsigned int>& scrambler_vec, 
                                                       const std::vector<D>& pseudoRoots_vec,
                                                       const std::vector<D>& fixedDrifts_vec, 
                                                       const std::vector<D>& displacements_vec,
                                                       const std::vector<D>& initial_rates_vec, 
                                                       const std::vector<D>& initial_log_rates_vec, 
                                                       const std::vector<D>& taus_vec, 
                                                   //    const std::vector<D>& initial_drifts_vec, 
                                                    //   const std::vector<int>& aliveIndices, 
                                                  //     const  std::vector<int>& alive_vec,
                                                  D cutOffLevel_,
                                                        std::vector<D>& evolved_rates_vec,
                                                        std::vector<D>& evolved_log_rates_vec,
                                                        // workspace
                                                          std::vector<unsigned int>&SobolInts_buffer_vec,
                                                         std::vector<unsigned int>& SobolInts_buffer_scrambled_vec,
                                                         std::vector<D>& quasiRandoms_buffer_vec,
                                                          std::vector<D>&bridgeVariates_vec,
                                                         std::vector<D>& correlatedVariates_vec,
                                                         std::vector<unsigned int>& directions_vec,
                                                         BrownianBridgeMultiDim<D> BB,
                                                         int tot_dimensions,
                                                            std::vector<D>& initial_log_Rates_vec // must equal logs of initial_Rates_vec
                                        ,   std::vector<D>&  these_Drifts_vec   //(rates);
                                        , std::vector<D>& these_Drifts_predicted_vec //(rates);
                                        ,   std::vector<D>& current_rates_vec//(rates);
                                        ,   std::vector<D>& current_log_rates_vec//(rates);
                                        ,   spotDriftComputer<D>& driftComputer
                                       , std::vector<D>& pathVariates_vec
                                        , std::vector<D>& bridgedPathVariates_vec 
                                                        )
{
    int NintVariates = tot_dimensions*paths*factors;

    SobolInts_buffer_vec.resize(NintVariates); 
    SobolInts_buffer_scrambled_vec.resize(NintVariates); 

    quasiRandoms_buffer_vec.resize(NintVariates); 

    bridgeVariates_vec.resize(NintVariates); 
    correlatedVariates_vec.resize(paths*rates*stepsForEvolution);
    evolved_rates_vec.resize(paths*rates*stepsForEvolution);
    evolved_log_rates_vec.resize(paths*rates*stepsForEvolution);
    CubeFacade<D> bridgedVariatesFacade(&bridgeVariates_vec[0],tot_dimensions,paths,factors);
    sobolCPUintsOffset(paths,  tot_dimensions*factors, pathOffset, &directions_vec[0], &SobolInts_buffer_vec[0]);



    // scrambling is in place
    scramble_gold(SobolInts_buffer_vec, // random numbers
        SobolInts_buffer_scrambled_vec,
        scrambler_vec,
        tot_dimensions*factors,
        paths);


   // inverseCumulativeShawBrickmanUnsignedIntGold<D> inv;
     inverseCumulativeUnsignedIntGold<D> inv;
 //   for (int j=0; j < static_cast<int>(SobolInts_buffer_vec.size()); ++j)
  //      quasiRandoms_buffer_vec[j] = inv(SobolInts_buffer_vec[j]);

    std::transform(SobolInts_buffer_vec.begin(),SobolInts_buffer_vec.end(),quasiRandoms_buffer_vec.begin(),inv);

  
    for (int i=0; i < paths; ++i)
    {
        for (int d=0; d < tot_dimensions*factors; ++d)
            pathVariates_vec[d] = quasiRandoms_buffer_vec[i+d*paths];

        BB.GenerateBridge(&pathVariates_vec[0],&bridgedPathVariates_vec[0]);

        for (int f=0; f < factors; ++f)
            for (int d=0; d< tot_dimensions; ++d)
                bridgedVariatesFacade(d,i,f) = bridgedPathVariates_vec[d+f*tot_dimensions];

    }



    correlate_drift_paths_gold(bridgeVariates_vec, // randon numbers
        correlatedVariates_vec, // correlated rate increments 
        pseudoRoots_vec, // correlator 
        rates*factors,
        fixedDrifts_vec, // drifts 
        factors, 
        rates,
        paths,
        stepsForEvolution);

    LMM_evolver_pc_with_workspace_gold(initial_rates_vec,
        taus_vec,
        correlatedVariates_vec, //  AZ  + mu_fixed
        pseudoRoots_vec,
        displacements_vec,
        paths,
        factors,
        stepsForEvolution, 
        rates, 
        evolved_rates_vec,
        evolved_log_rates_vec  ,
        cutOffLevel_
        ,
        initial_log_Rates_vec // must equal logs of initial_Rates_vec
                                        , these_Drifts_vec   //(rates);
                                        ,  these_Drifts_predicted_vec //(rates);
                                        ,    current_rates_vec//(rates);
                                        ,  current_log_rates_vec//(rates);
                                        ,    driftComputer//(taus_vec,factors,displacements_vec);
                                        );


}
template<class D>
D LMMEvolutionRoutineGold(int paths, 
                                                       int pathOffset, 
                                                       int rates, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       const std::vector<unsigned int>& scrambler_vec, 
                                                       const std::vector<D>& pseudoRoots_vec,
                                                       const std::vector<D>& fixedDrifts_vec, 
                                                       const std::vector<D>& displacements_vec,
                                                       const std::vector<D>& initial_rates_vec, 
                                                       const std::vector<D>& initial_log_rates_vec, 
                                                       const std::vector<D>& taus_vec, 
                                                        std::vector<D>& evolved_rates_vec,
                                                        std::vector<D>& evolved_log_rates_vec
                                                        )
{



    int t0=clock();

    int tot_dimensions = intPower(2,powerOfTwoForVariates);

    int NintVariates = tot_dimensions*paths*factors;



    std::vector<unsigned int> SobolInts_buffer_vec(NintVariates); 
    std::vector<unsigned int> SobolInts_buffer_scrambled_vec(NintVariates); 

    std::vector<D>  quasiRandoms_buffer_vec(NintVariates); 

    std::vector<D> bridgeVariates_vec(NintVariates); 
    std::vector<D> correlatedVariates_vec(paths*rates*stepsForEvolution);

    std::vector<unsigned int> directions_vec(tot_dimensions*n_directions*factors);

    initSobolDirectionVectors(tot_dimensions*factors, &directions_vec[0]);
    BrownianBridgeMultiDim<D>::ordering allocator( BrownianBridgeMultiDim<D>::triangular);

    BrownianBridgeMultiDim<D> BB(powerOfTwoForVariates, factors, allocator);
    std::vector<D> pathVariates_vec(tot_dimensions*factors);
    std::vector<D> bridgedPathVariates_vec(tot_dimensions*factors);

    
    std::vector<D> initial_log_Rates_vec(initial_rates_vec);
    for (int i=0; i < rates; ++i)
        initial_log_Rates_vec[i] = log(initial_log_Rates_vec[i]+displacements_vec[i]);

     spotDriftComputer<D> driftComputer(taus_vec,factors,displacements_vec);

     std::vector<D> these_Drifts_vec(rates);
     std::vector<D> these_Drifts_predicted_vec(rates);
     std::vector<D> current_rates_vec(rates);
     std::vector<D> current_log_rates_vec(rates);

     D cutOffLevel = 10;

    LMMEvolutionRoutineUsingWorkspacesGold( paths, 
                                                        pathOffset, 
                                                       rates, 
                                                        factors, 
                                                        stepsForEvolution, 
                                                        powerOfTwoForVariates,
                                                      scrambler_vec, 
                                                       pseudoRoots_vec,
                                                        fixedDrifts_vec, 
                                                        displacements_vec,
                                                       initial_rates_vec, 
                                                        initial_log_rates_vec, 
                                                       taus_vec, 
                                                   //    const std::vector<D>& initial_drifts_vec, 
                                                    //   const std::vector<int>& aliveIndices, 
                                                  //     const  std::vector<int>& alive_vec,
                                                  cutOffLevel,
                                                      evolved_rates_vec,
                                                        evolved_log_rates_vec,
                                                        // workspace
                                                         SobolInts_buffer_vec,
                                                         SobolInts_buffer_scrambled_vec,
                                                         quasiRandoms_buffer_vec,
                                                         bridgeVariates_vec,
                                                          correlatedVariates_vec,
                                                         directions_vec,
                                                          BB,
                                                          tot_dimensions,
                                                           initial_log_Rates_vec // must equal logs of initial_Rates_vec
                                        ,    these_Drifts_vec   //(rates);
                                        ,  these_Drifts_predicted_vec //(rates);
                                        ,   current_rates_vec//(rates);
                                        ,   current_log_rates_vec//(rates);
                                        ,    driftComputer
                                        , pathVariates_vec,
                                        , bridgedPathVariates_vec
                                                        );
  



  

    int t1=clock();

    return (t1-t0)/static_cast<D>(CLOCKS_PER_SEC);


}


#endif
