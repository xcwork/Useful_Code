//
//
//
//                                           MultiD_BS_evolver_classes_gold.h
//
//

#ifndef MULTID_EVOLVER_CLASSES_GOLD_H

#define MULTID_EVOLVER_CLASSES_GOLD_H

#include <gold/MultiD_Path_Gen_BS_gold.h>
#include <vector> 

#include <gold/sobol_gold.h>
#include <gold/scramble_gold.h>
#include <gold/InverseCumulatives_gold.h>
#include <gold/Bridge_gold.h>
#include <gold/MatrixFacade.h> 
#include <gold/Correlate_drift_gold.h>
#include <ctime>

// template implemementations are at the end of the file

class exponentials
{
public: 
    exponentials(){}

    double operator()(double x)const {return exp(x);}
};

// this routine assumes all the workspace stuff is set up correctly
// it's supposed to be called from the non workspace or class version
template<class D>
void MultiDBSUsingWorkspacesGold(int paths, 
                                                       int pathOffset, 
                                                       int stocks, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                        const std::vector<unsigned int>& scrambler_vec, 
                                                        const std::vector<D>& pseudoRoots_vec,
                                                        const std::vector<D>& fixedDrifts_vec, 
                                                        const std::vector<D>& initial_log_stocks_vec, 
                                                        std::vector<D>& evolved_stocks_vec,
                                                        std::vector<D>& evolved_log_stocks_vec,
                                                        // workspace
                                                          std::vector<unsigned int>&SobolInts_buffer_vec,
                                                         std::vector<unsigned int>& SobolInts_buffer_scrambled_vec,
                                                         std::vector<D>& quasiRandoms_buffer_vec,
                                                          std::vector<D>&bridgeVariates_vec,
                                                         std::vector<D>& correlatedVariates_vec,
                                                         std::vector<unsigned int>& directions_vec,
                                                         BrownianBridgeMultiDim<D> BB,
                                                         int tot_dimensions,
                                         std::vector<D>& pathVariates_vec,
                                         std::vector<D>& bridgedPathVariates_vec 
                                                        )
{
    int NintVariates = tot_dimensions*paths*factors;

    SobolInts_buffer_vec.resize(NintVariates); 
    SobolInts_buffer_scrambled_vec.resize(NintVariates); 

    quasiRandoms_buffer_vec.resize(NintVariates); 

    bridgeVariates_vec.resize(NintVariates); 
    correlatedVariates_vec.resize(paths*stocks*stepsForEvolution);
    evolved_stocks_vec.resize(paths*stocks*stepsForEvolution);
    evolved_log_stocks_vec.resize(paths*stocks*stepsForEvolution);
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
        stocks*factors,
        fixedDrifts_vec, // drifts 
        factors, 
        stocks,
        paths,
        stepsForEvolution);
   
    multi_dim_BS_path_generator_log_gold(initial_log_stocks_vec,
                                         correlatedVariates_vec,
                                   //      fixedDrifts_vec,
                                         evolved_log_stocks_vec, 
                                         paths,
                                         stocks,
                                         stepsForEvolution);
    exponentials exponentialfunction;

    std::transform(evolved_log_stocks_vec.begin(),evolved_log_stocks_vec.end(),evolved_stocks_vec.begin(),exponentialfunction);
   


}

// if you want to use an evolver several times without having to reset up workspace etc


template<class D>
class MultiDBS_PathGen_Sobol_gold
{
public:
    MultiDBS_PathGen_Sobol_gold(int indicativePaths, 
        int numberStocks, 
        int factors, 
        int stepsForEvolution, 
        int powerOfTwoForVariates,
        const std::vector<D>& pseudoRoots_vec,
        const std::vector<D>& fixed_drifts_vec,
        const std::vector<D>& initial_stocks_vec
        );

    D getPaths(int paths,int pathOffset, const std::vector<unsigned int>& scrambler_vec,  
        std::vector<D>& evolved_stocks_vec,
        std::vector<D>& evolved_log_stocks_vec);



private:
    int lastPaths_;
    int stocks_;
    int factors_;
    int stepsForEvolution_; 
    int powerOfTwoForVariates_;

    std::vector<D> pseudoRoots_vec_;
    std::vector<D> fixedDrifts_vec_; 

    std::vector<D> initial_stocks_vec_;
    std::vector<D> initial_log_stocks_vec_; 

    D cutOffLevel_;

    // stuff to set up
    int tot_dimensions_;
    int NintVariates_;
    std::vector<unsigned int> SobolInts_buffer_vec_;//(NintVariates); 
    std::vector<unsigned int> SobolInts_buffer_scrambled_vec_;//(NintVariates); 

    std::vector<D>  quasiRandoms_buffer_vec_;//(NintVariates); 

    std::vector<D> bridgeVariates_vec_;//(NintVariates); 
    std::vector<D> correlatedVariates_vec_;//(paths*stocks*stepsForEvolution);

    std::vector<unsigned int> directions_vec_;//(tot_dimensions*n_directions*factors);

 //   BrownianBridgeMultiDim<D>::ordering allocator_; //( BrownianBridgeMultiDim::triangular);

    BrownianBridgeMultiDim<D> BB_;//(powerOfTwoForVariates, factors, allocator);
    std::vector<D> variates_; //(tot_dimensions*factors);
    std::vector<D> bridgedVariates_; //(tot_dimensions*factors);


   
    std::vector<D> current_stock_vec_; // (stocks);
    std::vector<D> current_stocks_rates_vec_; //(stocks);
    std::vector<D> pathVariates_vec_; //(stocks);
    std::vector<D> bridgedPathVariates_vec_; //(stocks);
};

// template implemementations

template<class D>
MultiDBS_PathGen_Sobol_gold<D>::MultiDBS_PathGen_Sobol_gold(int indicativePaths, 
        int numberStocks, 
        int factors, 
        int stepsForEvolution, 
        int powerOfTwoForVariates,
        const std::vector<D>& pseudoRoots_vec,
        const std::vector<D>& fixed_drifts_vec,
        const std::vector<D>& initial_stocks_vec
                                                              )
                                                              :lastPaths_(indicativePaths),
                                                              stocks_(numberStocks), 
                                                              factors_(factors), 
                                                              stepsForEvolution_(stepsForEvolution), 
                                                              powerOfTwoForVariates_(powerOfTwoForVariates),
                                                              pseudoRoots_vec_(pseudoRoots_vec),
                                                              fixedDrifts_vec_(fixed_drifts_vec), 
                                                              initial_stocks_vec_(initial_stocks_vec),
                                                              initial_log_stocks_vec_(initial_stocks_vec),
                                                              tot_dimensions_(intPower(2,powerOfTwoForVariates)),
                                                              NintVariates_(tot_dimensions_*indicativePaths*factors),
                                                              SobolInts_buffer_vec_(NintVariates_), 
                                                              SobolInts_buffer_scrambled_vec_(NintVariates_), 
                                                              quasiRandoms_buffer_vec_(NintVariates_),
                                                              bridgeVariates_vec_(NintVariates_),
                                                              correlatedVariates_vec_(indicativePaths*numberStocks*stepsForEvolution),
                                                              directions_vec_(tot_dimensions_*n_directions*factors),
                                                      //        allocator_( BrownianBridgeMultiDim::triangular),
                                                              BB_(powerOfTwoForVariates_, factors, BrownianBridgeMultiDim<D>::triangular),
                                                              variates_(tot_dimensions_*factors),
                                                              bridgedVariates_(tot_dimensions_*factors),
                                                         //    current_stocks_vec_(numberStocks),
                                                         //    current_log_stocks_vec_(numberStocks),
                                                              pathVariates_vec_(tot_dimensions_*factors),
                                                              bridgedPathVariates_vec_(tot_dimensions_*factors)

{

    initSobolDirectionVectors(tot_dimensions_*factors_, &directions_vec_[0]);

    for (int i=0; i < stocks_; ++i)
        initial_log_stocks_vec_[i] = log(initial_stocks_vec_[i]);

    CubeConstFacade<D> pseudo_roots_cube(&pseudoRoots_vec_[0],stepsForEvolution_,stocks_,factors);
    MatrixFacade<D> fixedDrifts_mat(&fixedDrifts_vec_[0],stepsForEvolution_,stocks_);
 
    // subtract the variance terms from the drifts, so go from r-d to r-d -0.5 *\sigma^2
    for (int s=0; s < stepsForEvolution_; ++s)
    {
        for (int r=0; r< stocks_; ++r)
        {
            D v=0.0;
            for (int f=0; f < factors;++f)
            {
                D x= pseudo_roots_cube(s,r,f);
                v+= x*x;
            }
            fixedDrifts_mat(s,r) -= 0.5*v;
        }
    }


}
template<class D>
D MultiDBS_PathGen_Sobol_gold<D>::getPaths(int paths, 
                                            int pathOffset,
                                            const std::vector<unsigned int>& scrambler_vec,  
                                            std::vector<D>& evolved_stocks_vec,
                                            std::vector<D>& evolved_log_stocks_vec) 
{
    int t0=clock();

    if (paths!=lastPaths_)
    {
        int NintVariates = tot_dimensions_*paths*factors_;

        SobolInts_buffer_vec_.resize(NintVariates); 
        SobolInts_buffer_scrambled_vec_.resize(NintVariates); 

        quasiRandoms_buffer_vec_.resize(NintVariates); 

        bridgeVariates_vec_.resize(NintVariates); 
        correlatedVariates_vec_.resize(paths*stocks_*stepsForEvolution_);
        lastPaths_=paths;
    }



    MultiDBSUsingWorkspacesGold<D>( paths, 
        pathOffset, 
        stocks_, 
        factors_, 
        stepsForEvolution_, 
        powerOfTwoForVariates_,
        scrambler_vec, 
        pseudoRoots_vec_,
        fixedDrifts_vec_, 
            initial_log_stocks_vec_, 
        evolved_stocks_vec,
        evolved_log_stocks_vec,
        // workspace
        SobolInts_buffer_vec_,
        SobolInts_buffer_scrambled_vec_,
        quasiRandoms_buffer_vec_,
        bridgeVariates_vec_,
        correlatedVariates_vec_,
        directions_vec_,
        BB_,
        tot_dimensions_,
         pathVariates_vec_
        , bridgedPathVariates_vec_
        );


    int t1=clock();

    return (t1-t0)/static_cast<D>(CLOCKS_PER_SEC);


}


#endif

