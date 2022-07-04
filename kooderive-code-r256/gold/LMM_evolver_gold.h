//
//
//                                                                                                                 LMM_evolver_gold.h
//
//

// (c) Mark Joshi 2010, 2012,2014
// This code is released under the GNU public licence version 3

#ifndef LMM_EVOLVER_GOLD_H
#define LMM_EVOLVER_GOLD_H
#include <vector>
#include <gold/MatrixFacade.h> 
#include <gold/MatrixFacade.h> 
#include <gold/Errors.h> 
#include <algorithm>
#include <cmath>
#define logCeiling  log(10.0f)

// out puts in last two entries which are appropriately resized
template<class D>
void LMM_evolver_Euler_gold(const std::vector<D>& initial_Rates_vec,
                            const std::vector<D>& taus_vec,
                            const std::vector<D>& correlatedBrownianIncrements_vec, //  AZ  + mu_fixed
                            const std::vector<D>& pseudo_roots_vec,
                            const std::vector<D>& displacements_vec,
                            int paths,
                            int factors,
                            int steps, 
                            int rates, 
                            std::vector<D>& evolved_rates_vec,
                            std::vector<D>& evolved_log_rates_vec);
template<class D>
void LMM_evolver_pc_gold(const std::vector<D>& initial_Rates_vec,
                         const std::vector<D>& taus_vec,
                         const std::vector<D>& correlatedBrownianIncrements_vec, //  AZ  + mu_fixed
                         const std::vector<D>& pseudo_roots_vec,
                         const std::vector<D>& displacements_vec, 
                         int paths,
                         int factors,
                         int steps, 
                         int rates, 
                         std::vector<D>& evolved_rates_vec,
                         std::vector<D>& evolved_log_rates_vec,
                         D cutoffLevel=100.0f);

template<class D>
class spotDriftComputer
{
public:
    spotDriftComputer(const std::vector<D>& taus_vec, int factors, const std::vector<D>& displacements_vec);

    void getDrifts(const MatrixConstFacade<D>& A, 
        const std::vector<D>& rates,
        std::vector<D>& drifts) const;
    void getDrifts(const MatrixConstFacade<D>& A, 
        const std::vector<D>& rates,
        int aliveIndex,
        std::vector<D>& drifts) const;
private:
    std::vector<D> taus_vec_;
    std::vector<D> displacements_vec_;
    mutable std::vector<D> g_;
    int factors_;
    int n_rates_;

    mutable std::vector<D> e_vec_;
    mutable MatrixFacade<D> e_;
};


template<class D>
void discount_ratios_computation_gold( const std::vector<D>& evolved_rates_vec, 
                                      const std::vector<D>& taus_vec, 
                                      const std::vector<int>& aliveIndices, 
                                      int paths,
                                      int steps, 
                                      int rates, 
                                      std::vector<D>& discounts_vec,  // for output
                                      int firstStepToDo=0
                                      );

template<class D>
void coterminal_annuity_ratios_computation_gold(  const std::vector<D>&  discounts_vec, 
                                                const std::vector<D>& taus_vec, 
                                                const std::vector<int>& aliveIndices, 
                                                int paths,
                                                int steps, 
                                                int rates, 
                                                std::vector<D>&  annuities_vec // for output 
                                                , int firstStepToDo=0
                                                );
template<class D>
void coterminal_swap_rates_computation_gold(  const std::vector<D>&  discounts_vec, 
                                            const std::vector<D>& annuities_vec, 
                                            const std::vector<int>& aliveIndices, 
                                            int paths,
                                            int steps, 
                                            int rates, 
                                            std::vector<D>&  swap_rates_vec // for output 
                                            , int firstStepToDo=0
                                            );
template<class D>
void swap_rate_computation_gold( const std::vector<D>& discounts_vec, 
                                const std::vector<D>&taus_vec, 
                                int startIndex,
                                int endIndex, 
                                int paths,
                                int step_for_offset_in,
                                int step_for_offset_out, 
                                int steps,
                                int rates, 
                                std::vector<D>& swap_rate_vec  
                                );

/*
template<class D>
void spot_measure_numeraires_computation_gold(   const std::vector<D>& discount_factors_vec,
                                              int paths,
                                              int n_rates,
                                              std::vector<D>& numeraire_values_vec //output
                                              );
                                              */
template<class D>
void spot_measure_numeraires_computation_offset_gold(   const std::vector<D>& discount_factors_vec,
                                                     int paths,
                                                     int outputPaths,
                                                     int offsetForOutput,
                                                     int n_rates,
                                                     std::vector<D>& numeraire_values_vec //output
                                                     );
template<class D>
void forward_rate_extraction_gold(  const std::vector<D>& forwards_vec, 
                                  const std::vector<int>& forwardIndices,                          
                                  int paths,
                                  int steps,
                                  int rates, 
                                  std::vector<D>&  select_forwards_vec,
                                   int firstStepToDo=0
                                  );
template<class D>
void forward_rate_extraction_selecting_steps_gold(  const std::vector<D>& forwards_vec, 
                                                  const std::vector<int>& forwardIndices,     
                                                  const std::vector<int>& stepIndices,
                                                  int paths,
                                                  int rates, 
                                                  int steps,
                                                  std::vector<D>&  select_forwards_vec  ,
                                                   int firstStepToDo=0
                                                  );

// we only want the numeraire values at the exercise times when doing 
// regression, this routine extracts those values
// some_numeraire_values_vec is resized to numberExerciseTimes*paths
// as usual paths is the smallest dimension
template<class D>
void spot_measure_numeraires_extraction_gold(   const  std::vector<D>& all_numeraire_values_vec,
                                             int paths,
                                             int n_rates,
                                             int numberExerciseTimes, 
                                             const std::vector<int>& exerciseIndices_vec,
                                             std::vector<D>& some_numeraire_values_vec //output
                                             );



template<class D>
spotDriftComputer<D>::spotDriftComputer(const std::vector<D>& taus_vec,
                                        int factors,
                                        const std::vector<D>& displacements_vec): taus_vec_(taus_vec), displacements_vec_(displacements_vec),
                                        g_(taus_vec.size()), 
                                        factors_(factors), 
                                        n_rates_(static_cast<int>(taus_vec.size())), e_vec_(factors*n_rates_), e_(&e_vec_[0],n_rates_,factors)
{

}
template<class D>
void   spotDriftComputer<D>::getDrifts(const MatrixConstFacade<D>& A, 
                                       const std::vector<D>& rates,
                                       std::vector<D>& drifts) const
{

    // first compute g

    for (int r=0; r < n_rates_; ++r)
        g_[r] = (rates[r]+displacements_vec_[r])*taus_vec_[r]/(1+ rates[r]*taus_vec_[r]);

    for (int f=0; f < factors_; ++f)
    {
        e_(0,f) = A(0,f)*g_[0];
        for (int r=1; r < n_rates_; ++r)
            e_(r,f) = e_(r-1,f) + A(r,f)*g_[r];
    }

    for (int r=0; r < n_rates_; ++r)
    {
        D x=0.0f;
        for (int f=0; f < factors_; ++f)
            x+= A(r,f)*e_(r,f);
        drifts[r] = x;
    }

}

template<class D>
void   spotDriftComputer<D>::getDrifts(const MatrixConstFacade<D>& A, 
                                       const std::vector<D>& rates,
                                       int aliveIndex,
                                       std::vector<D>& drifts) const
{

    // first compute g

    for (int r=aliveIndex; r < n_rates_; ++r)
        g_[r] = (rates[r]+displacements_vec_[r])*taus_vec_[r]/(1+ rates[r]*taus_vec_[r]);

    for (int f=0; f < factors_; ++f)
    {
        e_(aliveIndex,f) = A(aliveIndex,f)*g_[0];
        for (int r=aliveIndex+1; r < n_rates_; ++r)
            e_(r,f) = e_(r-1,f) + A(r,f)*g_[r];
    }

    for (int r=aliveIndex; r < n_rates_; ++r)
    {
        D x=0.0f;
        for (int f=0; f < factors_; ++f)
            x+= A(r,f)*e_(r,f);
        drifts[r] = x;
    }

}


template<class D>
void LMM_evolver_Euler_gold(const std::vector<D>& initial_Rates_vec,
                            const std::vector<D>& taus_vec,
                            const std::vector<D>& correlatedBrownianIncrements_vec, //  AZ  + mu_fixed
                            const std::vector<D>& pseudo_roots_vec,
                            const std::vector<D>& displacements_vec, 
                            int paths,
                            int factors,
                            int steps, 
                            int rates, 
                            std::vector<D>& evolved_rates_vec,
                            std::vector<D>& evolved_log_rates_vec)
{
    if (initial_Rates_vec.size()    != rates)
        GenerateError("rates array missized on entry to LMM_evolver_Euler_gold.");

    if (taus_vec.size() != rates)
        GenerateError("taus array missized on entry to LMM_evolver_Euler_gold.");

    if (displacements_vec.size() != rates)
        GenerateError("taus array missized on entry to LMM_evolver_Euler_gold.");

    if (static_cast<int>(correlatedBrownianIncrements_vec.size()) < paths*rates*steps)
        GenerateError("correlatedBrownianIncrements_vec missized.");

    evolved_rates_vec.resize(paths*rates*steps); 
    evolved_log_rates_vec.resize(paths*rates*steps); 

    CubeFacade<D> outputRates_cube(&evolved_rates_vec[0],steps,rates,paths);
    CubeFacade<D> outputLogRates_cube(&evolved_log_rates_vec[0],steps,rates,paths);

    CubeConstFacade<D> correlatedBrownianIncrements_cube(&correlatedBrownianIncrements_vec[0],steps,rates,paths);
    CubeConstFacade<D> pseudo_roots_cube(&pseudo_roots_vec[0],steps,rates,factors);

    std::vector<D> initial_logRates_vec(rates);
    for (int i=0; i < rates; ++i)
        initial_logRates_vec[i] = logf(initial_Rates_vec[i]+displacements_vec[i]);


    std::vector<D> these_Drifts_vec(rates);

    std::vector<D> current_rates_vec(rates);
    std::vector<D> current_log_rates_vec(rates);

    spotDriftComputer<D> driftComputer(taus_vec,factors,displacements_vec);

    for (int p=0; p < paths; ++p)
    {
        current_rates_vec = initial_Rates_vec;
        current_log_rates_vec = initial_logRates_vec;


        for (int s=0; s < steps; ++s)
        {
            driftComputer.getDrifts(pseudo_roots_cube[s],  current_rates_vec,  these_Drifts_vec);

            for (int r=0; r < rates; ++r)
            {
                D x = (current_log_rates_vec[r] += these_Drifts_vec[r] +
                    correlatedBrownianIncrements_cube(s,r,p));

                if (x >logCeiling)
                    x = current_log_rates_vec[r] =logCeiling;

                outputRates_cube(s,r,p) =  current_rates_vec[r] = expf(x)-displacements_vec[r];
                outputLogRates_cube(s,r,p) = x;


            }

        }
    }

}


// all inputs should be correctly sized on entry
// this is really an auxiliary function not to be called directly
// use the class version or LMM_evolver_pc_gold 
template<class D>
void LMM_evolver_pc_with_workspace_gold(const std::vector<D>& initial_Rates_vec,
                                        const std::vector<D>& taus_vec,
                                        const std::vector<D>& correlatedBrownianIncrements_vec, //  AZ  + mu_fixed
                                        const std::vector<D>& pseudo_roots_vec,
                                        const std::vector<D>& displacements_vec, 
                                        int paths,
                                        int factors,
                                        int steps, 
                                        int rates, 
                                        std::vector<D>& evolved_rates_vec,
                                        std::vector<D>& evolved_log_rates_vec,
                                        D cutoffLevel, 
                                        // precomputed and wsp 
                                        const std::vector<D>& initial_log_Rates_vec // must equal logs of initial_Rates_vec
                                        ,   std::vector<D>&  these_Drifts_vec   //(rates);
                                        , std::vector<D>& these_Drifts_predicted_vec //(rates);
                                        ,   std::vector<D>& current_rates_vec//(rates);
                                        ,   std::vector<D>& current_log_rates_vec//(rates);
                                        ,   const spotDriftComputer<D>& driftComputer//(taus_vec,factors,displacements_vec);
                                        )
{


    CubeFacade<D> outputRates_cube(&evolved_rates_vec[0],steps,rates,paths);
    CubeFacade<D> outputLogRates_cube(&evolved_log_rates_vec[0],steps,rates,paths);

    CubeConstFacade<D> correlatedBrownianIncrements_cube(&correlatedBrownianIncrements_vec[0],steps,rates,paths);
    CubeConstFacade<D> pseudo_roots_cube(&pseudo_roots_vec[0],steps,rates,factors);




    for (int p=0; p < paths; ++p)
    {
        current_rates_vec = initial_Rates_vec;
        current_log_rates_vec = initial_log_Rates_vec;


        for (int s=0; s < steps; ++s)
        {
            driftComputer.getDrifts(pseudo_roots_cube[s],  current_rates_vec,  these_Drifts_vec);

            for (int r=0; r < rates; ++r)
            {
                current_log_rates_vec[r] += these_Drifts_vec[r] +    correlatedBrownianIncrements_cube(s,r,p);
                current_rates_vec[r] = exp( current_log_rates_vec[r])-displacements_vec[r];

            }

            driftComputer.getDrifts(pseudo_roots_cube[s],  current_rates_vec,  these_Drifts_predicted_vec);


            for (int r=0; r < rates; ++r)
            {
                D x = (current_log_rates_vec[r] += 0.5f*(these_Drifts_predicted_vec[r] -  these_Drifts_vec[r] ));
                if (x>cutoffLevel)
                {
                    x=cutoffLevel;
                    current_log_rates_vec[r]=x;

                }
                current_rates_vec[r] = exp( current_log_rates_vec[r])-displacements_vec[r];

                outputRates_cube(s,r,p) =  current_rates_vec[r] ;
                outputLogRates_cube(s,r,p) = x;


            }

        }
    }

}


template<class D>
void LMM_evolver_pc_gold(const std::vector<D>& initial_Rates_vec,
                         const std::vector<D>& taus_vec,
                         const std::vector<D>& correlatedBrownianIncrements_vec, //  AZ  + mu_fixed
                         const std::vector<D>& pseudo_roots_vec,
                         const std::vector<D>& displacements_vec, 
                         int paths,
                         int factors,
                         int steps, 
                         int rates, 
                         std::vector<D>& evolved_rates_vec,
                         std::vector<D>& evolved_log_rates_vec,
                         D cutoffLevel)
{
    if (initial_Rates_vec.size()    != rates)
        GenerateError("rates array missized on entry to LMM_evolver_pc_gold.");

    if (taus_vec.size() != rates)
        GenerateError("taus array missized on entry to LMM_evolver_pc_gold.");

    if (displacements_vec.size() != rates)
        GenerateError("taus array missized on entry to LMM_evolver_pc_gold.");

    if (static_cast<int>(correlatedBrownianIncrements_vec.size()) < paths*rates*steps)
        GenerateError("correlatedBrownianIncrements_vec missized.");

    evolved_rates_vec.resize(paths*rates*steps); 
    evolved_log_rates_vec.resize(paths*rates*steps); 

    CubeFacade<D> outputRates_cube(&evolved_rates_vec[0],steps,rates,paths);
    CubeFacade<D> outputLogRates_cube(&evolved_log_rates_vec[0],steps,rates,paths);

    CubeConstFacade<D> correlatedBrownianIncrements_cube(&correlatedBrownianIncrements_vec[0],steps,rates,paths);
    CubeConstFacade<D> pseudo_roots_cube(&pseudo_roots_vec[0],steps,rates,factors);

    std::vector<D> initial_log_Rates_vec(initial_Rates_vec);
    for (int i=0; i < rates; ++i)
        initial_log_Rates_vec[i] = log(initial_log_Rates_vec[i]+displacements_vec[i]);

     spotDriftComputer<D> driftComputer(taus_vec,factors,displacements_vec);

     std::vector<D> these_Drifts_vec(rates);
     std::vector<D> these_Drifts_predicted_vec(rates);
     std::vector<D> current_rates_vec(rates);
     std::vector<D> current_log_rates_vec(rates);

     LMM_evolver_pc_with_workspace_gold<D>( initial_Rates_vec,
                                         taus_vec,
                                         correlatedBrownianIncrements_vec, //  AZ  + mu_fixed
                                         pseudo_roots_vec,
                                        displacements_vec, 
                                         paths,
                                        factors,
                                         steps, 
                                         rates, 
                                         evolved_rates_vec,
                                        evolved_log_rates_vec,
                                        cutoffLevel, 
                                        // precomputed and wsp 
                                         initial_log_Rates_vec // must equal logs of initial_Rates_vec
                                        ,    these_Drifts_vec   //(rates);
                                        ,  these_Drifts_predicted_vec //(rates);
                                        ,    current_rates_vec//(rates);
                                        ,    current_log_rates_vec//(rates);
                                        ,    driftComputer//(taus_vec,factors,displacements_vec);
                                        );

}

template<class D>
void discount_ratios_computation_gold( const std::vector<D>& evolved_rates_vec, 
                                      const std::vector<D>& taus_vec, 
                                      const std::vector<int>& aliveIndices, 
                                      int paths,
                                      int steps, 
                                      int rates, 
                                      std::vector<D>& discounts_vec,  // for output 
                                      int firstStepToDo
                                      )
{

    CubeConstFacade<D> evolved_rates_cube(&evolved_rates_vec[0], steps, rates, paths);
    CubeFacade<D> discounts_cube(&discounts_vec[0], steps, rates+1, paths);

    for (int s=firstStepToDo; s < steps; ++s)
    {
        int alive = aliveIndices[s];

        for (int p=0; p < paths; ++p)
        {
            D df=1.0;
            discounts_cube(s,alive,p)=df;

            for (int r=alive; r < rates; ++r)
            {
                df /= 1 + taus_vec[r]*evolved_rates_cube(s,r,p);
                discounts_cube(s,r+1,p)=df;
            }
        }

    }
}

template<class D>
void coterminal_annuity_ratios_computation_gold(  const std::vector<D>&  discounts_vec, 
                                                const std::vector<D>& taus_vec, 
                                                const std::vector<int>& aliveIndices, 
                                                int paths,
                                                int steps, 
                                                int rates, 
                                                std::vector<D>&  annuities_vec // for output 
                                                 , int firstStepToDo
                                                )
{
    CubeConstFacade<D> discounts_cube(&discounts_vec[0], steps, rates+1, paths);
    CubeFacade<D> annuities_cube(&annuities_vec[0], steps, rates, paths);

    for (int p=0; p < paths; ++p)
        for (int s=firstStepToDo; s < steps; ++s)
        {
            int alive = aliveIndices[s];

            D thisAnn =0.0;
            for (int r= rates-1; r >= alive; --r)
            {
                thisAnn += taus_vec[r]*discounts_cube(s,r+1,p);

                annuities_cube(s,r,p) = thisAnn;
            }

        }

}

template<class D>
void coterminal_swap_rates_computation_gold(  const std::vector<D>&  discounts_vec, 
                                            const std::vector<D>& annuities_vec, 
                                            const std::vector<int>& aliveIndices, 
                                            int paths,
                                            int steps, 
                                            int rates, 
                                            std::vector<D>&  swap_rates_vec // for output 
                                            , int firstStepToDo
                                            )
{
    CubeConstFacade<D> discounts_cube(&discounts_vec[0], steps, rates+1, paths);
    CubeConstFacade<D> annuities_cube(&annuities_vec[0], steps, rates, paths);
    CubeFacade<D> swap_rates_cube(&swap_rates_vec[0], steps, rates, paths);


    for (int p=0; p < paths; ++p)
        for (int s=firstStepToDo; s < steps; ++s)
        {
            int alive = aliveIndices[s];


            for (int r= alive; r < rates; ++r)
            {
                D thisAnn = annuities_cube(s,r,p);
                D df2 = discounts_cube(s,rates,p);
                D df1 = discounts_cube(s,r,p);

                swap_rates_cube(s,r,p) =(df1-df2)/thisAnn;
            }

        }

}

template<class D>
void swap_rate_computation_gold( const std::vector<D>& discounts_vec, 
                                const std::vector<D>&taus_vec, 
                                int startIndex,
                                int endIndex, 
                                int paths,
                                int step_for_offset_in,
                                int step_for_offset_out, 
                                int steps,
                                int rates, 
                                std::vector<D>& swap_rate_vec 
                                )
{
    CubeConstFacade<D> discounts_cube(&discounts_vec[0], steps, rates+1, paths);
    MatrixFacade<D> SR_matrix(&swap_rate_vec[0],steps,paths);

    for (int p=0; p < paths; ++p)
    {
        D numerator = discounts_cube(step_for_offset_in,startIndex,p) - discounts_cube(step_for_offset_in,endIndex,p);

        D denom =0.0;

        for (int r = startIndex; r < endIndex; ++r)
            denom += taus_vec[r]*discounts_cube(step_for_offset_in,r+1,p);

        D sr = numerator/denom;

        SR_matrix(step_for_offset_out,p) = sr;
    }

}

template<class D>
void spot_measure_numeraires_computation_gold(   const std::vector<D>& discount_factors_vec,
                                              int paths,
                                              int n_rates,
                                              std::vector<D>& numeraire_values_vec, //output
                                              int startStep=0
                                              )
{
    CubeConstFacade<D> discounts_cube(&discount_factors_vec[0], n_rates, n_rates+1, paths);
    MatrixFacade<D> numeraires_matrix(&numeraire_values_vec[0],n_rates,paths);

    for (int p=0; p < paths; ++p)
    {
        numeraires_matrix(startStep,p)=1.0;

        for (int r=startStep+1; r < n_rates; ++r)
        {
            numeraires_matrix(r,p)= numeraires_matrix(r-1,p) *discounts_cube(r-1,r-1,p)/discounts_cube(r-1,r,p);
        }
    }
}

// writes value of discretely compounding money market account along the path into numeraire_values_vec
// output is matrix with rates/steps in large dim and paths in small.
// matrix dimension is outputPaths not paths, offsetForOutput says where to place the output
template<class D>
void spot_measure_numeraires_computation_offset_gold(   const std::vector<D>& discount_factors_vec,
                                                     int paths,
                                                     int outputPaths,
                                                     int offsetForOutput,
                                                     int n_rates,
                                                     std::vector<D>& numeraire_values_vec //output
                                                     )
{
    CubeConstFacade<D> discounts_cube(&discount_factors_vec[0], n_rates, n_rates+1, paths);
    MatrixFacade<D> numeraires_matrix(&numeraire_values_vec[0],n_rates,outputPaths);

    for (int p=0; p < paths; ++p)
    {
        int outpath = p + offsetForOutput;
        numeraires_matrix(0,outpath)=1.0;

        for (int r=1; r < n_rates; ++r)
        {
            numeraires_matrix(r,outpath)= numeraires_matrix(r-1,outpath) *discounts_cube(r-1,r-1,p)/discounts_cube(r-1,r,p);
#ifdef _DEBUG
            if ( numeraires_matrix(r,outpath)<0)
                throw("negative spot measure numeraire: something has gone badly wrong.");
#endif
        }
    }
}

template<class D>
void forward_rate_extraction_gold(  const std::vector<D>& forwards_vec, 
                                  const std::vector<int>& forwardIndices,                          
                                  int paths,
                                  int steps,
                                  int rates, 
                                  std::vector<D>&  select_forwards_vec   ,
                                  int firstStepToDo
                                  )
{
    CubeConstFacade<D> forwards_cube(&forwards_vec[0], steps, rates, paths);
    MatrixFacade<D> select_forwards_matrix(&select_forwards_vec[0],steps,paths);

    if (forwardIndices.size() != steps)
        throw("forward indices is not of size steps in forward_rate_extraction_gold");

    for (int p=0; p < paths; ++p)
        for (int s=firstStepToDo; s < steps; ++s)
            select_forwards_matrix(s,p) = forwards_cube(s,forwardIndices[s],p);


}

template<class D>
void forward_rate_extraction_selecting_steps_gold(  const std::vector<D>& forwards_vec, 
                                                  const std::vector<int>& forwardIndices,     
                                                  const std::vector<int>& stepIndices,
                                                  int paths,
                                                  int rates, 
                                                  int steps,
                                                  std::vector<D>&  select_forwards_vec,
                                                  int firstStepToDo
                                                  )
{
    if (forwardIndices.size() != stepIndices.size() )
        throw("forward indices is not of size stepIndices in forward_rate_extraction_gold");

    int stepsToConsider = static_cast<int>(forwardIndices.size());

    CubeConstFacade<D> forwards_cube(&forwards_vec[0], steps, rates, paths);
    MatrixFacade<D> select_forwards_matrix(&select_forwards_vec[0],stepsToConsider,paths);



    for (int p=0; p < paths; ++p)
        for (int s=firstStepToDo; s < stepsToConsider; ++s)
            select_forwards_matrix(s,p) = forwards_cube(stepIndices[s],forwardIndices[s],p);


}

template<class D>
void spot_measure_numeraires_extraction_gold(   const  std::vector<D>& all_numeraire_values_vec,
                                             int paths,
                                             int n_rates,
                                             int numberExerciseTimes, 
                                             const std::vector<int>& exerciseIndices_vec,
                                             std::vector<D>& some_numeraire_values_vec //output
                                             )
{
    MatrixConstFacade<D> all_numeraires_matrix(&all_numeraire_values_vec[0],n_rates,paths);
    MatrixFacade<D> some_numeraires_matrix(&some_numeraire_values_vec[0],numberExerciseTimes,paths);
    some_numeraire_values_vec.resize(paths*numberExerciseTimes);

    for (int p=0; p < paths; ++p)
    {

        for (int r=0; r < numberExerciseTimes; ++r)
        {
            some_numeraires_matrix(r,p)= all_numeraires_matrix(exerciseIndices_vec[r],p);
        }
    }
}
#endif
