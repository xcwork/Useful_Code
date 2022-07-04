/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
//
//
//                                                LMM_evolver_all_gpu
//
//
//

// (c) Mark Joshi 2010, 2013
// This code is released under the GNU public licence version 3

/*
    do all the drifts and rate increments simultaneously
*/
#include "LMM_evolver_all.h"

#include "cudaMacros.h"
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <iostream>
#include <CUDAConditionalMacros.h>

#define upperMax 20.0f

texture<float, 1, cudaReadModeElementType> tex_variates;
texture<int, 1, cudaReadModeElementType> tex_alive;
texture<float, 1, cudaReadModeElementType> tex_fixed_drifts;
texture<float, 1, cudaReadModeElementType> tex_pseudoRoots;
texture<float, 1, cudaReadModeElementType> tex_taus;
texture<float, 1, cudaReadModeElementType> tex_displacements;

// for step 0 only
texture<float, 1, cudaReadModeElementType> tex_initialLogRates;
texture<float, 1, cudaReadModeElementType> tex_initialRates;
texture<float, 1, cudaReadModeElementType> tex_initial_drifts;



#define SHAREDSIZE 1280
const int extraOff = 0;


__global__ void LMM_evolver_all_steps_pc_kernel(int pseudoRoots_offset, // texture offset 
                                                int taus_offset,  // texture offset 
                                                int dis_offset, // texture offset
                                                int fixed_drifts_offset, // texture offset 
                                                int AOffsetPerStep,  // change in texture offset for each pseudo-root 
                                                int alive_offset, // offset for tex_alive
                                                int initialLogRatesOffset, // offset for tex_initialLogRates;
                                                int initialRatesOffset, // offset for tex_initialRates
                                                int initial_drifts_offset, // texture offset 
                                                int variates_offset, 
                                                float* rates_output, 
                                                int factors, 
                                                int paths,
                                                int rates,
                                                int steps
                                                )
{ // start of function

    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
   
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int bwidthe = bwidth+extraOff;
    int bwidthl=  bwidth+extraOff;

    int pathsPerThread = 1 + (( paths -1)/width);
 
    __shared__ float eShared[SHAREDSIZE];   
     float* logShared=eShared+2*factors*bwidthe;
    
    float* logSharedLoc = logShared+tx;  
    float* e_loc =eShared+tx;
    float* e_pred_loc = eShared+tx+factors*bwidthe;
 
    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        
        if (pathNumber < paths)
        {
            // step zero as special case

            int aliveIndex = tex1Dfetch(tex_alive, alive_offset);
            
            // set shared memory for logs to initial log values 
            for (int r=aliveIndex; r < rates; ++r)
                logSharedLoc[r*bwidthl] = tex1Dfetch( tex_initialLogRates, initialLogRatesOffset+r);

            int pathOffset =  pathNumber;
    
            int variatesOffset =    variates_offset +pathOffset*factors;

            float* rates_out_location = rates_output+pathOffset;

            int pseudoRootsStepOffset = pseudoRoots_offset;
            
            // note this is outside the rates loops  
            for (int f=0; f< factors; ++f)
               e_pred_loc[f*bwidthe]=0;
  
            for (int r=aliveIndex; r < rates; ++r)
            {

               float logfr =logSharedLoc[r*bwidthl];
             
              logfr +=  tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r);   

               float mur=  tex1Dfetch(tex_initial_drifts,initial_drifts_offset+r);                       

               logfr +=  mur;

               float dW=0.0f;

               for (int f=0; f < factors; ++f)
                   dW+= tex1Dfetch(tex_pseudoRoots,pseudoRootsStepOffset+r*factors+f) 
                                   * tex1Dfetch(tex_variates,variatesOffset+f);
               
               logfr += dW; 
                    
               // predicted, now correct
                    
               float frpDissed  = exp(logfr);
               float dis = tex1Dfetch(tex_displacements,dis_offset+r);
               float frp = frpDissed-dis;
               float taur = tex1Dfetch(tex_taus,taus_offset+r);                    
               float grp = frpDissed*taur/(1.0f+frp*taur);
               float murp=0.0f;
                    
               for (int f=0; f < factors; ++f)
               {
                  float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+r*factors+f);
                  float erIncp =  grp*arf;
                  float  thise=   ( e_pred_loc[f*bwidthe]+= erIncp);
                  murp+= arf*thise;   
               }
               logfr += 0.5f*(murp-mur);
               logSharedLoc[r*bwidthl]=logfr;
               
               rates_out_location[r*paths] =exp(logfr)-dis;
                
            } // end rates loop for step zero
            
            // now do the other steps
            for (int s =1; s < steps; ++s)
            {
                float* rates_start_step_location = rates_out_location;
                rates_out_location+= paths*rates;
                variatesOffset += factors*paths; 
                int aliveIndex = tex1Dfetch(tex_alive, alive_offset+s);
                int alivestarpaths = aliveIndex*paths;
                int pseudoRoots_step_offset = pseudoRoots_offset+s*rates*factors;
                int r=aliveIndex;
                { 
                    float taur =tex1Dfetch(tex_taus,taus_offset+r);
                    float fr = rates_start_step_location[alivestarpaths];
                    float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                    float frtaur = fr*taur;
                    float frtaud = frtaur+dis*taur;
                    float gr =  frtaud/(1.0f+frtaur);
                    float mur=0.0f;
                
                    float arf;
                    float dW=0.0f;
               
                    for (int f=0; f < factors; ++f)
                    {
                        arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float er =  gr*arf;
                        mur+= arf*er;
                        e_loc[f*bwidthe] = er;
                        dW+=arf*tex1Dfetch(tex_variates, variatesOffset+f);
                    }
               
                    //          int rateOffset = alivestarpaths;
                    
                    float change = mur+ dW+tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r+s*rates);
                    float logfr =  logSharedLoc[r*bwidthl];
                    logfr += change;
                    
                    // we have predicted logfr, now correct it 
                    
                    float frpDissed  = exp(logfr);
                    float frp = frpDissed-dis;
                    
                    float grp = frpDissed*taur/(1.0f+frp*taur);
                    float murp=0.0f;
                    
                    for (int f=0; f < factors; ++f)
                    {
                        arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float erp =  grp*arf;
                        murp+= arf*erp;
                        e_pred_loc[f*bwidthe] = erp;
                    }
                    logfr += 0.5f*(murp-mur);
                    
                    
                    logSharedLoc[r*bwidthl] = logfr;
                    rates_out_location[r*paths] = exp(logfr)-dis; 
                }
                    

                    // rate aliveIndex is done, now do other rates 
                    ++r;
                    for (; r < rates; ++r)
                    {
                        float taur =  tex1Dfetch(tex_taus,taus_offset+r);
                        float fr = rates_start_step_location[r*paths];
                        float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                        float frtaur = fr*taur;
                        float frtaud = frtaur+dis*taur;
                        float gr =  frtaud/(1.0f+frtaur);
                        float mur=0.0f;
                        
                        float dW=0.0f;
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                            float eInc =  gr*arf;
                            float  thise=   ( e_loc[f*bwidthe]+= eInc);
                            mur +=  thise*arf;
               
                            dW+= tex1Dfetch(tex_pseudoRoots,pseudoRoots_step_offset+r*factors+f) 
                                                      * tex1Dfetch(tex_variates,variatesOffset+f);
               
                        }    
                   
                   
                    
                        float change = mur+ dW+tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r+s*rates);
                        float logfr =  logSharedLoc[r*bwidthl];
                        logfr += change;
                    
                        // we have predicted logfr, now correct it 
                    
                        float frpDissed  = exp(logfr);
                        float frp = frpDissed-dis;
                    
                        float grp = frpDissed*taur/(1.0f+frp*taur);
                        float murp=0.0f;
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                            float erIncp =  grp*arf;
                            float  thise=   ( e_pred_loc[f*bwidthe]+= erIncp);
                            murp+= arf*thise;   
                         }

                        logfr += 0.5f*(murp-mur);
                    
                        logSharedLoc[r*bwidthl] = logfr;
                        rates_out_location[r*paths] = exp(logfr)-dis; 
                    
                       
                
                } // end of rates loop
            
    
            } // end of steps loop
        } // end of if pathnumber < paths
    } // end of paths per thread loop

            
} // end of function






__global__ void LMM_evolver_all_steps_pc_kernel_devicelog(int pseudoRoots_offset, // texture offset 
                                                int taus_offset,  // texture offset 
                                                int dis_offset, // texture offset
                                                int fixed_drifts_offset, // texture offset 
                                                int AOffsetPerStep,  // change in texture offset for each pseudo-root 
                                                int alive_offset, // offset for tex_alive
                                                int initialLogRatesOffset, // offset for tex_initialLogRates;
                                                int initialRatesOffset, // offset for tex_initialRates
                                                int initial_drifts_offset, // texture offset 
                                                int variates_offset, 
                                                float* rates_output, 
                                                float* log_rates_output,
                                                int factors, 
                                                int paths,
                                                int rates,
                                                int steps
                                                )
{ // start of function

    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);
 
    __shared__ float eShared[SHAREDSIZE];   

    
    float* e_loc =eShared+tx;
    float* e_pred_loc = eShared+tx+factors*bwidth;
 
    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        
        if (pathNumber < paths)
        {
            // step zero as special case

            int aliveIndex = tex1Dfetch(tex_alive, alive_offset);
            
            int pathOffset =  pathNumber;
    
            int variatesOffset =    variates_offset +pathOffset*factors;
            float* logRates_out_location = log_rates_output+pathOffset;

            float* rates_out_location = rates_output+pathOffset;

            int pseudoRootsStepOffset = pseudoRoots_offset;
            
            // note this is outside the rates loops  
            for (int f=0; f< factors; ++f)
               e_pred_loc[f*bwidth]=0;
  
            for (int r=aliveIndex; r < rates; ++r)
            {

               float logfr =    tex1Dfetch( tex_initialLogRates, initialLogRatesOffset+r);
             
               float mur=  tex1Dfetch(tex_initial_drifts,initial_drifts_offset+r);                       

               logfr +=  mur+tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r);   

               float dW=0.0f;

               for (int f=0; f < factors; ++f)
                   dW+= tex1Dfetch(tex_pseudoRoots,pseudoRootsStepOffset+r*factors+f) 
                                   * tex1Dfetch(tex_variates,variatesOffset+f);
               
               logfr += dW; 
                    
               // predicted, now correct
                    
               float frpDissed  = exp(logfr);
               float dis = tex1Dfetch(tex_displacements,dis_offset+r);
               float frp = frpDissed-dis;
               float taur = tex1Dfetch(tex_taus,taus_offset+r);                    
               float grp = frpDissed*taur/(1.0f+frp*taur);
               float murp=0.0f;
                    
               for (int f=0; f < factors; ++f)
               {
                  float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+r*factors+f);
                  float erIncp =  grp*arf;
                  float  thise=   ( e_pred_loc[f*bwidth]+= erIncp);
                  murp+= arf*thise;   
               }
               logfr += 0.5f*(murp-mur);
               logRates_out_location[r*paths]=logfr;
               
               rates_out_location[r*paths] =exp(logfr)-dis;
                
            } // end rates loop for step zero
            
            // now do the other steps
            for (int s =1; s < steps; ++s)
            {
                
                float* rates_start_step_location = rates_out_location;
                rates_out_location+= paths*rates;
                float* log_rates_start_step_location = logRates_out_location;
                logRates_out_location+= paths*rates;
                
                
                variatesOffset += factors*paths; 
                int aliveIndex = tex1Dfetch(tex_alive, alive_offset+s);
                int alivestarpaths = aliveIndex*paths;
                int pseudoRoots_step_offset = pseudoRoots_offset+s*rates*factors;
                int r=aliveIndex;
                { 
                    float taur =tex1Dfetch(tex_taus,taus_offset+r);
                    float fr = rates_start_step_location[alivestarpaths];
                    float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                    float frtaur = fr*taur;
                    float frtaud = frtaur+dis*taur;
                    float gr =  frtaud/(1.0f+frtaur);
                    float mur=0.0f;
                
                    float arf;
                    float dW=0.0f;
               
                    for (int f=0; f < factors; ++f)
                    {
                        arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float er =  gr*arf;
                        mur+= arf*er;
                        e_loc[f*bwidth] = er;
                        dW+=arf*tex1Dfetch(tex_variates, variatesOffset+f);
                    }
               
                    //          int rateOffset = alivestarpaths;
                    
                    float change = mur+ dW+tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r+s*rates);
                    float logfr =  log_rates_start_step_location[r*paths];
                    logfr += change;
                    
                    // we have predicted logfr, now correct it 
                    
                    float frpDissed  = exp(logfr);
                    float frp = frpDissed-dis;
                    
                    float grp = frpDissed*taur/(1.0f+frp*taur);
                    float murp=0.0f;
                    
                    for (int f=0; f < factors; ++f)
                    {
                        arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float erp =  grp*arf;
                        murp+= arf*erp;
                        e_pred_loc[f*bwidth] = erp;
                    }
                    logfr += 0.5f*(murp-mur);
                    
                    
                    logRates_out_location[r*paths] = logfr;
                    rates_out_location[r*paths] = exp(logfr)-dis; 
                }
                    

                    // rate aliveIndex is done, now do other rates 
                    ++r;
                    for (; r < rates; ++r)
                    {
                        float taurecip =  1.0f/tex1Dfetch(tex_taus,taus_offset+r);
                        float fr = rates_start_step_location[r*paths];
                        float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                        float gr =  (fr+dis)/(taurecip+fr);
                        float mur=0.0f;
                        
                        float dW=0.0f;
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                
                            mur +=  arf*( e_loc[f*bwidth]+= gr*arf);
               
                            dW+= tex1Dfetch(tex_pseudoRoots,pseudoRoots_step_offset+r*factors+f) 
                                                      * tex1Dfetch(tex_variates,variatesOffset+f);
               
                        }    
                   
                   
                    
                        float change = mur+ dW+tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r+s*rates);
                        float logfr =  log_rates_start_step_location[r*paths];
                        logfr += change;
                    
                        // we have predicted logfr, now correct it 
                        
                        float murp=0.0f;
                        float grp;
                    
                        {
                            float frpDissed  = exp(logfr);
                            float frp = frpDissed-dis;
                    
                            grp = frpDissed/(taurecip+frp);
                        }
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);                           
                            murp += arf* ( e_pred_loc[f*bwidth]+= grp*arf);
                         }

                        logfr += 0.5f*(murp-mur);
                    
                        logRates_out_location[r*paths] = logfr;
                        rates_out_location[r*paths] = exp(logfr)-dis; 
                    
                       
                
                } // end of rates loop
            
    
            } // end of steps loop
        } // end of if pathnumber < paths
    } // end of paths per thread loop
}// end of function



__global__ void LMM_evolver_all_steps_pc_kernel_devicelog_discounts(int pseudoRoots_offset, // texture offset 
                                                int taus_offset,  // texture offset 
                                                int dis_offset, // texture offset
                                                int fixed_drifts_offset, // texture offset 
                                                int AOffsetPerStep,  // change in texture offset for each pseudo-root 
                                                int alive_offset, // offset for tex_alive
                                                int initialLogRatesOffset, // offset for tex_initialLogRates;
                                                int initialRatesOffset, // offset for tex_initialRates
                                                int initial_drifts_offset, // texture offset 
                                                int variates_offset, 
                                                float* rates_output, 
                                                float* log_rates_output,
                                                float* discounts_output,
                                                int factors, 
                                                int paths,
                                                int rates,
                                                int steps,
                                                int pathmult,
                                                int factmult
                                                )
{ // start of function
	 extern  __shared__ float eShared[];   

    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);
 
 
    
    float* e_loc =eShared+tx;
    float* e_pred_loc = eShared+tx+factors*bwidth;
 
    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        
        if (pathNumber < paths)
        {
            // step zero as special case

            int aliveIndex = tex1Dfetch(tex_alive, alive_offset);
            
            int pathOffset =  pathNumber;
    
            int variatesOffset =   variates_offset +pathOffset*pathmult;  // variates_offset +pathOffset*factors;
            float* logRates_out_location = log_rates_output+pathOffset;

            float* rates_out_location = rates_output+pathOffset;
            float* disc_location = discounts_output+pathOffset;

            int pseudoRootsStepOffset = pseudoRoots_offset;
            
            {
            // note this is outside the rates loops  
            for (int f=0; f< factors; ++f)
               e_pred_loc[f*bwidth]=0;
               
            disc_location[aliveIndex] = 1.0f;   
            float lastdf = 1.0f;
  
            for (int r=aliveIndex; r < rates; ++r)
            {

               float logfr =    tex1Dfetch( tex_initialLogRates, initialLogRatesOffset+r);
             
               float mur=  tex1Dfetch(tex_initial_drifts,initial_drifts_offset+r);                       

               logfr +=  mur+tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r);   

               float dW=0.0f;

               for (int f=0; f < factors; ++f)
                   dW+= tex1Dfetch(tex_pseudoRoots,pseudoRootsStepOffset+r*factors+f) 
                                   * tex1Dfetch(tex_variates,variatesOffset+f*factmult);
               
               logfr += dW; 
                    
               // predicted, now correct
                    
               float frpDissed  = exp(logfr);
               float dis = tex1Dfetch(tex_displacements,dis_offset+r);
               float frp = frpDissed-dis;
               float taur = tex1Dfetch(tex_taus,taus_offset+r);                    
               float grp = frpDissed*taur/(1.0f+frp*taur);
               float murp=0.0f;
                    
               for (int f=0; f < factors; ++f)
               {
                  float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+r*factors+f);
                  float erIncp =  grp*arf;
                  float  thise=   ( e_pred_loc[f*bwidth]+= erIncp);
                  murp+= arf*thise;   
               }
               logfr += 0.5f*(murp-mur);
               logRates_out_location[r*paths]=logfr;
               float rate = exp(logfr)-dis;
               lastdf /= (1.0f+rate*taur);
               disc_location[(r+1)*paths] = lastdf;               
               rates_out_location[r*paths] =rate;
                
            } // end rates loop for step zero
            }
            
            // now do the other steps
            for (int s =1; s < steps; ++s)
            {
                
                float* rates_start_step_location = rates_out_location;
                rates_out_location+= paths*rates;
                float* log_rates_start_step_location = logRates_out_location;
                logRates_out_location+= paths*rates;
                
                disc_location += paths*(rates+1);
                
                
                variatesOffset += factors*paths; 
                int aliveIndex = tex1Dfetch(tex_alive, alive_offset+s);
                int alivestarpaths = aliveIndex*paths;
                int pseudoRoots_step_offset = pseudoRoots_offset+s*rates*factors;
                int r=aliveIndex;
                
                float lastdf= 1.0f;
                disc_location[r*paths]= lastdf; 
                
                { 
                    float taur =tex1Dfetch(tex_taus,taus_offset+r);
                    float fr = rates_start_step_location[alivestarpaths];
                    float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                    float frtaur = fr*taur;
                    float frtaud = frtaur+dis*taur;
                    float gr =  frtaud/(1.0f+frtaur);
                    float mur=0.0f;
                
                    float arf;
                    float dW=0.0f;
               
                    for (int f=0; f < factors; ++f)
                    {
                        arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float er =  gr*arf;
                        mur+= arf*er;
                        e_loc[f*bwidth] = er;
                        dW+=arf*tex1Dfetch(tex_variates, variatesOffset+f*factmult);
                    }
               
                    //          int rateOffset = alivestarpaths;
                    
                    float change = mur+ dW+tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r+s*rates);
                    float logfr =  log_rates_start_step_location[r*paths];
                    logfr += change;
                    
                    // we have predicted logfr, now correct it 
                    
                    float frpDissed  = exp(logfr);
                    float frp = frpDissed-dis;
                    
                    float grp = frpDissed*taur/(1.0f+frp*taur);
                    float murp=0.0f;
                    
                    for (int f=0; f < factors; ++f)
                    {
                        arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float erp =  grp*arf;
                        murp+= arf*erp;
                        e_pred_loc[f*bwidth] = erp;
                    }
                    logfr += 0.5f*(murp-mur);
                    
                    
                    logRates_out_location[r*paths] = logfr;
                    float rate = exp(logfr)-dis; 
                    rates_out_location[r*paths] = rate;
                    lastdf /= 1 + rate*taur;
                    disc_location[(r+1)*paths] = lastdf;
                }
                    

                    // rate aliveIndex is done, now do other rates 
                    ++r;
                    for (; r < rates; ++r)
                    {
                        float taur = tex1Dfetch(tex_taus,taus_offset+r);
                        float fr = rates_start_step_location[r*paths];
                        float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                        float gr =  (fr+dis)/(1.0f/taur+fr);
                        float mur=0.0f;
                        
                        float dW=0.0f;
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                
                            mur +=  arf*( e_loc[f*bwidth]+= gr*arf);
               
                            dW+= tex1Dfetch(tex_pseudoRoots,pseudoRoots_step_offset+r*factors+f) 
                                                      * tex1Dfetch(tex_variates,variatesOffset+f*factmult);
               
                        }    
                    
                        float change = mur+ dW+tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r+s*rates);
                        float logfr =  log_rates_start_step_location[r*paths];
                        logfr += change;
                    
                        // we have predicted logfr, now correct it 
                        
                        float murp=0.0f;
                        float grp;
                    
                        {
                            float frpDissed  = exp(logfr);
                            float frp = frpDissed-dis;
                    
                            grp = frpDissed/(1.0f/taur+frp);
                        }
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);                           
                            murp += arf* ( e_pred_loc[f*bwidth]+= grp*arf);
                         }

                        logfr += 0.5f*(murp-mur);
                    
                        logRates_out_location[r*paths] = logfr;
                        float rate =  exp(logfr)-dis; 
                        lastdf /= 1 + rate*taur;
                        disc_location[(r+1)*paths] = lastdf;
                        rates_out_location[r*paths] = rate;
                    
                } // end of rates loop
            
    
            } // end of steps loop
        } // end of if pathnumber < paths
    } // end of paths per thread loop
}// end of function

__global__ void LMM_evolver_all_steps_pc_kernel_devicelog_discounts_kepler(const float* 
																		   __restrict__ 
																		   pseudoRootsStep_ptr, 
                                                const float* 
												__restrict__ 
												taus_constptr,  
                                                const float* 
												__restrict__ 
												dis_constptr, 
                                                const float* 
												__restrict__ 
												fixed_drifts_constptr, 
                                                int AOffsetPerStep,  // change in address for each pseudo-root 
                                                const int* 
												__restrict__  
												alive_constptr,
                                                const float* 
												__restrict__  
												initialLogRates_constptr, 
                                                const float* 
												__restrict__  
												initialRates_constptr, // 
                                                const float* 
												__restrict__ 
												initial_drifts_constptr, // 
                                                const float* 
												__restrict__ 
												variates_constptr, 
                                                float* rates_output, 
                                                float* log_rates_output,
                                                float* discounts_output,
                                                int factors, 
                                                int paths,
                                                int rates,
                                                int steps,
                                                int pathmult,
                                                int factmult
                                                )
{ // start of function
	extern __shared__ float eShared[];
    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);
 
  //  __shared__ float eShared[SHAREDSIZE];   

    
    float* e_loc =eShared+tx;
    float* e_pred_loc = eShared+tx+factors*bwidth;
 
    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        
        if (pathNumber < paths)
        {
            // step zero as special case

            int aliveIndex = *alive_constptr; //tex1Dfetch(tex_alive, alive_offset);
            
            int pathOffset =  pathNumber;
    
            float* logRates_out_location = log_rates_output+pathOffset;

            float* rates_out_location = rates_output+pathOffset;
            float* disc_location = discounts_output+pathOffset;

		    int variatesOffset = pathOffset*pathmult;  // variates_offset +pathOffset*factors;
    

      //      const float* __restrict__ 
//pseudoRootsStep_ptr = pseudoRoots_constptr;
            
            {
            // note this is outside the rates loops  
            for (int f=0; f< factors; ++f)
               e_pred_loc[f*bwidth]=0;
               
            disc_location[aliveIndex] = 1.0f;   
            float lastdf = 1.0f;
  
            for (int r=aliveIndex; r < rates; ++r)
            {

               float logfr =  LDG(initialLogRates_constptr+r); // tex1Dfetch( tex_initialLogRates, initialLogRatesOffset+r);
             
               float mur= LDG(initial_drifts_constptr+r); //  tex1Dfetch(tex_initial_drifts,initial_drifts_offset+r);                       

               logfr +=  mur   +   LDG(fixed_drifts_constptr+r);   // +tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r);   

               float dW=0.0f;

               for (int f=0; f < factors; ++f)
                   dW+= pseudoRootsStep_ptr[r*factors+f]*LDG(variates_constptr+variatesOffset+f*factmult);
				   
				   // tex1Dfetch(tex_pseudoRoots,pseudoRootsStepOffset+r*factors+f) 
 //                                  * tex1Dfetch(tex_variates,variatesOffset+f*factmult);
               
               logfr += dW; 
                    
               // predicted, now correct
                    
               float frpDissed  = exp(logfr);
               float dis =LDG( dis_constptr+r); //tex1Dfetch(tex_displacements,dis_offset+r);
               float frp = frpDissed-dis;
               float taur =LDG( taus_constptr+r);  //tex1Dfetch(tex_taus,taus_offset+r);                    
               float grp = frpDissed*taur/(1.0f+frp*taur);
               float murp=0.0f;
                    
               for (int f=0; f < factors; ++f)
               {
                  float arf =pseudoRootsStep_ptr[r*factors+f];// tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+r*factors+f);
                  float erIncp =  grp*arf;
                  float  thise=   ( e_pred_loc[f*bwidth]+= erIncp);
                  murp+= arf*thise;   
               }
               logfr += 0.5f*(murp-mur);
               logRates_out_location[r*paths]=logfr;
               float rate = exp(logfr)-dis;
               lastdf /= (1.0f+rate*taur);
               disc_location[(r+1)*paths] = lastdf;               
               rates_out_location[r*paths] =rate;
                
            } // end rates loop for step zero
            }

            // now do the other steps
            for (int s =1; s < steps; ++s)
            {
                
                float* rates_start_step_location = rates_out_location;
                rates_out_location+= paths*rates;
                float* log_rates_start_step_location = logRates_out_location;
                logRates_out_location+= paths*rates;
                
                disc_location += paths*(rates+1);
                
                
                variatesOffset += factors*paths; 
                int aliveIndex =LDG(alive_constptr+s); //tex1Dfetch(tex_alive, alive_offset+s);
                int alivestarpaths = aliveIndex*paths;
                int pseudoRoots_step_offset = s*rates*factors;
                int r=aliveIndex;
                
                float lastdf= 1.0f;
                disc_location[r*paths]= lastdf; 
                
                { 
                    float taur = LDG(taus_constptr+r);                 // tex1Dfetch(tex_taus,taus_offset+r);
                    float fr = rates_start_step_location[alivestarpaths];
                    float dis =  LDG(dis_constptr+r);       //tex1Dfetch(tex_displacements,dis_offset+r);
                    float frtaur = fr*taur;
                    float frtaud = frtaur+dis*taur;
                    float gr =  frtaud/(1.0f+frtaur);
                    float mur=0.0f;
                
                    float arf;
                    float dW=0.0f;
               
                    for (int f=0; f < factors; ++f)
                    {
                        arf = LDG(pseudoRootsStep_ptr+pseudoRoots_step_offset+f+r*factors); //tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float er =  gr*arf;
                        mur+= arf*er;
                        e_loc[f*bwidth] = er;
                        dW+=arf* LDG(variates_constptr+variatesOffset+f*factmult);                //tex1Dfetch(tex_variates, variatesOffset+f*factmult);
                    }
               
                    //          int rateOffset = alivestarpaths;
                    
                    float change = mur+ dW+LDG(fixed_drifts_constptr+r+s*rates);   //tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r+s*rates);
                    float logfr =  log_rates_start_step_location[r*paths];
                    logfr += change;
                    
                    // we have predicted logfr, now correct it 
                    
                    float frpDissed  = exp(logfr);
                    float frp = frpDissed-dis;
                    
                    float grp = frpDissed*taur/(1.0f+frp*taur);
                    float murp=0.0f;
                    
                    for (int f=0; f < factors; ++f)
                    {
                        arf =  LDG(pseudoRootsStep_ptr+pseudoRoots_step_offset+f+r*factors); //tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float erp =  grp*arf;
                        murp+= arf*erp;
                        e_pred_loc[f*bwidth] = erp;
                    }
                    logfr += 0.5f*(murp-mur);
                    
                    
                    logRates_out_location[r*paths] = logfr;
                    float rate = exp(logfr)-dis; 
                    rates_out_location[r*paths] = rate;
                    lastdf /= 1 + rate*taur;
                    disc_location[(r+1)*paths] = lastdf;
                }
                    

                    // rate aliveIndex is done, now do other rates 
                    ++r;
                    for (; r < rates; ++r)
                    {
                        float taur =LDG(taus_constptr+r) ; // tex1Dfetch(tex_taus,taus_offset+r);
                        float fr = rates_start_step_location[r*paths];
                        float dis = LDG(dis_constptr+r); //tex1Dfetch(tex_displacements,dis_offset+r);
                        float gr =  (fr+dis)/(1.0f/taur+fr);
                        float mur=0.0f;
                        
                        float dW=0.0f;
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = LDG(pseudoRootsStep_ptr+pseudoRoots_step_offset+f+r*factors);// tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                
                            mur +=  arf*( e_loc[f*bwidth]+= gr*arf);
               
                            dW+= //pseudoRootsStep_ptr[pseudoRoots_step_offset+f+r*factors]
							
								arf*LDG(variates_constptr+variatesOffset+f*factmult);
								// tex1Dfetch(tex_pseudoRoots,pseudoRoots_step_offset+r*factors+f) 
                                  //                    * tex1Dfetch(tex_variates,variatesOffset+f*factmult);
               
                        }    
                    
                        float change = mur+ dW+LDG(fixed_drifts_constptr+r+s*rates);
							//tex1Dfetch(tex_fixed_drifts,fixed_drifts_offset+r+s*rates);
                        float logfr =  log_rates_start_step_location[r*paths];
                        logfr += change;
                    
                        // we have predicted logfr, now correct it 
                        
                        float murp=0.0f;
                        float grp;
                    
                        {
                            float frpDissed  = exp(logfr);
                            float frp = frpDissed-dis;
                    
                            grp = frpDissed/(1.0f/taur+frp);
                        }
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = LDG(pseudoRootsStep_ptr+pseudoRoots_step_offset+f+r*factors);//tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);                           
                            murp += arf* ( e_pred_loc[f*bwidth]+= grp*arf);
                         }

                        logfr += 0.5f*(murp-mur);
                    
                        logRates_out_location[r*paths] = logfr;
                        float rate =  exp(logfr)-dis; 
                        lastdf /= 1 + rate*taur;
                        disc_location[(r+1)*paths] = lastdf;
                        rates_out_location[r*paths] = rate;
                    
                } // end of rates loop
            
    
            } // end of steps loop
        } // end of if pathnumber < paths
    } // end of paths per thread loop
}// end of function





// put all fixed terms except pseudos into shared memory
__global__ void LMM_evolver_all_steps_pc_kernel_shared_fix_discounts(int pseudoRoots_offset, // texture offset 
                                                float* taus_global,  
                                                float* dis_global, 
                                                float* fixed_drifts_global, 
                                                int* alive_global, 
                                                float* initialLogRates_global, 
                                                float* initialRates_global, 
                                                float* initial_drifts_global, 
                                                int variates_offset, 
                                                float* rates_output, 
                                                float* log_rates_output,
                                                float* discounts_output,
                                                float* e_loc_global,
                                                float* e_pred_global,
                                                int factors, 
                                                int paths,
                                                int rates,
                                                int steps,
                                                int pathmult,
                                                int factmult
                                                )
{ // start of function

    int bx = blockIdx.x;
    
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);
 
    extern __shared__ float data_shared[];
    float* taus_s = data_shared;
    float* dis_s = taus_s+rates;
    float* fixed_drifts_s = dis_s +rates;
    float* initialLogRates_s= fixed_drifts_s+rates*steps;
    float* initialRates_s =  initialLogRates_s+ rates;
    float* initial_drifts_s = initialRates_s + rates;
    int* alive_s = reinterpret_cast<int*>(initial_drifts_s+rates); 
  
    int tx = threadIdx.x;

    while (tx < rates)
    {
        taus_s[tx] = taus_global[tx];
        dis_s[tx] = dis_global[tx];
        initialLogRates_s[tx] = initialLogRates_global[tx];
        initialRates_s[tx] = initialRates_global[tx];
        initial_drifts_s[tx] = initial_drifts_global[tx];
        tx += blockDim.x;
    }

    tx = threadIdx.x;
    while (tx < steps)
    {
        alive_s[tx] = alive_global[tx];
        tx += blockDim.x;
    }

    tx = threadIdx.x;
    while (tx < steps*rates)
    {
        fixed_drifts_s[tx] = fixed_drifts_global[tx];
        tx += blockDim.x;
    }
    tx =  threadIdx.x;
   
    __syncthreads();
    // coalesced so hopefully not too slow
      for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        
        if (pathNumber < paths)
        {
            // step zero as special case
            float* e_loc =e_loc_global+pathNumber;
            float* e_pred_loc = e_pred_global+pathNumber;
  


            int aliveIndex = *alive_s; // i.e. alive_s[0]
            
            int pathOffset =  pathNumber;
    
            int variatesOffset =   variates_offset +pathOffset*pathmult;  // variates_offset +pathOffset*factors;
            float* logRates_out_location = log_rates_output+pathOffset;

            float* rates_out_location = rates_output+pathOffset;
            float* disc_location = discounts_output+pathOffset;

            int pseudoRootsStepOffset = pseudoRoots_offset;
            
            {
            // note this is outside the rates loops  
            for (int f=0; f< factors; ++f)
               e_pred_loc[f*paths]=0;
               
            disc_location[aliveIndex] = 1.0f;   
            float lastdf = 1.0f;
  
            for (int r=aliveIndex; r < rates; ++r)
            {

               float logfr =  initialLogRates_s[r];
               float mur=  initial_drifts_s[r];
               logfr +=  mur+fixed_drifts_s[r];

               float dW=0.0f;

               for (int f=0; f < factors; ++f)
                   dW+= tex1Dfetch(tex_pseudoRoots,pseudoRootsStepOffset+r*factors+f) 
                                   * tex1Dfetch(tex_variates,variatesOffset+f*factmult);
               
               logfr += dW; 
                    
               // predicted, now correct
                    
               float frpDissed  = exp(logfr);
               float dis = dis_s[r];
               float frp = frpDissed-dis;
               float taur = taus_s[r];                   
               float grp = frpDissed*taur/(1.0f+frp*taur);
               float murp=0.0f;
                    
               for (int f=0; f < factors; ++f)
               {
                  float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+r*factors+f);
                  float erIncp =  grp*arf;
                  float  thise=   ( e_pred_loc[f*paths]+= erIncp);
                  murp+= arf*thise;   
               }
               logfr += 0.5f*(murp-mur);

			   if (logfr > upperMax)
				   logfr = upperMax;

               logRates_out_location[r*paths]=logfr;
               float rate = exp(logfr)-dis;
               lastdf /= (1.0f+rate*taur);
               disc_location[(r+1)*paths] = lastdf;               
               rates_out_location[r*paths] =rate;
                
            } // end rates loop for step zero
            }
            
            // now do the other steps
            for (int s =1; s < steps; ++s)
            {
                
                float* rates_start_step_location = rates_out_location;
                rates_out_location+= paths*rates;
                float* log_rates_start_step_location = logRates_out_location;
                logRates_out_location+= paths*rates;
                
                disc_location += paths*(rates+1);
                
                
                variatesOffset += factors*paths; 
                int aliveIndex = alive_s[s];
                int alivestarpaths = aliveIndex*paths;
                int pseudoRoots_step_offset = pseudoRoots_offset+s*rates*factors;
                int r=aliveIndex;
                
                float lastdf= 1.0f;
                disc_location[r*paths]= lastdf; 
                
                { 
                    float taur =taus_s[r];
                    float fr = rates_start_step_location[alivestarpaths];
                    float dis = dis_s[r];
                    float frtaur = fr*taur;
                    float frtaud = frtaur+dis*taur;
                    float gr =  frtaud/(1.0f+frtaur);
                    float mur=0.0f;
                
                    float arf;
                    float dW=0.0f;
               
                    for (int f=0; f < factors; ++f)
                    {
                        arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float er =  gr*arf;
                        mur+= arf*er;
                        e_loc[f*paths] = er;
                        dW+=arf*tex1Dfetch(tex_variates, variatesOffset+f*factmult);
                    }
               
      
                    
                    float change = mur+ dW+fixed_drifts_s[r+s*rates];
                    float logfr =  log_rates_start_step_location[r*paths];
                    logfr += change;
                    
                    // we have predicted logfr, now correct it 
                    
                    float frpDissed  = exp(logfr);
                    float frp = frpDissed-dis;
                    
                    float grp = frpDissed*taur/(1.0f+frp*taur);
                    float murp=0.0f;
                    
                    for (int f=0; f < factors; ++f)
                    {
                        arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                        float erp =  grp*arf;
                        murp+= arf*erp;
                        e_pred_loc[f*paths] = erp;
                    }
                    logfr += 0.5f*(murp-mur);

				    if (logfr > upperMax)
						logfr = upperMax;

                    
                    
                    logRates_out_location[r*paths] = logfr;
                    float rate = exp(logfr)-dis; 
                    rates_out_location[r*paths] = rate;
                    lastdf /= 1 + rate*taur;
                    disc_location[(r+1)*paths] = lastdf;
                }
                    

                    // rate aliveIndex is done, now do other rates 
                    ++r;
                    for (; r < rates; ++r)
                    {
                        float taur =taus_s[r];
                        float fr = rates_start_step_location[r*paths];
                        float dis = dis_s[r];
                        float gr =  (fr+dis)/(1.0f/taur+fr);
                        float mur=0.0f;
                        
                        float dW=0.0f;
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);
                
                            mur +=  arf*( e_loc[f*paths]+= gr*arf);
               
                            dW+= tex1Dfetch(tex_pseudoRoots,pseudoRoots_step_offset+r*factors+f) 
                                                      * tex1Dfetch(tex_variates,variatesOffset+f*factmult);
               
                        }    
                    
                        float change = mur+ dW+fixed_drifts_s[r+s*rates];
                        float logfr =  log_rates_start_step_location[r*paths];
                        logfr += change;
                    
                        // we have predicted logfr, now correct it 
                        
                        float murp=0.0f;
                        float grp;
                    
                        {
                            float frpDissed  = exp(logfr);
                            float frp = frpDissed-dis;
                    
                            grp = frpDissed/(1.0f/taur+frp);
                        }
                    
                        for (int f=0; f < factors; ++f)
                        {
                            float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_step_offset+f+r*factors);                           
                            murp += arf* ( e_pred_loc[f*paths]+= grp*arf);
                         }

                        logfr += 0.5f*(murp-mur);

						if (logfr > upperMax)
							logfr =upperMax;

                    
                        logRates_out_location[r*paths] = logfr;
                        float rate =  exp(logfr)-dis; 
                        lastdf /= 1 + rate*taur;
                        disc_location[(r+1)*paths] = lastdf;
                        rates_out_location[r*paths] = rate;
                    
                } // end of rates loop
            
    
            } // end of steps loop
        } // end of if pathnumber < paths
    } // end of paths per thread loop
}// end of function









extern "C"
void LMM_evolver_pc_all_gpu(  float* initial_rates_device, 
                              float* initial_log_rates_device, 
                              float* taus_device, 
                              float* variates_device,
                              float* pseudo_roots_device,
                              float* initial_drifts_device, 
                              float* fixed_drifts_device, 
                              float* displacements_device,
                              int* alive_device,
                              int paths,
                              int factors,
                              int steps, 
                              int rates, 
                              float* e_buffer_device,
                              float* e_buffer_pred_device,
                              float* evolved_rates_device, // for output
                              float* evolved_log_rates_device,
                              float* discounts_device,
                              bool sharedMemForLogs,
                              bool transposedVariates,
							  cudaStream_t streamNumber,
							  int numberThreads,
							    bool doDiscounts 
                              )
{
//	std::cout << "Ns " << NS << " Stream number: " << streamNumber << "\n";

    int threadsperblock = numberThreads>0 ? numberThreads : 128;
  
    
    int sharedMemPerThread = 2*factors*sizeof(float);
    sharedMemPerThread += sharedMemForLogs && !doDiscounts ? rates : 0;

	int device;
	cudaDeviceProp deviceproperty;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceproperty, device);          
	int maxShared = static_cast<int>(deviceproperty.sharedMemPerBlock);

	int maxBlocks =static_cast<int>(deviceproperty.maxGridSize[0]);
 
    
    int sharedMemNeeded = (threadsperblock+extraOff)*sharedMemPerThread;
  //  std::cout << " sharedmem per thread " << sharedMemPerThread << "\n";
  // std::cout << " maxShared " << maxShared << "\n";
 // std::cout << " sharedMemNeeded " << sharedMemNeeded << "\n";
   
    while (sharedMemNeeded > maxShared)
    {
        threadsperblock/=2;
        sharedMemNeeded = (threadsperblock+extraOff)*sharedMemPerThread;
    }
    
  // std::cout << " threads per block " << threadsperblock << "\n";

 //   std::cout << " use of shared memory for log storage :" << sharedMemForLogs << "\n";
    
  //  std::cout << " doing discounts :" << doDiscounts << "\n";
    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;
    
    if (dimGrid.x > static_cast<unsigned int>(maxBlocks)) 
        dimGrid.x=maxBlocks;

  //  dimGrid.x =2400;

 //   dimGrid.x =3000;
  
 //   std::cout << "  blocks " << dimGrid.x << "\n";



    // Fix the number of threads
    dimBlock.x = threadsperblock;
      
    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

    Timer h1;

   CUT_CHECK_ERR("LMM_evolver_pc_all_gpu execution failed before entering kernel\n");
   
     
        int offset=0;
        int Asize = rates*factors;
        int factMult =1;
        int pathMult= factors;

        if (transposedVariates)
            if (doDiscounts)
            {
                factMult = paths;
                pathMult=1;

            }
            else
                std::cout << " transposed data option ignored since not provided for non-discounts engine.";
   
   // set up textures


    
               // allocate array and copy image data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

        // set texture parameters
         tex_variates.addressMode[0] = cudaAddressModeWrap;
         tex_variates.addressMode[1] = cudaAddressModeWrap;
         tex_variates.filterMode = cudaFilterModeLinear;
         tex_variates.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_variates, variates_device, channelDesc);
          

        // set texture parameters
         tex_initialRates.addressMode[0] = cudaAddressModeWrap;
         tex_initialRates.addressMode[1] = cudaAddressModeWrap;
         tex_initialRates.filterMode = cudaFilterModeLinear;
         tex_initialRates.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_initialRates, initial_rates_device, channelDesc);
         
         // set texture parameters
         tex_initialLogRates.addressMode[0] = cudaAddressModeWrap;
         tex_initialLogRates.addressMode[1] = cudaAddressModeWrap;
         tex_initialLogRates.filterMode = cudaFilterModeLinear;
         tex_initialLogRates.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_initialLogRates, initial_log_rates_device, channelDesc);
    
    
          // set texture parameters
         tex_taus.addressMode[0] = cudaAddressModeWrap;
         tex_taus.addressMode[1] = cudaAddressModeWrap;
         tex_taus.filterMode = cudaFilterModeLinear;
         tex_taus.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_taus, taus_device, channelDesc);
         
             // set texture parameters
         tex_pseudoRoots.addressMode[0] = cudaAddressModeWrap;
         tex_pseudoRoots.addressMode[1] = cudaAddressModeWrap;
         tex_pseudoRoots.filterMode = cudaFilterModeLinear;
         tex_pseudoRoots.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_pseudoRoots, pseudo_roots_device, channelDesc);
         
         // set texture parameters
         tex_initial_drifts.addressMode[0] = cudaAddressModeWrap;
         tex_initial_drifts.addressMode[1] = cudaAddressModeWrap;
         tex_initial_drifts.filterMode = cudaFilterModeLinear;
         tex_initial_drifts.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_initial_drifts, initial_drifts_device, channelDesc);
           // set texture parameters
         tex_fixed_drifts.addressMode[0] = cudaAddressModeWrap;
         tex_fixed_drifts.addressMode[1] = cudaAddressModeWrap;
         tex_fixed_drifts.filterMode = cudaFilterModeLinear;
         tex_fixed_drifts.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_fixed_drifts, fixed_drifts_device, channelDesc);
         
               // set texture parameters
         tex_displacements.addressMode[0] = cudaAddressModeWrap;
         tex_displacements.addressMode[1] = cudaAddressModeWrap;
         tex_displacements.filterMode = cudaFilterModeLinear;
         tex_displacements.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_displacements, displacements_device, channelDesc);

   
    
         tex_alive.addressMode[0] = cudaAddressModeWrap;
         tex_alive.addressMode[1] = cudaAddressModeWrap;
         tex_alive.filterMode = cudaFilterModeLinear;
         tex_alive.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_alive, alive_device, channelDesc);

    
         
         // textures are now set up
                
        CUT_CHECK_ERR("LMM_evolver_pc_all_kernel textures execution failed before entering kernel\n");
   
   
          
        if (doDiscounts)
              LMM_evolver_all_steps_pc_kernel_devicelog_discounts<<<dimGrid , dimBlock , sharedMemNeeded, streamNumber>>>(offset, // pseudos texture offset 
                                                                      offset,  // taus texture offset 
                                                                      offset,  //dis  texture offset 
                                                                      offset, // fixed drift texture offset
                                                                      Asize, // A offset per step
                                                                      offset, // texture offset for alive index
                                                                      offset, // offset for initial log rates 
                                                                      offset, // offset for initial rates
                                                                      offset, // offset for initial drifts
                                                                      offset, // offset for variates
                                                                      evolved_rates_device, // for output
                                                                      evolved_log_rates_device, // for output
                                                                      discounts_device,
                                                                      factors, 
                                                                      paths,
                                                                      rates,
                                                                      steps,
                                                                      pathMult,
                                                                      factMult
                                                                      );
    else
            if (sharedMemForLogs)
                 LMM_evolver_all_steps_pc_kernel<<<dimGrid , dimBlock, sharedMemNeeded, streamNumber >>>(offset, // pseudos texture offset 
                                                                      offset,  // taus texture offset 
                                                                      offset,  //dis  texture offset 
                                                                      offset, // fixed drift texture offset
                                                                      Asize, // A offset per step
                                                                      offset, // texture offset for alive index
                                                                      offset, // offset for initial log rates 
                                                                      offset, // offset for initial rates
                                                                      offset, // offset for initial drifts
                                                                      offset, // offset for variates
                                                                      evolved_rates_device, // for output
                                                                      factors, 
                                                                      paths,
                                                                      rates,
                                                                      steps
                                                                      );
             else
                 LMM_evolver_all_steps_pc_kernel_devicelog<<<dimGrid , dimBlock, sharedMemNeeded, streamNumber >>>(offset, // pseudos texture offset 
                                                                      offset,  // taus texture offset 
                                                                      offset,  //dis  texture offset 
                                                                      offset, // fixed drift texture offset
                                                                      Asize, // A offset per step
                                                                      offset, // texture offset for alive index
                                                                      offset, // offset for initial log rates 
                                                                      offset, // offset for initial rates
                                                                      offset, // offset for initial drifts
                                                                      offset, // offset for variates
                                                                      evolved_rates_device, // for output
                                                                      evolved_log_rates_device, // for output
                                                                      factors, 
                                                                      paths,
                                                                      rates,
                                                                      steps
                                                                      );
   
        cutilSafeCall(cudaThreadSynchronize());         
   
         CUT_CHECK_ERR("LMM_evolver_all_pc_kernel execution failed after entering kernel\n");     
    
    
         cudaUnbindTexture(tex_variates);
         cudaUnbindTexture(tex_alive);
         cudaUnbindTexture(tex_fixed_drifts);
         cudaUnbindTexture(tex_pseudoRoots);
         cudaUnbindTexture(tex_taus);
         cudaUnbindTexture(tex_displacements);
         cudaUnbindTexture( tex_initialLogRates);
         cudaUnbindTexture(tex_initialRates);
         cudaUnbindTexture(tex_initial_drifts);

  //         std::cout << " time taken for LMM evolution : " 
   //         << h1.timePassed() << std::endl;
         
     

}

void LMM_evolver_pc_all_gpu_kepler(  float* initial_rates_global, 
                              float* initial_log_rates_global, 
                              float* taus_global, 
                              float* variates_global,
                              float* pseudo_roots_global,
                              float* initial_drifts_global, 
                              float* fixed_drifts_global, 
                              float* displacements_global,
                              int* alive_global,
                              int paths,
                              int factors,
                              int steps, 
                              int rates, 
                              float* e_buffer_global,
                              float* e_buffer_pred_global,
                              float* evolved_rates_global, // for output
                              float* evolved_log_rates_global,
                              float* discounts_global,
                              bool transposedVariates,
							  int threadsperblock,
							  cudaStream_t streamNumber
                              )
{
//	std::cout << "Ns " << NS << " Stream number: " << streamNumber << "\n";

	if (threadsperblock==0)
     threadsperblock =512;
    const int maxBlocks = 1024;
    
    int sharedMemPerThread = 8*factors;
//    sharedMemPerThread += sharedMemForLogs && !doDiscounts ? rates : 0;

 //     std::cout << " sharedmem per thread " << sharedMemPerThread << "\n";
	int device;
	 cudaDeviceProp deviceproperty;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceproperty, device);          
	int maxShared =static_cast<int>( deviceproperty.sharedMemPerBlock);
    
    int sharedMemNeeded = (threadsperblock+extraOff)*sharedMemPerThread;
    
    while (sharedMemNeeded > maxShared)
    {
        threadsperblock/=2;
        sharedMemNeeded = (threadsperblock+extraOff)*sharedMemPerThread;
    }
    
    std::cout << " threads per block " << threadsperblock << "\n";

 //   std::cout << " use of shared memory for log storage :" << sharedMemForLogs << "\n";
    
  //  std::cout << " doing discounts :" << doDiscounts << "\n";
    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;
    
    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

   // dimGrid.x =2400;

 //   dimGrid.x =3000;
  
  //  std::cout << "  blocks " << dimGrid.x << "\n";



    // Fix the number of threads
    dimBlock.x = threadsperblock;
    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

   CUT_CHECK_ERR("LMM_evolver_pc_all_gpu_kepler execution failed before entering kernel\n");
   
     
     
   int Asize = rates*factors;
   
   int factMult =1;
   
   int pathMult= factors;

   
   if (transposedVariates)
   {                
			factMult = paths;
            pathMult=1;
   
   }
        

        LMM_evolver_all_steps_pc_kernel_devicelog_discounts_kepler<<<dimGrid , dimBlock , sharedMemNeeded, streamNumber>>>(pseudo_roots_global, // pseudos texture offset 
                                                                      taus_global,  // taus texture offset 
                                                                      displacements_global,  //dis  texture offset 
                                                                      fixed_drifts_global, // fixed drift texture offset
                                                                      Asize, // A offset per step
                                                                      alive_global, // texture offset for alive index
                                                                      initial_log_rates_global, // offset for initial log rates 
                                                                      initial_rates_global, // offset for initial rates
                                                                      initial_drifts_global, // offset for initial drifts
                                                                      variates_global, // offset for variates
                                                                      evolved_rates_global, // for output
                                                                      evolved_log_rates_global, // for output
                                                                      discounts_global,
                                                                      factors, 
                                                                      paths,
                                                                      rates,
                                                                      steps,
                                                                      pathMult,
                                                                      factMult
                                                                      );
        cutilSafeCall(cudaThreadSynchronize());         
   
         CUT_CHECK_ERR("LMM_evolver_all_pc_kernel_kepler execution failed after entering kernel\n");     
    
     

}

extern "C"
void LMM_evolver_pc_all_fermi_gpu(  float* initial_rates_global, 
                              float* initial_log_rates_global, 
                              float* taus_global, 
                              float* variates_global,
                              float* pseudo_roots_global,
                              float* initial_drifts_global, 
                              float* fixed_drifts_global, 
                              float* displacements_global,
                              int* alive_global,
                              int paths,
                              int factors,
                              int steps, 
                              int rates, 
                              float* e_buffer_global,
                              float* e_buffer_pred_global,
                              float* evolved_rates_global, // for output
                              float* evolved_log_rates_global,
                              float* discounts_global,
                              bool transposedVariates,
							  cudaStream_t streamNumber
                              )
{
	std::cout <<  " Stream number: " << streamNumber << "\n";

    int threadsperblock =256;
    const int maxBlocks = 1024;
    
    std::cout << " threads per block " << threadsperblock << "\n";

    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;
    
    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    dimGrid.x =2400;

 //   dimGrid.x =3000;
  
//    std::cout << "  blocks " << dimGrid.x << "\n";



    // Fix the number of threads
    dimBlock.x = threadsperblock;
    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
 
    Timer h1;

   CUT_CHECK_ERR("LMM_evolver_pc_all_fermi_gpu execution failed before entering kernel\n");
   
   
   // set up textures


    
               // allocate array and copy image data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

        // set texture parameters
         tex_variates.addressMode[0] = cudaAddressModeWrap;
         tex_variates.addressMode[1] = cudaAddressModeWrap;
         tex_variates.filterMode = cudaFilterModeLinear;
         tex_variates.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_variates, variates_global, channelDesc);
          

             
       
       
             // set texture parameters
         tex_pseudoRoots.addressMode[0] = cudaAddressModeWrap;
         tex_pseudoRoots.addressMode[1] = cudaAddressModeWrap;
         tex_pseudoRoots.filterMode = cudaFilterModeLinear;
         tex_pseudoRoots.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_pseudoRoots, pseudo_roots_global, channelDesc);
         
           
   
   
    
     
         
         // textures are now set up
                
        CUT_CHECK_ERR("LMM_evolver_pc_all_fermi_gpu  execution failed before entering kernel\n");
   
   
       
        int offset=0;
       
        int factMult =1;
        int pathMult= factors;

        if (transposedVariates)
            {
                factMult = paths;
                pathMult=1;

            }

        int sharedsize = (5+steps)*rates*sizeof(float)+ 1*steps*sizeof(int);
   
        LMM_evolver_all_steps_pc_kernel_shared_fix_discounts<<<dimGrid , dimBlock ,sharedsize, streamNumber>>>(offset, // pseudos texture offset 
                                                                      taus_global,  // taus
                                                                      displacements_global,  //dis  
                                                                      fixed_drifts_global, // fixed drift
                                                                      alive_global, //  alive index
                                                                      initial_log_rates_global, //  initial log rates 
                                                                      initial_rates_global, // r initial rates
                                                                      initial_drifts_global, //  initial drifts
                                                                      offset, //  variates
                                                                      evolved_rates_global, // for output
                                                                      evolved_log_rates_global, // for output
                                                                      discounts_global,
                                                                      e_buffer_global,
                                                                      e_buffer_pred_global,
                                                                      factors, 
                                                                      paths,
                                                                      rates,
                                                                      steps,
                                                                      pathMult,
                                                                      factMult
                                                                      );
 
        cutilSafeCall(cudaThreadSynchronize());         
   
         CUT_CHECK_ERR("LMM_evolver_pc_all_fermi_gpu execution failed after entering kernel\n");     
    
    
         cudaUnbindTexture(tex_variates);
         cudaUnbindTexture(tex_pseudoRoots);
         
         
            std::cout << " time taken for LMM evolution : " 
            << h1.timePassed() << std::endl;
         
     

}
