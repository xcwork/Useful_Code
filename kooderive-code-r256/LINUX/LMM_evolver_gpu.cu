//
//
//                             LMM_Evolver_gpu
//
//
//

// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#include "LMM_evolver_gpu.h"
#include "cudaMacros.h"
#include <cuda_runtime.h>
#include <cutil.h>

#include <cutil_inline.h>
#include <iostream>
#include <gold/Errors.h>

#define logCeiling  log(10.0f)


texture<float, 1, cudaReadModeElementType> tex_pseudoRoots;
texture<float, 1, cudaReadModeElementType> tex_taus;
texture<float, 1, cudaReadModeElementType> tex_displacements;

// for step 0 only
texture<float, 1, cudaReadModeElementType> tex_initialLogRates;
texture<float, 1, cudaReadModeElementType> tex_initialRates;
texture<float, 1, cudaReadModeElementType> tex_drifts;

// for discounts kernel
texture<float, 1, cudaReadModeElementType> tex_evolvedRates;
texture<int, 1, cudaReadModeElementType> tex_alive;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAXESIZE 1024



__global__ void LMM_evolver_noninitial_shared_e_step_pc_kernel(int pseudoRoots_offset, // texture offset 
                                                               int taus_offset,  // texture offset 
                                                               int dis_offset, // texture offset
                                                               int aliveIndex, 
                                                               float* logRates_start_step,
                                                               float* rates_start_step,
                                                               float* logRates_end_step,
                                                               float* rates_end_step,
                                                               float* correlatedBrownianIncrements,
                                                               int factors, 
                                                               int paths,
                                                               int rates,
                                                               int s // step number
                                                               )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);

    __shared__ float eShared[MAXESIZE];   

    float* e_loc =eShared+tx;
    float* e_pred_loc = eShared+tx+factors*bwidth;

    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        int alivestarpaths = aliveIndex*paths;

        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;
            float* incrementsLocation =     correlatedBrownianIncrements +pathOffset;
            float* logRates_start_step_location = logRates_start_step+pathOffset;
            float* rates_start_step_location = rates_start_step+pathOffset;
            float* logRates_end_step_location = logRates_end_step+pathOffset;
            float* rates_end_step_location = rates_end_step+pathOffset;

            // do rate aliveIndex as special case 
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

                for (int f=0; f < factors; ++f)
                {
                    arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float er =  gr*arf;
                    mur+= arf*er;
                    e_loc[f*bwidth] = er;
                }

                int rateOffset = alivestarpaths;

                float change = mur+ incrementsLocation[rateOffset];
                float logfr =  logRates_start_step_location[r*paths];
                logfr += change;

                // we have predicted logfr, now correct it 

                float frpDissed  = exp(logfr);
                float frp = frpDissed-dis;

                float grp = frpDissed*taur/(1.0f+frp*taur);
                float murp=0.0f;

                for (int f=0; f < factors; ++f)
                {
                    arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float erp =  grp*arf;
                    murp+= arf*erp;
                    e_pred_loc[f*bwidth] = erp;
                }
                logfr += 0.5f*(murp-mur);



                logRates_end_step_location[rateOffset] = logfr;

                rates_end_step_location[rateOffset] =exp(logfr)-dis;

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

                for (int f=0; f < factors; ++f)
                {
                    float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float eInc =  gr*arf;
                    float  thise=   ( e_loc[f*bwidth]+= eInc);
                    mur +=  thise*arf;
                }    

                int rateOffset = r*paths;

                float change = mur+ incrementsLocation[rateOffset];
                float logfr =  logRates_start_step_location[r*paths];
                logfr += change;

                // we have predicted logfr, now correct it 

                float frpDissed  = exp(logfr);
                float frp = frpDissed-dis;

                float grp = frpDissed*taur/(1.0f+frp*taur);
                float murp=0.0f;

                for (int f=0; f < factors; ++f)
                {
                    float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float erIncp =  grp*arf;
                    float  thise=   ( e_pred_loc[f*bwidth]+= erIncp);
                    murp+= arf*thise;   
                }
                logfr += 0.5f*(murp-mur);

                logRates_end_step_location[rateOffset] = logfr;

                rates_end_step_location[rateOffset] =  exp(logfr)-dis;

            }  

        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




__global__ void LMM_evolver_noninitial_step_pc_kernel(int pseudoRoots_offset, // texture offset 
                                                      int taus_offset,  // texture offset 
                                                      int dis_offset, // texture offset
                                                      int aliveIndex, 
                                                      float* logRates_start_step,
                                                      float* rates_start_step,
                                                      float* logRates_end_step,
                                                      float* rates_end_step,
                                                      float* correlatedBrownianIncrements,
                                                      float* evalues, 
                                                      float*  evaluesPredicted,
                                                      int factors, 
                                                      int paths,
                                                      int rates,
                                                      int s // step number
                                                      )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;




    int pathsPerThread = 1 + (( paths -1)/width);





    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        int alivestarpaths = aliveIndex*paths;

        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;
            float* incrementsLocation =     correlatedBrownianIncrements +pathOffset;
            float* logRates_start_step_location = logRates_start_step+pathOffset;
            float* rates_start_step_location = rates_start_step+pathOffset;
            float* logRates_end_step_location = logRates_end_step+pathOffset;
            float* rates_end_step_location = rates_end_step+pathOffset;
            float* e_loc = evalues+pathOffset;
            float* e_pred_loc = evaluesPredicted+pathOffset;



            // do rate aliveIndex as special case 
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

                for (int f=0; f < factors; ++f)
                {
                    arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float er =  gr*arf;
                    mur+= arf*er;
                    e_loc[f*paths] = er;
                }

                int rateOffset = alivestarpaths;

                float change = mur+ incrementsLocation[rateOffset];
                float logfr =  logRates_start_step_location[r*paths];
                logfr += change;

                // we have predicted logfr, now correct it 

                float frpDissed  = exp(logfr);
                float frp = frpDissed-dis;

                float grp = frpDissed*taur/(1.0f+frp*taur);
                float murp=0.0f;

                for (int f=0; f < factors; ++f)
                {
                    arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float erp =  grp*arf;
                    murp+= arf*erp;
                    e_pred_loc[f*paths] = erp;
                }
                logfr += 0.5f*(murp-mur);



                logRates_end_step_location[rateOffset] = logfr;

                rates_end_step_location[rateOffset] = exp(logfr)-dis;

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

                for (int f=0; f < factors; ++f)
                {
                    float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float eInc =  gr*arf;
                    float  thise=   ( e_loc[f*paths]+= eInc);
                    mur +=  thise*arf;
                }    

                int rateOffset = r*paths;

                float change = mur+ incrementsLocation[rateOffset];
                float logfr =  logRates_start_step_location[r*paths];
                logfr += change;

                // we have predicted logfr, now correct it 

                float frpDissed  = exp(logfr);
                float frp = frpDissed-dis;

                float grp = frpDissed*taur/(1.0f+frp*taur);
                float murp=0.0f;

                for (int f=0; f < factors; ++f)
                {
                    float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float erIncp =  grp*arf;
                    float  thise=   ( e_pred_loc[f*paths]+= erIncp);
                    murp+= arf*thise;   
                }
                logfr += 0.5f*(murp-mur);

                logRates_end_step_location[rateOffset] = logfr;

                rates_end_step_location[rateOffset] = exp(logfr)-dis;

            }  

        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAXMATRIXSIZE 1024

__global__ void LMM_evolver_noninitial_step_shared_pc_kernel(float* pseudoRoots_global, // texture offset 
                                                             int taus_offset,  // texture offset 
                                                             int dis_offset, // texture offset
                                                             int aliveIndex, 
                                                             float* logRates_start_step,
                                                             float* rates_start_step,
                                                             float* logRates_end_step,
                                                             float* rates_end_step,
                                                             float* correlatedBrownianIncrements,
                                                             float* evalues, 
                                                             float*  evaluesPredicted,
                                                             int factors, 
                                                             int paths,
                                                             int rates,
                                                             int s // step number
                                                             )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);

    __shared__ float pseudoShared[MAXMATRIXSIZE];

    //copy pseudoroot to shared memory

    int matrixMemSize = factors*rates;

    int loc = tx;

    while (loc < matrixMemSize)
    {
        pseudoShared[loc] = pseudoRoots_global[tx]; // tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+loc);
        loc += bwidth;
    }


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;
            float* incrementsLocation =     correlatedBrownianIncrements +pathOffset;
            float* logRates_start_step_location = logRates_start_step+pathOffset;
            float* rates_start_step_location = rates_start_step+pathOffset;
            float* logRates_end_step_location = logRates_end_step+pathOffset;
            float* rates_end_step_location = rates_end_step+pathOffset;
            float* e_loc = evalues+pathOffset;
            float* e_pred_loc = evaluesPredicted+pathOffset;



            // do rate aliveIndex as special case 
            int r=aliveIndex;
            { 
                float taur = tex1Dfetch(tex_taus,taus_offset+r);
                float fr = rates_start_step_location[r*paths];
                float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                float frtaur = fr*taur;
                float frtaud = frtaur+dis*taur;
                float gr =  frtaud/(1.0f+frtaur);
                float mur=0.0f;

                float arf;

                for (int f=0; f < factors; ++f)
                {
                    arf = pseudoShared[f+r*factors]; // tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float er =  gr*arf;
                    mur+= arf*er;
                    e_loc[f*paths] = er;
                }

                int rateOffset = r*paths;

                float change = mur+ incrementsLocation[rateOffset];
                float logfr =  logRates_start_step_location[r*paths];
                logfr += change;

                // we have predicted logfr, now correct it 

                float frpDissed  = exp(logfr);
                float frp = frpDissed-dis;

                float grp = frpDissed*taur/(1.0f+frp*taur);
                float murp=0.0f;

                for (int f=0; f < factors; ++f)
                {
                    arf =pseudoShared[f+r*factors]; // tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float erp =  grp*arf;
                    murp+= arf*erp;
                    e_pred_loc[f*paths] = erp;
                }
                logfr += 0.5f*(murp-mur);



                logRates_end_step_location[rateOffset] = logfr;

                rates_end_step_location[rateOffset] =exp(logfr)-dis;

            }
            // rate aliveIndex is done, now do other rates 
            ++r;
            for (; r < rates; ++r)
            {
                float taur = tex1Dfetch(tex_taus,taus_offset+r);
                float fr = rates_start_step_location[r*paths];
                float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                float frtaur = fr*taur;
                float frtaud = frtaur+dis*taur;
                float gr =  frtaud/(1.0f+frtaur);
                float mur=0.0f;

                for (int f=0; f < factors; ++f)
                {
                    float arf =pseudoShared[f+r*factors]; // tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float eInc =  gr*arf;
                    float  thise=   ( e_loc[f*paths]+= eInc);
                    mur +=  thise*arf;
                }    

                int rateOffset = r*paths;

                float change = mur+ incrementsLocation[rateOffset];
                float logfr =  logRates_start_step_location[r*paths];
                logfr += change;

                // we have predicted logfr, now correct it 

                float frpDissed  = exp(logfr);
                float frp = frpDissed-dis;

                float grp = frpDissed*taur/(1.0f+frp*taur);
                float murp=0.0f;

                for (int f=0; f < factors; ++f)
                {
                    float arf =pseudoShared[f+r*factors]; // tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float erIncp =  grp*arf;
                    float  thise=   ( e_pred_loc[f*paths]+= erIncp);
                    murp+= arf*thise;   
                }
                logfr += 0.5f*(murp-mur);

                logRates_end_step_location[rateOffset] = logfr;

                rates_end_step_location[rateOffset] =exp(logfr)-dis;

            }  

        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void LMM_evolver_initial_step_pc_kernel(int pseudoRoots_offset, // texture offset 
                                                   int taus_offset,  // texture offset 
                                                   int dis_offset, // texture offset
                                                   int drifts_offset, // texture offset
                                                   int initialLogRates_offset, // texture offset
                                                   float* logRates_end_step,
                                                   float* rates_end_step,
                                                   float* correlatedBrownianIncrements,
                                                   float*  evaluesPredicted,
                                                   int factors, 
                                                   int paths,
                                                   int rates
                                                   )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);

    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;

            float* incrementsLocation =     correlatedBrownianIncrements +pathOffset;
            float* logRates_end_step_location = logRates_end_step+pathOffset;
            float* rates_end_step_location = rates_end_step+pathOffset;
            float* e_pred_loc = evaluesPredicted+pathOffset;

            for (int f=0; f< factors; ++f)
                e_pred_loc[f*paths]=0;

            for (int r=0; r < rates; ++r)
            {
                int rateOffset = r*paths;
                float logfr = tex1Dfetch(tex_initialLogRates,initialLogRates_offset+r);

                float mur=  tex1Dfetch(tex_drifts,drifts_offset+r);                       
                float change = mur+ incrementsLocation[rateOffset];
                logfr += change;

                // predicted, now correct

                float frpDissed  = exp(logfr);
                float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                float frp = frpDissed-dis;
                float taur = tex1Dfetch(tex_taus,taus_offset+r);

                float grp = frpDissed*taur/(1.0f+frp*taur);
                float murp=0.0f;

                for (int f=0; f < factors; ++f)
                {
                    float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float erIncp =  grp*arf;
                    float  thise=   ( e_pred_loc[f*paths]+= erIncp);
                    murp+= arf*thise;   
                }
                logfr += 0.5f*(murp-mur);

                logRates_end_step_location[rateOffset] = logfr;
                rates_end_step_location[rateOffset] =exp(logfr)-dis;
            }  
        }
    }
}


extern "C"
void LMM_evolver_pc_gpu(  float* initial_rates_device, 
                        float* initial_log_rates_device, 
                        float* taus_device, 
                        float* correlatedBrownianIncrements_device,
                        float* pseudo_roots_device,
                        float* initial_drifts_device, 
                        float* displacements_device,
                        const std::vector<int>& aliveIndices, 
                        int paths,
                        int factors,
                        int steps, 
                        int rates, 
                        float* e_buffer_device,
                        float* e_buffer_pred_device,
                        float* evolved_rates_device, // for output
                        float* evolved_log_rates_device,  // for output 
                        bool useShared
                        )
{

    const int threadsperblock =64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;

   Timer h1;

    CUT_CHECK_ERR("LMM_evolver_pc_initial_step_gpu execution failed before entering kernel\n");


    // set up textures


    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

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
    tex_drifts.addressMode[0] = cudaAddressModeWrap;
    tex_drifts.addressMode[1] = cudaAddressModeWrap;
    tex_drifts.filterMode = cudaFilterModeLinear;
    tex_drifts.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_drifts, initial_drifts_device, channelDesc);

    // set texture parameters
    tex_displacements.addressMode[0] = cudaAddressModeWrap;
    tex_displacements.addressMode[1] = cudaAddressModeWrap;
    tex_displacements.filterMode = cudaFilterModeLinear;
    tex_displacements.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_displacements, displacements_device, channelDesc);


    // textures are now set up

    CUT_CHECK_ERR("LMM_evolver_pc_initial_step_kernel textures execution failed before entering kernel\n");



    int offset=0;

    LMM_evolver_initial_step_pc_kernel<<<dimGrid , dimBlock >>>(offset, // texture offset 
        offset,  // texture offset 
        offset,  // texture offset 
        offset, // texture offset
        offset, // texture offset
        evolved_log_rates_device,
        evolved_rates_device,
        correlatedBrownianIncrements_device,
        e_buffer_device,
        factors, 
        paths,
        rates
        );



    cutilSafeCall(cudaThreadSynchronize());         

    CUT_CHECK_ERR("LMM_evolver_initial_step_pc_kernel execution failed after entering kernel\n");     

    int layerSize = rates*paths;



    for (int s=1; s < steps; ++s)
    {
        CUT_CHECK_ERR("LMM_evolver_noninitial_step_pc_kernel execution failed before entering kernel\n");     

        int pseudoRoots_offset = s*rates*factors;
        int taus_offset =0;
        int dis_offset =0;
        int aliveIndex =aliveIndices[s] ;


        if (factors*threadsperblock*2 <=MAXESIZE)
        {


            LMM_evolver_noninitial_shared_e_step_pc_kernel<<<dimGrid , dimBlock >>>(pseudoRoots_offset, // texture offset 
                taus_offset,  // texture offset 
                dis_offset, // displacement offset
                aliveIndex, 
                evolved_log_rates_device+layerSize*(s-1),
                evolved_rates_device+layerSize*(s-1),
                evolved_log_rates_device+layerSize*s,
                evolved_rates_device+layerSize*s,
                correlatedBrownianIncrements_device+layerSize*s,
                factors, 
                paths,
                rates,
                s
                );
        }
        else    
            if (factors * rates<= MAXMATRIXSIZE && useShared)
                LMM_evolver_noninitial_step_shared_pc_kernel<<<dimGrid , dimBlock >>>(pseudo_roots_device+pseudoRoots_offset,
                taus_offset,  // texture offset 
                dis_offset, // displacement offset
                aliveIndex, 
                evolved_log_rates_device+layerSize*(s-1),
                evolved_rates_device+layerSize*(s-1),
                evolved_log_rates_device+layerSize*s,
                evolved_rates_device+layerSize*s,
                correlatedBrownianIncrements_device+layerSize*s,
                e_buffer_device,
                e_buffer_pred_device,
                factors, 
                paths,
                rates,
                s
                );
            else             
                LMM_evolver_noninitial_step_pc_kernel<<<dimGrid , dimBlock >>>(pseudoRoots_offset, // texture offset 
                taus_offset,  // texture offset 
                dis_offset, // displacement offset
                aliveIndex, 
                evolved_log_rates_device+layerSize*(s-1),
                evolved_rates_device+layerSize*(s-1),
                evolved_log_rates_device+layerSize*s,
                evolved_rates_device+layerSize*s,
                correlatedBrownianIncrements_device+layerSize*s,
                e_buffer_device,
                e_buffer_pred_device,
                factors, 
                paths,
                rates,
                s
                );

        cutilSafeCall(cudaThreadSynchronize());                                                 

        CUT_CHECK_ERR("LMM_evolver_euler_noninitial_step_kernel execution failed after entering kernel\n");     

    }

    cudaUnbindTexture(tex_initialRates);
    cudaUnbindTexture(tex_initialLogRates);
    cudaUnbindTexture(tex_taus);
    cudaUnbindTexture(tex_pseudoRoots);
    cudaUnbindTexture(tex_drifts);


   
    double time = h1.timePassed();
    std::cout << " time taken for LMM evolution : " 
        << time  << std::endl;



}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void LMM_evolver_euler_noninitial_step_kernel(int pseudoRoots_offset, // texture offset 
                                                         int taus_offset,  // texture offset 
                                                         int dis_offset, // texture offset
                                                         int aliveIndex, 
                                                         float* logRates_start_step,
                                                         float* rates_start_step,
                                                         float* logRates_end_step,
                                                         float* rates_end_step,
                                                         float* correlatedBrownianIncrements,
                                                         float* evalues, 
                                                         int factors, 
                                                         int paths,
                                                         int rates,
                                                         int s // step number
                                                         )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;
            float* incrementsLocation =     correlatedBrownianIncrements +pathOffset;
            float* logRates_start_step_location = logRates_start_step+pathOffset;
            float* rates_start_step_location = rates_start_step+pathOffset;
            float* logRates_end_step_location = logRates_end_step+pathOffset;
            float* rates_end_step_location = rates_end_step+pathOffset;
            float* e_loc = evalues+pathOffset;



            // do rate aliveIndex as special case 
            int r=aliveIndex;
            { 
                float taur = tex1Dfetch(tex_taus,taus_offset+r);
                float fr = rates_start_step_location[r*paths];
                float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                float frtaur = fr*taur;
                float frtaud = frtaur+dis*taur;
                float gr =  frtaud/(1.0f+frtaur);
                float mur=0.0f;

                float arf;

                for (int f=0; f < factors; ++f)
                {
                    arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float er =  gr*arf;
                    mur+= arf*er;
                    e_loc[f*paths] = er;
                }

                int rateOffset = r*paths;

                float change = mur+ incrementsLocation[rateOffset];
                float logfr =  logRates_start_step_location[r*paths];
                logfr += change;

				if (logfr > logCeiling)
					logfr= logCeiling;

                logRates_end_step_location[rateOffset] = logfr;

                rates_end_step_location[rateOffset] =exp(logfr)-dis;

            }
            // rate aliveIndex is done, now do other rates 
            ++r;
            for (; r < rates; ++r)
            {
                float taur = tex1Dfetch(tex_taus,taus_offset+r);
                float fr = rates_start_step_location[r*paths];
                float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                float frtaur = fr*taur;
                float frtaud = frtaur+dis*taur;
                float gr =  frtaud/(1.0f+frtaur);
                float mur=0.0f;

                for (int f=0; f < factors; ++f)
                {
                    float arf = tex1Dfetch(tex_pseudoRoots, pseudoRoots_offset+f+r*factors);
                    float eInc =  gr*arf;
                    float  thise=   ( e_loc[f*paths]+= eInc);
                    mur +=  thise*arf;
                }    

                int rateOffset = r*paths;

                float change = mur+ incrementsLocation[rateOffset];
                float logfr =  logRates_start_step_location[r*paths];
                logfr += change;

				
				if (logfr > logCeiling)
					logfr= logCeiling;

                logRates_end_step_location[rateOffset] = logfr;

                rates_end_step_location[rateOffset] =exp(logfr)-dis;

            }  

        }
    }
}

// initial step is much simpler because the drifts are known in advance

__global__ void LMM_evolver_euler_initial_step_kernel(int pseudoRoots_offset, // texture offset 
                                                      int taus_offset,  // texture offset 
                                                      int dis_offset, // texture offset
                                                      int drifts_offset, // texture offset
                                                      int initialLogRates_offset, // texture offset
                                                      float* logRates_end_step,
                                                      float* rates_end_step,
                                                      float* correlatedBrownianIncrements,
                                                      int factors, 
                                                      int paths,
                                                      int rates
                                                      )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);

    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;

            float* incrementsLocation =     correlatedBrownianIncrements +pathOffset;
            float* logRates_end_step_location = logRates_end_step+pathOffset;
            float* rates_end_step_location = rates_end_step+pathOffset;

            for (int r=0; r < rates; ++r)
            {
                int rateOffset = r*paths;
                float logfr = tex1Dfetch(tex_initialLogRates,initialLogRates_offset+r);

                float mur=  tex1Dfetch(tex_drifts,drifts_offset+r);                       
                float change = mur+ incrementsLocation[rateOffset];
                logfr += change;

				if (logfr > logCeiling) 
					logfr = logCeiling;

                logRates_end_step_location[rateOffset] = logfr;
                float dis = tex1Dfetch(tex_displacements,dis_offset+r);
                rates_end_step_location[rateOffset] =exp(logfr)-dis;
            }  
        }
    }
}


extern "C"
void LMM_evolver_euler_gpu(  float* initial_rates_device, 
                           float* initial_log_rates_device, 
                           float* taus_device, 
                           float* correlatedBrownianIncrements_device,
                           float* pseudo_roots_device,
                           float* initial_drifts_device, 
                           float* displacements_device,
                           const std::vector<int>& aliveIndices, 
                           int paths,
                           int factors,
                           int steps, 
                           int rates, 
                           float* e_buffer_device,
                           float* evolved_rates_device, // for output
                           float* evolved_log_rates_device  // for output 
                           )
{

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;

   Timer h1;

    CUT_CHECK_ERR("LMM_evolver_euler_initial_step_gpu execution failed before entering kernel\n");


    // set up textures


    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

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
    tex_drifts.addressMode[0] = cudaAddressModeWrap;
    tex_drifts.addressMode[1] = cudaAddressModeWrap;
    tex_drifts.filterMode = cudaFilterModeLinear;
    tex_drifts.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_drifts, initial_drifts_device, channelDesc);

    // set texture parameters
    tex_displacements.addressMode[0] = cudaAddressModeWrap;
    tex_displacements.addressMode[1] = cudaAddressModeWrap;
    tex_displacements.filterMode = cudaFilterModeLinear;
    tex_displacements.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_displacements, displacements_device, channelDesc);


    // textures are now set up

    CUT_CHECK_ERR("LMM_evolver_euler_initial_step_kernel textures execution failed before entering kernel\n");



    int offset=0;

    LMM_evolver_euler_initial_step_kernel<<<dimGrid , dimBlock >>>(offset, // texture offset 
        offset,  // texture offset 
        offset,  // texture offset 
        offset, // texture offset
        offset, // texture offset
        evolved_log_rates_device,
        evolved_rates_device,
        correlatedBrownianIncrements_device,
        factors, 
        paths,
        rates
        );



    cutilSafeCall(cudaThreadSynchronize());         

    CUT_CHECK_ERR("LMM_evolver_euler_initial_step_kernel execution failed after entering kernel\n");     

    int layerSize = rates*paths;

    for (int s=1; s < steps; ++s)
    {
        CUT_CHECK_ERR("LMM_evolver_euler_noninitial_step_kernel execution failed before entering kernel\n");     

        int pseudoRoots_offset = s*rates*factors;
        int taus_offset =0;
        int dis_offset =0;
        int aliveIndex =aliveIndices[s] ;
        LMM_evolver_euler_noninitial_step_kernel<<<dimGrid , dimBlock >>>(pseudoRoots_offset, // texture offset 
            taus_offset,  // texture offset 
            dis_offset, // displacement offset
            aliveIndex, 
            evolved_log_rates_device+layerSize*(s-1),
            evolved_rates_device+layerSize*(s-1),
            evolved_log_rates_device+layerSize*s,
            evolved_rates_device+layerSize*s,
            //                 out_drifts_device+layerSize*s, 
            correlatedBrownianIncrements_device+layerSize*s,
            e_buffer_device,
            factors, 
            paths,
            rates,
            s
            );

        cutilSafeCall(cudaThreadSynchronize());                                                 

        CUT_CHECK_ERR("LMM_evolver_euler_noninitial_step_kernel execution failed after entering kernel\n");     

    }

    cudaUnbindTexture(tex_initialRates);
    cudaUnbindTexture(tex_initialLogRates);
    cudaUnbindTexture(tex_taus);
    cudaUnbindTexture(tex_pseudoRoots);
    cudaUnbindTexture(tex_drifts);


 
    double time = h1.timePassed();
 //   std::cout << " time taken for LMM evolution : " 
   //     << time  << std::endl;



}

__global__ void discounts_ratios_computation_kernel_textures(   int taus_offset, 
                                                             int aliveIndex, 
                                                             int rates_offset,
                                                             float* discount_factors_global,
                                                             int paths,
                                                             int n_rates
                                                             )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;

            float* discounts_location  = discount_factors_global+pathOffset+aliveIndex*paths;
            *discounts_location = 1.0f;

            float df = 1.0f;

            int tex_rates_offset = rates_offset+pathOffset+aliveIndex*paths;

            for (int r=aliveIndex; r < n_rates; ++r)
            {
                discounts_location+=paths;
                float taur = tex1Dfetch(tex_taus,taus_offset+r);
                float f = tex1Dfetch(tex_evolvedRates,tex_rates_offset);
                df /= (1.0f+ taur*f);
                *discounts_location = df;  
                tex_rates_offset +=    paths;

            }             
        }
    }
}


__global__ void discounts_ratios_computation_kernel_textures_all_steps(   int taus_offset, 
                                                                       int aliveIndex_offset, 
                                                                       int rates_offset,
                                                                       float* discount_factors_global,
                                                                       int paths,
                                                                       int n_rates,
                                                                       int number_steps,
                                                                       int discLayerSize
                                                                       )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;

            for (int s=0; s < number_steps; ++s)
            {
                int aliveIndex =  tex1Dfetch(tex_alive,aliveIndex_offset+s);
                float* discounts_location  = discount_factors_global+s*discLayerSize+pathOffset+aliveIndex*paths;
                *discounts_location = 1.0f;

                float df = 1.0f;

                int tex_rates_offset = rates_offset+pathOffset+aliveIndex*paths+s*paths*n_rates;

                for (int r=aliveIndex; r < n_rates; ++r)
                {
                    discounts_location+=paths;
                    float taur = tex1Dfetch(tex_taus,taus_offset+r);
                    float f = tex1Dfetch(tex_evolvedRates,tex_rates_offset);
                    df /= (1.0f+ taur*f);
                    *discounts_location = df;  
                    tex_rates_offset +=    paths;

                } 
            }            
        }
    }
}


extern "C"
void discount_ratios_computation_gpu(  float* evolved_rates_device, 
                                     float* taus_device, 
                                     const std::vector<int>& aliveIndices, 
                                     int* alive_device,
                                     int paths,
                                     int steps, 
                                     int rates, 
                                     float* discounts_device,  // for output
                                     bool allStepsAtOnce 
                                     )
{

    const int threadsperblock = 512;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;

	Timer h1;

	// set up textures

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);


    // set texture parameters
    tex_taus.addressMode[0] = cudaAddressModeWrap;
    tex_taus.addressMode[1] = cudaAddressModeWrap;
    tex_taus.filterMode = cudaFilterModeLinear;
    tex_taus.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_taus, taus_device, channelDesc);

    // set texture parameters
    tex_evolvedRates.addressMode[0] = cudaAddressModeWrap;
    tex_evolvedRates.addressMode[1] = cudaAddressModeWrap;
    tex_evolvedRates.filterMode = cudaFilterModeLinear;
    tex_evolvedRates.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_evolvedRates, evolved_rates_device, channelDesc);


    tex_alive.addressMode[0] = cudaAddressModeWrap;
    tex_alive.addressMode[1] = cudaAddressModeWrap;
    tex_alive.filterMode = cudaFilterModeLinear;
    tex_alive.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_alive, alive_device, channelDesc);


    // textures are now set up

    CUT_CHECK_ERR("discount_ratios_computation_gpu execution failed before entering kernel\n");


    int offset=0;

    if (allStepsAtOnce)
    {
        int discLayerSize = paths*(rates+1);
        discounts_ratios_computation_kernel_textures_all_steps<<<dimGrid , dimBlock >>>( offset, 
            offset, 
            offset,
            discounts_device,
            paths,
            rates,
            steps,
            discLayerSize
            );
    }
    else
    {

        for (int s=0; s < steps; ++s)  
        { 

            discounts_ratios_computation_kernel_textures<<<dimGrid , dimBlock >>>(offset, // texture offset 
                aliveIndices[s],
                s*paths*rates,
                discounts_device+s*paths*(rates+1),
                paths,
                rates);

            cutilSafeCall(cudaThreadSynchronize());         
        }                                                                                                                            
    }                                                                                                        




    cutilSafeCall(cudaThreadSynchronize());         

    CUT_CHECK_ERR("discount_ratios_computation_gpu execution failed after entering kernel\n");     

    cudaUnbindTexture(tex_taus);


   
    double time = h1.timePassed();
   // std::cout << " discounts computation. " 
     //   << time << std::endl;



}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void coterminal_annuity_ratios_computation_kernel(   int taus_offset, 
                                                             int aliveIndex, 
                                                             const float* __restrict__ discounts_global,
                                                             float* annuities_global,
                                                             int paths,
                                                             int n_rates
                                                             )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;

            int discounts_locationOffset  = pathOffset+paths;
            float* annuities_location  = annuities_global+pathOffset;

            float thisAnnuity =0.0f;


            for (int r=n_rates-1; r >= aliveIndex; --r)
            {
                float taur = tex1Dfetch(tex_taus,taus_offset+r);
                float df = discounts_global[discounts_locationOffset+r*paths];
                thisAnnuity += df*taur;

                annuities_location[r*paths] = thisAnnuity;     
            }             
        }
    }
}


extern "C"
void coterminal_annuity_ratios_computation_gpu(  float* discounts_device, 
                                               float* taus_device, 
                                               const std::vector<int>& aliveIndices, 
                                               int paths,
                                               int steps, 
                                               int rates, 
                                               float* annuities_device  // for output 
                                               )
{

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;

	Timer h1;

    // set up textures

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);



    // set texture parameters
    tex_taus.addressMode[0] = cudaAddressModeWrap;
    tex_taus.addressMode[1] = cudaAddressModeWrap;
    tex_taus.filterMode = cudaFilterModeLinear;
    tex_taus.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_taus, taus_device, channelDesc);

    // textures are now set up

    CUT_CHECK_ERR("coterminal_annuity_ratios_computation_gpu execution failed before entering kernel\n");


    int offset=0;

    for (int s=0; s < steps; ++s)  
    { 

        coterminal_annuity_ratios_computation_kernel<<<dimGrid , dimBlock >>>(offset, // texture offset 
            aliveIndices[s],
            discounts_device+s*paths*(rates+1),
            annuities_device+s*paths*rates,
            paths,
            rates);


    }                                                                                                        




    cutilSafeCall(cudaThreadSynchronize());         

    CUT_CHECK_ERR("coterminal_annuity_ratios_computation_kernel execution failed after entering kernel\n");     

    cudaUnbindTexture(tex_taus);


    
    double time = h1.timePassed();
//    std::cout << " discounts computation. " 
  //      << time  << std::endl;



}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void coterminal_swap_rate_computation_kernel(  
    int aliveIndex, 
    const float* __restrict__ discounts_global,
    const float* __restrict__ annuities_global,
    float* coterminal_swap_rates_global,
    int paths,
    int n_rates
    )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;

       //     float* discounts_location  = discounts_global+pathOffset;
         //   float* annuities_location  = annuities_global+pathOffset;
            float* cot_swaps_location  = coterminal_swap_rates_global+pathOffset;

        
            for (int r=aliveIndex; r < n_rates; ++r)
            {
				float numerator = discounts_global[pathOffset+r*paths]
									-discounts_global[pathOffset+n_rates*paths];

                float annuity=   annuities_global[pathOffset+r*paths];

                cot_swaps_location[r*paths] = numerator/annuity;     
            }  
        }
    }
}


extern "C"
void coterminal_swap_rate_computation_gpu(  float* discounts_device, 
                                          float* annuities_device ,
                                          const std::vector<int>& aliveIndices, 
                                          int paths,
                                          int steps, 
                                          int rates, 
                                          float* cot_swap_rates_device
                                          )
{

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;

    Timer h1;
    for (int s=0; s < steps; ++s)  
    { 
    //    std::cout << s << "of " << steps <<  "\n";
        CUT_CHECK_ERR("coterminal_swap_rate_computation_gpu execution failed before entering kernel\n");

        coterminal_swap_rate_computation_kernel<<<dimGrid , dimBlock >>>(
            aliveIndices[s],
            discounts_device+s*paths*(rates+1),
            annuities_device+s*paths*rates,
            cot_swap_rates_device+s*paths*rates,
            paths,
            rates);

        cutilSafeCall(cudaThreadSynchronize());         

        CUT_CHECK_ERR("coterminal_swap_rate_computation_kernel execution failed after entering kernel\n");     




    }                                                                                                        


//    std::cout << " swap rates computation: " 
  //      << h1.timePassed() << std::endl;



}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void swap_rate_computation_kernel(   int taus_offset, 
                                             int startIndex,
                                             int endIndex,
                                             float* discounts_global,
                                             float* swap_rates_global, 
                                             int paths,
                                             int n_rates
                                             )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;

            float* discounts_location  = discounts_global+pathOffset;
            float* swap_rate_location  = swap_rates_global+pathOffset;


            float initialDf = discounts_location[startIndex*paths];
            float finalDf = discounts_location[endIndex*paths];


            float theAnnuity =0.0f;


            for (int r=startIndex; r < endIndex; ++r)
            {
                float taur = tex1Dfetch(tex_taus,taus_offset+r);
                float df = discounts_location[(r+1)*paths];
                theAnnuity += df*taur;
            }             

            float SR = (initialDf - finalDf)/ theAnnuity;
            *swap_rate_location=SR;

        }
    }
}


extern "C"
void swap_rate_computation_gpu(  float* discounts_device, 
                               float* taus_device, 
                               int startIndex,
                               int endIndex, 
                               int paths,
                               int step_for_offset_in,
                               int step_for_offset_out, 
                               int rates, 
                               float* swap_rate_device  // for output 
                               )
{

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;


    // set up textures

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // set texture parameters
    tex_taus.addressMode[0] = cudaAddressModeWrap;
    tex_taus.addressMode[1] = cudaAddressModeWrap;
    tex_taus.filterMode = cudaFilterModeLinear;
    tex_taus.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_taus, taus_device, channelDesc);

    // textures are now set up

    CUT_CHECK_ERR("swap rate computation gpu execution failed before entering kernel\n");


    int offset=0;


    swap_rate_computation_kernel<<<dimGrid , dimBlock >>>(offset, // texture offset 
        startIndex,
        endIndex,
        discounts_device+step_for_offset_in*paths*(rates+1),
        swap_rate_device+step_for_offset_out*paths,
        paths,
        rates);






    cutilSafeCall(cudaThreadSynchronize());         

    CUT_CHECK_ERR("coterminal_annuity_ratios_computation_kernel execution failed after entering kernel\n");     

    cudaUnbindTexture(tex_taus);




}

// assumption that rates and steps are the same here 
__global__ void spot_measure_numeraires_computation_kernel(  
    const float* __restrict__ discount_factors_global,
    float* numeraire_values_global,
    int paths,
    int pathsForOutput, 
    int pathsOffsetForOutput,
    int n_rates
    )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        if (pathNumber < paths)
        {
            float numeraireValue =1.0f;
            int offsetNumeraire =  pathNumber+pathsOffsetForOutput;
            int offsetDiscounts = pathNumber;

            numeraire_values_global[offsetNumeraire] = numeraireValue;

            for (int i=1; i < n_rates; ++i)
            {
                offsetNumeraire +=pathsForOutput;

                int offsetDiscountsNext = offsetDiscounts+paths*i+paths*(n_rates+1)*(i-1);
                int offsetDiscountsCurrent= offsetDiscounts+paths*(i-1)+paths*(n_rates+1)*(i-1);

                float dfNext = discount_factors_global[offsetDiscountsNext];
                float dfNow = discount_factors_global[offsetDiscountsCurrent];

                float factorIncrease = dfNow/dfNext;
                numeraireValue *= factorIncrease;

                numeraire_values_global[offsetNumeraire] = numeraireValue;

            }
        }
    }
}


extern "C"
void spot_measure_numeraires_computation_offset_gpu(        float* discount_factors_global,
                                                    float* numeraire_values_global, //output
                                                    int paths,
                                                    int pathsForOutput,
                                                    int pathsOffsetForOutput,
                                                    int n_rates
                                                    )
{

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;

    // textures are now set up

    CUT_CHECK_ERR("spot_measure_numeraires_computation gpu execution failed before entering kernel\n");



    spot_measure_numeraires_computation_kernel<<<dimGrid , dimBlock >>>(discount_factors_global,
        numeraire_values_global,
        paths,
        pathsForOutput,
        pathsOffsetForOutput,
        n_rates);






    cutilSafeCall(cudaThreadSynchronize());         

    CUT_CHECK_ERR("spot_measure_numeraires_computation execution failed after entering kernel\n");     

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C"
void spot_measure_numeraires_computation_gpu(        float* discount_factors_global,
                                             float* numeraire_values_global, //output
                                             int paths,
                                             int n_rates)
{
    spot_measure_numeraires_computation_offset_gpu(  discount_factors_global,
        numeraire_values_global, //output
        paths,
        paths,//pathsForOutput,
        0, // pathsOffsetForOutput,
        n_rates
        );
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void forward_rate_extraction_kernel(  int step,
                                               int forwardIndex, 
                                               const float* __restrict__ forward_rates_global, 
                                               float* forwards_out_global,
                                               int paths,
                                               int n_rates
                                               )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;

            const float* __restrict__ all_forwards_location  = forward_rates_global+pathOffset+step*paths*n_rates;
            float* forwards_out_location  = forwards_out_global+pathOffset+step*paths;

            float f =    all_forwards_location[forwardIndex*paths];
            *forwards_out_location =f;

        }
    }
}


extern "C"
void forward_rate_extraction_gpu(  float* all_forwards_device, 
                                 const std::vector<int>& forwardIndices,                          
                                 int paths,
                                 int steps,
                                 int rates, 
                                 float* select_forwards_device                  
                                 )
{

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;   

    CUT_CHECK_ERR("forward_rate_extraction_kernel gpu execution failed before entering kernel\n");




    for (int s=0; s<steps; ++s)
        forward_rate_extraction_kernel<<<dimGrid , dimBlock >>>(s,
        forwardIndices[s],
        all_forwards_device,
        select_forwards_device,
        paths,
        rates);





    cutilSafeCall(cudaThreadSynchronize());         

    CUT_CHECK_ERR("forward_rate_extraction_kernel execution failed after entering kernel\n");     
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void forward_rate_extraction_selecting_step_kernel(  int step,
                                               int forwardIndex, 
											   int stepIndex,
                                               const float* __restrict__ forward_rates_global, 
                                               float* forwards_out_global,
                                               int paths,
                                               int n_rates
                                               )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( paths -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        if (pathNumber < paths)
        {
            int pathOffset =  pathNumber;

      //      float* all_forwards_location  = forward_rates_global+pathOffset+stepIndex*paths*n_rates;
			int forwardoffset = pathOffset+stepIndex*paths*n_rates;
            float* forwards_out_location  = forwards_out_global+pathOffset+step*paths;

            float f =    forward_rates_global[forwardoffset+forwardIndex*paths];
            *forwards_out_location =f;

        }
    }
}


extern "C"
void forward_rate_extraction_selecting_step_gpu(  float* all_forwards_device, 
                                 const std::vector<int>& forwardIndices,        
								 const std::vector<int>& stepIndices,                          
                                 int paths,
                                 int rates, 
                                 float* select_forwards_device                  
                                 )
{
	int steps = forwardIndices.size();
	if (steps != stepIndices.size())
		GenerateError("size mismatch in forward_rate_extraction_selecting_step_gpu");

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;   

    CUT_CHECK_ERR("forward_rate_extraction_selecting_step_gpu gpu execution failed before entering kernel\n");


        ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

    for (int s=0; s<steps; ++s)
        forward_rate_extraction_selecting_step_kernel<<<dimGrid , dimBlock >>>(s,
        forwardIndices[s],
		stepIndices[s],
        all_forwards_device,
        select_forwards_device,
        paths,
        rates);





    cutilSafeCall(cudaThreadSynchronize());         

    CUT_CHECK_ERR("forward_rate_extraction_kernel execution failed after entering kernel\n");     
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void spot_measure_numeraires_extraction_kernel(  
    float* all_numeraire_values_global, 
    float* numeraires_out_global,
    int pathsInput,
    int pathsOutput,
    int offsetForOutput,
    int numberRates,
    int numberExerciseTimes, 
    int* exerciseIndices_global
    )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;


    extern __shared__ int indices_shared[];
    while (tx < numberExerciseTimes)
    {
        indices_shared[tx] = exerciseIndices_global[tx];
        tx+=bwidth;
    }
    __syncthreads();
    tx =  threadIdx.x;


    int pathsPerThread = 1 + (( pathsInput -1)/width);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;

        if (pathNumber < pathsInput)
        {
            for (int i=0; i < numberExerciseTimes; ++i)
                numeraires_out_global[offsetForOutput+pathNumber+pathsOutput*i] = all_numeraire_values_global[pathNumber+pathsInput*indices_shared[i]];


        }
    }
}

double spot_measure_numeraires_extraction_gpu(   float* all_numeraire_values_global,
                                             float* some_numeraire_values_global, // output
                                            int pathsInput,
                                            int pathsOutput,
                                            int offsetForOutput,
                                            int numberRates,
                                            int numberExerciseTimes, 
                                            int* exerciseIndices_vec
                                            )
{
    int sharedSize = numberExerciseTimes*sizeof(int);

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(pathsInput-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock; 
    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
	
    Timer h1;

     //  if (minimizeTextures)
    spot_measure_numeraires_extraction_kernel<<<dimGrid, dimBlock,sharedSize>>>(all_numeraire_values_global,
                                            some_numeraire_values_global,
                                            pathsInput,
                                             pathsOutput,
                                            offsetForOutput,
                                             numberRates,
                                            numberExerciseTimes, 
                                             exerciseIndices_vec //output
        );

    cutilSafeCall(cudaThreadSynchronize());

  
    return h1.timePassed();

}

