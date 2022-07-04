//
//                                  cash_flow_product_equity_one_dim_gpu.cu
//
//


// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3
#include <cash_flow_product_equity_one_dim_gpu.h>
#include <cash_flow_generation_equity_one_dim_gpu.h>


// product specific data is passed in via the texture tex_aux_data

class RatchetPut
{

public:

__device__ RatchetPut(int offset, int numberSteps)
{
    numberStepsMin1_ = numberSteps-1;
    strike_ = tex1Dfetch(tex_aux_data,offset); // the total coupon during the product's fixed life time at offset
  
}

__device__ float getPayOff(float* stockPaths,int paths,int pathNumber)
{
   
    float cStrike = strike_;

    for (int i=0; i < numberStepsMin1_; ++i)
    {
        float s= stockPaths[pathNumber+i*paths];
        cStrike = s > cStrike ? s : cStrike;


    }

    float s= stockPaths[pathNumber+numberStepsMin1_*paths];

    float result = s < cStrike ? cStrike - s : 0.0f;
    return result;
}


private:

    int numberStepsMin1_;
    float strike_;

};


extern "C"
void payoffGeneratorCallerRatchetPut(float* genFlows,  float* aux_data, int paths, int numberSteps,
                                                      float* stockPaths
 )

{
    cashFlowGeneratorEquityOneDim_gpu<RatchetPut>(genFlows,aux_data,paths,numberSteps,stockPaths);
}

