//
//
//                                  Transpose_main.h  
//                          
//
//
//

#include <transpose_main.h>
#include <transpose_gpu.h>

double transpose_using_gpu_main(thrust::device_vector<float>& input_data_dev, size_t inputrows, size_t inputcols,
                               thrust::device_vector<float>& output_dev,bool useTextures)
{
   if (useTextures)
        return transpose_using_texture_gpu(thrust::raw_pointer_cast(& input_data_dev[0]), inputrows,  inputcols,
                               thrust::raw_pointer_cast(& output_dev[0]));

   else

        return transpose_using_ldg_gpu(thrust::raw_pointer_cast(& input_data_dev[0]), inputrows,  inputcols,
                               thrust::raw_pointer_cast(& output_dev[0]));
}