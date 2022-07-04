
// (c) Mark Joshi 2009, 2010, 2013
// This code is released under the GNU public licence version 3


//  Utilities.cu


#include <Utilities.h>
#include <thrust/scan.h>
#include <thrust/inner_product.h>

void PartialSumsInt(thrust::device_vector<int>& input, int paths, thrust::device_vector<int>& output)
{
	thrust::inclusive_scan 	( input.begin(),
		                      input.begin()+paths,
							  output.begin()); 	
}


void doScatter(thrust::device_vector<float>& source,
			   int offset, 
			   int points,
			   int newPoints, 
			   thrust::device_vector<int>& indices,
			   thrust::device_vector<int>& selections_dev, 
			   thrust::device_vector<float>& wsp_dev,
			   int outOffset
			   )
{
		thrust::scatter_if( source.begin()+offset,
				source.begin()+points+offset,							
				indices.begin(),
				selections_dev.begin(),
				wsp_dev.begin() );

		thrust::copy(wsp_dev.begin()+1,
				wsp_dev.begin()+1+newPoints,
				source.begin()+outOffset);

}

double doInnerProduct(thrust::device_vector<float>& one_dev, thrust::device_vector<float>& two_dev)
{
	double res = thrust::inner_product(one_dev.begin(),one_dev.end(), two_dev.begin(),0.0f);
	return res;
}

		

void doScatterMulti(thrust::device_vector<float>& source, 
					int dataSize, 
					int points,
					int newPoints, 
					thrust::device_vector<int>& indices,
					thrust::device_vector<int>& selections_dev, 
					thrust::device_vector<float>& wsp_dev
			   )
{
	if (wsp_dev.size() < points*dataSize+1)
		GenerateError("wsp_dev is too small \n");

	for (int j=0; j < dataSize; ++j)
		thrust::scatter_if( source.begin()+points*j,
				source.begin()+points*j+points,							
				indices.begin(),
				selections_dev.begin(),
				wsp_dev.begin()+newPoints*j );
	

	thrust::copy(wsp_dev.begin()+1, wsp_dev.begin()+1+points*dataSize, source.begin() );

}
