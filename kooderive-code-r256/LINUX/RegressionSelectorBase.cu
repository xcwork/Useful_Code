//
//
//                                      RegressionSelectorBase.cu
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3


#include <RegressionSelectorBase.h>
#include <Utilities.h>

double tolerance = 1e-6;
double misMatchTol = 1e-5;

int RegressionSelector::Select_test_mode(int depth,
		        thrust::device_vector<float>::iterator& start, 
				thrust::device_vector<float>::iterator& end,
				thrust::device_vector<int>::iterator& selected,
				float& lowerCutOff,
				float& upperCutOff,
				bool& testPassed)
{

	 int count1 = Select(depth,
		        start, 
				end,
				selected,
				lowerCutOff,
				upperCutOff);

	 thrust::host_vector<float> data_host(start,end);
	 std::vector<float> data_vec(data_host.begin(),data_host.end());

	 std::vector<int> selected_vec(data_vec.size());

     thrust::host_vector<float> selected_host(selected,selected+data_vec.size());
	 std::vector<float> selected_gpu_vec(selected_host.begin(),selected_host.end());

	 float l,u;

	 int dataSize = data_vec.size();

	 Select_gold( depth,
				data_vec,
				dataSize,
				selected_vec,
				l,
				u);

	testPassed = true;

	if ( fabs(l-lowerCutOff) > tolerance || fabs(u-upperCutOff) > tolerance)
		testPassed=false;

	int mismatches =0;


	for (int j=0; j < dataSize; ++j)
		mismatches += selected_gpu_vec[j] != selected_vec[j] ? 1 : 0;

	if (mismatches > dataSize* misMatchTol)
		testPassed=false;

	if (!testPassed)
	{
		std::cout << " regression selector test failed \n" << "l," << l <<", lower ," <<  lowerCutOff <<  "u," << u <<", upper ," <<  upperCutOff <<"\n" ;
		std::cout << "mismatches, "<< mismatches << " , out of " << dataSize << " \n";
	}

	return count1;

}