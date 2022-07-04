#include "ratchet.h"
#include "bermudanIRDPricer.h"
#include <iostream>
#include <vector>
#include <cmath>

class dataPoint
{
public:
	int i;
	int j;
	double value;
};


int Ex2()
{
	int numberNonCallCoupons =0;
		
	double initialNumeraire =0.95;
		
	float firstForward=0.05f;
    float forwardIncrement=0.00f;
    float displacement=0.02f;
	float strike =0.05f;

	float beta=0.2f;				  
	float L=0.0f;
	double a= 0.0;
	double b = 0;
	double c= 1;
	double d=0.11;
	bool useFlatVols = true;

	double firstRateTime =0.5;
	double rateLength = 0.5;

	std::vector<dataPoint> res;
	
	double payReceive_d=1.0;
	bool useLog=true;
	int numberOfRates = 32;
	bool normalise = true;
	double lowerFrac = 0.7;
	double upperFrac=0.71;
	double initialSDguess = 2.5;
	double multiplier =0.8;
	int firstPassBatches = 10;
	int secondPassBatches = 10;
	bool useCrossTerms = true;
	int regressionDepth =3;	
	int lowerFracCut=1000;
	int duplicate =0;
	bool annul = true;

	bool globalDiscounting=false;

	double thisres=BermudanMultiLSPricerExample(32767,
									  firstPassBatches, 
									  secondPassBatches, 
									  useCrossTerms,
									  regressionDepth,
									  initialSDguess,
									  lowerFracCut,
									  lowerFrac,
									 upperFrac,
									 multiplier,
									normalise,
									duplicate,
									useLog,numberOfRates,
									firstForward,
								  forwardIncrement,
								  displacement,
								  strike,
								  beta,
								  L,a,b,c,d,
								  numberNonCallCoupons, firstRateTime, rateLength,payReceive_d, useFlatVols,annul,initialNumeraire,
								  globalDiscounting,
								  true,-1,0,false,false);

	std::cout << "Berm price " << thisres << "\n";

	return 0;

}

// examples from Beveridge Joshi Tang JEDC paper on PPI

int Ex1(int choice, int numberNonCallCoupons, bool useLog, bool normalise, bool useCrossTerms, int gpuChoice, bool scrambleFirst,bool scrambleSecond)
{

	std::vector<float> xa(3);
	xa[0] = 0.01f;
	xa[1] = 0.005f;
	xa[2] = 0.0f;

	std::vector<int> rates(3);
	rates[0] = 10+numberNonCallCoupons;
	rates[1] = 18+numberNonCallCoupons;
	rates[2] = 38+numberNonCallCoupons;

	int numberOfRates = rates[choice];
	
	float x =xa[choice];
	float firstForward=0.008f+x;
    float forwardIncrement=0.002f;
	double rateLength = 0.5;

	double initialNumeraireValue = 1.0/(1+firstForward*rateLength);
	for (int i=0; i < 2-numberNonCallCoupons; ++i)
		initialNumeraireValue /= 1+rateLength*(firstForward+(i+1)*forwardIncrement);


	
	firstForward += (3-numberNonCallCoupons)*forwardIncrement;
    float displacement=0.015f;
	float strike =0.04f;

	float beta=2*0.0669f;				  
	float L=0.0f;
	double a= 0.05;
	double b = 0.09;
	double c= 0.44;
	double d=0.2;
	bool useFlatVols = false;
	

	double firstRateTime =1.5-rateLength*numberNonCallCoupons;

	std::vector<dataPoint> res;
	
	double payReceive_d=1.0;
//	bool useLog=true;
	
	//bool normalise = false;
	bool annul = false;
	int duplicate =1;
	bool globalDiscounting=false;
	
//	bool useCrossTerms = true;

    float tlevel = 0.1f;
	int secondPassBatches = 32;
	int pathsPerBatch = 65536;
	int LMMthreads = 128;

	for (int i=10; i < 11; ++i, ++i)
	{
		for (int j=1; j < 5; ++j)
		{
			float lc = j> 0 ? pow(tlevel,(1.0f/(j-1))) : 0.0f;
			float uc = lc*1.01f;

			std::cout<< "\n\n" << i << "," << j << "\n";
			double thisres=BermudanMultiLSPricerExample(pathsPerBatch,
									  i, 
									  secondPassBatches, 
									  useCrossTerms,
									  j,
									  2.5f,
									  900,
									  lc,
									 uc,
									 0.8f,
									normalise,
									duplicate,
									useLog,numberOfRates,
									firstForward,
								  forwardIncrement,
								  displacement,
								  strike,
								  beta,
								  L,a,b,c,d,
								  numberNonCallCoupons, firstRateTime, rateLength,payReceive_d, useFlatVols,annul
								  , initialNumeraireValue
								  , globalDiscounting
								  ,
								  true, gpuChoice,LMMthreads,scrambleFirst,scrambleSecond);

			dataPoint thisValue = {i,j,thisres};

			res.push_back(thisValue);

		}
	}


	for (size_t i=0; i < res.size(); ++i)
		std::cout << res[i].i << "," << res[i].j<< "," << res[i].value << "\n";

	return 0;
}

int main()
{
//	BermudanPricerExample(8192,1, 1,true);

//	
	int gpuChoice =0;


	for (int i=0; i < 1;++i)
		Ex1(2,2, i%2 >0  , (i/2)%2 >0 , (i/4)%2>0, gpuChoice,false,true);
								  
 // BermudanPricerExample(16383,10, 10,true);

    	//
   RatchetPricerExample();
 //   char c;
  // std::cin >> c;
    return 0;
}
