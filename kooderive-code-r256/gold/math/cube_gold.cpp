//
//
//                                             CUBE_GOLD_CPP
//
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3


#include <gold/math/cube_gold.h>

// dumb function to force compilation of templates
void funcCubeTest()
{
	Cube_gold<double> cubby(1,1,1,0.0);
	
	std::vector<double> v;
	Cube_gold<double> cubv(1,1,1,v);
	
	CubeFacade<double> cubf(v,1,1,1);
	Cube_gold<double> cubvf(cubf);

	CubeConstFacade<double> cubfc(v,1,1,1);
	const Cube_gold<double> cubvfc(cubfc);

	double x= cubvfc(0,0,0) + cubvf(0,0,0);

	CubeConstFacade<double> cubvfcf(cubvfc);
	CubeFacade<double> cubvfcff(cubvf);

	cubvf[0](0,0) =0.0+cubvfc[0](0,0);

	CubeFacade<float> test2(CubeTypeConvert<float,double>(cubv));

 }


