//
//\
//                  Mersenne_gold.cpp
//

/*

Copyright (c) 2012 Mark Joshi

released under the GPL v 3.0

adapted from code with the following license

This file is part of QuantLib, a free-software/open-source library
for financial quantitative analysts and developers - http://quantlib.org/

QuantLib is free software: you can redistribute it and/or modify it
under the terms of the QuantLib license.  You should have received a
copy of the license along with this program; if not, please email
<quantlib-dev@lists.sf.net>. The license is also available online at
<http://quantlib.org/license.shtml>.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the license for more details.
*/
#include <gold/Mersenne_gold.h>
#include <gold/MatrixFacade.h>
#include <time.h>


// Period parameters
const size_t MersenneTwisterUniformRng::N = 624;
const size_t MersenneTwisterUniformRng::M = 397;
// constant vector a
const unsigned long MersenneTwisterUniformRng::MATRIX_A = 0x9908b0dfUL;
// most significant w-r bits
const unsigned long MersenneTwisterUniformRng::UPPER_MASK=0x80000000UL;
// least significant r bits
const unsigned long MersenneTwisterUniformRng::LOWER_MASK=0x7fffffffUL;

size_t MersenneTwisterUniformRng::mti;


MersenneTwisterUniformRng::MersenneTwisterUniformRng(unsigned long seed)
: mt(N) 
{
    seedInitialization(seed);
}

void MersenneTwisterUniformRng::seedInitialization(unsigned long seed) 
{
    /* initializes mt with a seed */
    //unsigned long s = (seed != 0 ? seed : long(time(0)));
    // change default seed to 1 not time
    unsigned long s = (seed != 0 ? seed : 1);
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) {
        mt[mti] =
            (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}

MersenneTwisterUniformRng::MersenneTwisterUniformRng(const std::vector<unsigned long>& seeds)
: mt(N) 
{
    seedInitialization(19650218UL);
    size_t i=1, j=0, k = (N>seeds.size() ? N : seeds.size());
    for (; k; k--) 
    {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
            + seeds[j] + j; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=N) { mt[0] = mt[N-1]; i=1; }
        if (j>=seeds.size()) j=0;
    }

    for (k=N-1; k; k--) 
    {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
            - i; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=N) { mt[0] = mt[N-1]; i=1; }
    }

    mt[0] = 0x80000000UL; /*MSB is 1; assuring non-zero initial array*/
}


unsigned long MersenneTwisterUniformRng::nextInt32() const 
{    
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        size_t kk;

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }

    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

//purely here to force compilation of template method
namespace{

    void getUniforms() 
    {
        std::vector<double> v(10);

        MersenneTwisterUniformRng rng;

        rng.getUniforms(v.begin(),v.end());

    }

      void getUniformsInts() 
    {
        std::vector<int> v(10);

        MersenneTwisterUniformRng rng;

        rng.getInts(v.begin(),v.end());

    }
      void getCube() 
    {
        std::vector<double> v(30);
        CubeFacade<double> vCube(v,2,3,5);


        MersenneTwisterUniformRng rng;

        rng.populateCubeWithUniforms<CubeFacade<double>, double>(vCube);
        rng.populateCubeWithNormals<CubeFacade<double>, double>(vCube);
    }
}
