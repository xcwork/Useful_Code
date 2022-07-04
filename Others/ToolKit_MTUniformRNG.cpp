#include "ToolKit/ToolKit_MTUniformRNG.h"
#pragma warning(disable: 4267)
#ifndef TOOLKIT_DEFAULTSEED
#define TOOLKIT_DEFAULTSEED 29
#endif
namespace ToolKit
	{
		// constant vector a
		const unsigned long MTUniformRNG::MATRIX_A = 0x9908b0dfUL;
		// most significant w-r bits
		const unsigned long MTUniformRNG::UPPER_MASK=0x80000000UL;
		// least significant r bits
		const unsigned long MTUniformRNG::LOWER_MASK=0x7fffffffUL;

		MTUniformRNG::MTUniformRNG(unsigned long seed) 
		{
        seedInitialization(seed);
		}

		void MTUniformRNG::seedInitialization(unsigned long seed) 
		{
			/* initializes mt with a seed */
			unsigned long s = (seed != 0 ? seed : TOOLKIT_DEFAULTSEED);
			mt[0]= s & 0xffffffffUL;
			for (mti=1; mti<N; mti++) {
				mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
				/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
				/* In the previous versions, MSBs of the seed affect   */
			    /* only MSBs of the array mt[].                        */
				/* 2002/01/09 modified by Makoto Matsumoto             */
				mt[mti] &= 0xffffffffUL;
				/* for >32 bit machines */
			}
		}
		MTUniformRNG::MTUniformRNG(const std::vector<unsigned long>& seeds) 
		{
			seedInitialization(19650218UL);
			unsigned int i=1, j=0, k = (N>seeds.size() ? N : seeds.size());
			for (; k; k--) {
				mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
					+ seeds[j] + j; /* non linear */
				mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
				i++; j++;
				if (i>=N) { mt[0] = mt[N-1]; i=1; }
				if (j>=seeds.size()) j=0;
			}
			for (k=N-1; k; k--) {
				mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
					- i; /* non linear */
				mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
				i++;
				if (i>=N) { mt[0] = mt[N-1]; i=1; }
			}

			mt[0] = UPPER_MASK; /*MSB is 1; assuring non-zero initial array*/
		}
		void MTUniformRNG::twist() const {
			static const unsigned long mag01[2]={0x0UL, MATRIX_A};
			/* mag01[x] = x * MATRIX_A  for x=0,1 */
			unsigned int kk;
			unsigned long y;

			for (kk=0;kk<N-M;kk++) {
				y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
				mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
			}
			for (;kk<N-1;kk++) {
				y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
				mt[kk] = mt[(kk+M)-N] ^ (y >> 1) ^ mag01[y & 0x1UL];
			}	
			y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
			mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

			mti = 0;
		}
	
}

