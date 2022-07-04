//
//
//                                  Mersenne.h
//
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


#ifndef mersennetwister_uniform_rng_h
#define mersennetwister_uniform_rng_h

#include <vector>
#include <gold/math/Normals_gold.h>
#include <algorithm>

//! Uniform random number generator
/*! Mersenne Twister random number generator of period 2**19937-1

  For more details see http://www.math.keio.ac.jp/matumoto/emt.html
*/
class MersenneTwisterUniformRng 
{
public:

    /*! if the given seed is 0, the seed 1 will be used */
    explicit MersenneTwisterUniformRng(unsigned long seed = 0);
    explicit MersenneTwisterUniformRng(
        const std::vector<unsigned long>& seeds);
        /*! returns a sample with weight 1.0 containing a random number
    on (0.0, 1.0)-real-interval  */
    double next() const 
    {
        // divide by 2^32
        double result = (double(nextInt32())+ 0.5)/4294967296.0;
        return result;
    }
    
    //! return  a random number on [0,0xffffffff]-interval 
    unsigned long nextInt32() const;

    template<class T>
    void getUniforms(T start, T end);

    template<class T>
    void getInts(T start, T end);

    template<class T, class S>
    void populateCubeWithUniforms(T& cubeData);

    template<class T, class S>
    void populateCubeWithNormals(T& cubeData);
private:
    void seedInitialization(unsigned long seed);
    mutable std::vector<unsigned long> mt;
    static const size_t N, M;
    static const unsigned long MATRIX_A, UPPER_MASK, LOWER_MASK;
    static size_t mti;
};


 template<class T>
 void MersenneTwisterUniformRng::getUniforms(T start, T end)
 {
    while (start != end)
    {
        *start =next();
        ++start;

    }
 }

    
 template<class T>
 void MersenneTwisterUniformRng::getInts(T start, T end)
 {
    while (start != end)
    {
        *start =nextInt32();
        ++start;

    }
 }

 template<class T, class S>
 void MersenneTwisterUniformRng::populateCubeWithUniforms(T& cubeData)
 {

     S* start = &cubeData(0,0,0);
     S* end = start + cubeData.numberRows() *cubeData.numberColumns() *cubeData.numberLayers();

     getUniforms(start, end);
 }

     
    
 template<class T, class S>
 void MersenneTwisterUniformRng::populateCubeWithNormals(T& cubeData)
 {

     S* start = &cubeData(0,0,0);
     S* end = start + cubeData.numberRows() *cubeData.numberColumns() *cubeData.numberLayers();

     getUniforms(start, end);

     inverseCumulativeGold<S> transformer;

     std::transform(start,end,start,transformer);
 }

#endif
