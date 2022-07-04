
//                                      normals_gold.h

/*

Copyright (c) 2012 Mark Joshi

released under the GPL v 3.0


*/



#ifndef NORMALS_GOLD_H
#define NORMALS_GOLD_H
#include <gold/math/typedefs_math_gold.h>
#include <gold/math/constants_gold.h>
#include <cmath>

Realv normalDensity(Realv x);

Realv cumulativeNormal(Realv x);

Realv inverseCumulativeNormalDerivative(Realv x);

Realv BivariateCumulativeNormal(Realv a, Realv b, Realv rho);
Realv TruncatedFirstMomentBivariateNormal(Realv lowerFirstVariable, Realv lowerSecondVariable, Realv rho);
Realv ConditionalTruncatedFirstMomentBivariateNormal(Realv lowerFirstVariable, Realv lowerSecondVariable, Realv rho);
Realv UpperTruncatedFirstMomentBivariateNormal(Realv upperFirstVariable, Realv upperSecondVariable, Realv rho);

Realv LowerUpperTruncatedFirstMomentBivariateNormal(Realv lowerFirstVariable, Realv upperSecondVariable, Realv rho);

Realv UpperTruncatedFirstMomentBivariateNormal(Realv upperFirstVariable, Realv upperSecondVariable, 
                                               Realv mux, Realv sigmax, Realv muy, Realv sigmay, Realv sigmaxy);

Realv LowerTruncatedFirstMomentBivariateNormal(Realv lowerFirstVariable, Realv lowerSecondVariable, 
                                               Realv mux, Realv sigmax, Realv muy, Realv sigmay, Realv sigmaxy);

Realv ExpectedMaximumOfTwoGausssiansAndAConstant(Realv mux, Realv muy, Realv c, Realv sigmax2, Realv sigmaxy, Realv sigmay2);


Realv ExpectedMaximumOfThreeIndependentGausssians(Realv mux, Realv muy, Realv muz, Realv sigmax2, Realv sigmay2, Realv sigmaz2);

#define CQPG(v) (P1+v*(P2+v*(P3+v*(P4+(P5+P6*v)*v))))

#define CQQG(v) (Q1+v*(Q2+v*(Q3+v*(Q4+(Q5+Q6*v)*v))))

template <typename T>
struct inverseCumulativeShawBrickmanGold
{
T operator()(const T& v) const 
{
const float P1 = 1.2533136835212087879;
const float P2 = 1.9797154223229267471;
const float P3 = 0.80002295072483916762;
const float P4 = 0.087403248265958578062;
const float P5 = 0.0020751409553756572917;
const float P6 = 4.744820732427972462e-6;
const float Q1 = 1.0;
const float Q2 = 2.0795584360534589311;
const float Q3 = 1.2499328117341603014;
const float Q4 = 0.23668431621373705623;
const float Q5 = 0.0120098270559197768;
const float Q6 = 0.00010590620919921025259;
float z;
int sgn;
sgn = (v >= 0.5f);
sgn = sgn - !sgn;
z = -logf(1.0f - (sgn * ((2.0f * v) - 1.0f)));
return sgn * z * CQPG(z) / CQQG(z);
}

};

template <typename T>
struct inverseCumulativeGilesGold
{
T operator()(const T& x) const 
{
  const float POLY[] = { 4.560330337e-12f ,   1.891879037e-09f ,
                 6.384940120e-11f ,   2.112199920e-08f ,
                 3.709377697e-10f ,  -4.363333596e-09f ,
                 1.235508984e-09f ,  -1.340785237e-06f ,
                -2.085483499e-08f ,  -7.816014247e-06f ,
                -8.052312355e-07f ,   2.418058136e-05f ,
                -1.789424115e-05f ,   0.0005984976896f ,
                -0.0003717238455f ,   0.003071246478f ,
                -0.009026965781f  ,  -0.01373818033f ,
                -0.4222077998f    ,  -0.6926406482f ,
                 9.066901046f     ,   4.381417061f     };

 float MEAN[] = { -11.36390346f ,  -2.693891179f };
 float MID    = -6.114811502f;
 float w, p;
 int   n;

 w = __logf(4.0f*x*(1.0f-x));
 n = w>MID;
 w = w - MEAN[n];

 p = POLY[n];
 for (; n<22; n+=2)
  p = p*w + POLY[n];

 return (x-0.5f)*p;
}
};

template <typename T>
struct inverseCumulativeGold
{
T operator()(const T& u) const 
{
 
    const float a0= 2.50662823884f;
    const float a1=-18.61500062529f;
    const float a2= 41.39119773534f;
    const float a3 = -25.44106049637f;


    const float b0 = -8.47351093090f;
    const float b1 =  23.08336743743f;
    const float b2 = -21.06224101826f;
    const float b3 =  3.13082909833f;


    const float c0 = 0.3374754822726147f;
    const float c1 = 0.9761690190917186f;
    const float c2 = 0.1607979714918209f;
    const float c3 = 0.0276438810333863f;
    const float c4 = 0.0038405729373609f;
    const float c5 = 0.0003951896511919f;
    const float c6 = 0.0000321767881768f;
    const float c7 = 0.0000002888167364f;
    const float c8 = 0.0000003960315187f;
   
    
    T x=static_cast<T>(u-0.5);
    T r;
    
      if (fabs(x)<0.42) // Beasley-Springer
        {
           double y=x*x;
        
           r=x*(((a3*y+a2)*y+a1)*y+a0)/
                     ((((b3*y+b2)*y+b1)*y+b0)*y+1.0);
               
        }
        else // Moro
        {
    
            r=u;
    
            if (x>0.0) 
                r=1.0-u;
  
            r=log(-log(r));
        
            r=c0+r*(c1+r*(c2+r*(c3+r*(c4+r*(c5+r*(c6+
                    r*(c7+r*c8)))))));
        
            if (x<0.0) 
                 r=-r;
    
        }

   

   return static_cast<T>(r);
}


};
template <typename T>
struct inverseCumulativeUnsignedIntGold
{
T operator()(unsigned int u) const 
{
    static inverseCumulativeGold<T> obj;
    T v = u*2.3283064E-10F;
    return obj(v);
}
};


 template <typename T>
struct inverseCumulativeShawBrickmanUnsignedIntGold
{
T operator()(const unsigned int & vi) const 
{
const float P1 = 1.2533136835212087879f;
const float P2 = 1.9797154223229267471f;
const float P3 = 0.80002295072483916762f;
const float P4 = 0.087403248265958578062f;
const float P5 = 0.0020751409553756572917f;
const float P6 = 4.744820732427972462e-6f;
const float Q1 = 1.0f;
const float Q2 = 2.0795584360534589311f;
const float Q3 = 1.2499328117341603014f;
const float Q4 = 0.23668431621373705623f;
const float Q5 = 0.0120098270559197768f;
const float Q6 = 0.00010590620919921025259f;
float z;
int sgn;
T v = vi*2.3283064E-10F; // i.e. 2 power neg 32
sgn = (v >= 0.5f);
sgn = sgn - !sgn;
z = -logf(1.0f - (sgn * ((2.0f * v) - 1.0f)));
return sgn * z * CQPG(z) / CQQG(z);
}

};


struct toUniformFloatGold
{
   float operator()(unsigned int x)
    {
        return x*2.3283064E-10F;
    }
};


#endif
