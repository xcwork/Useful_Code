
//                                      normals_gold.cpp

/*

Copyright (c) 2012, 2014 Mark Joshi

released under the GPL v 3.0

adapted from code with the following license Copyright (C) 2000, 2001, 2002, 2003 RiskMap srl
Copyright (C) 2002, 2003 Ferdinando Ametrano
Copyright (C) 2008 StatPro Italia srl
Copyright (C) 2010 Kakhkhor Abdijalilov

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

#include <gold/math/Normals_gold.h>
#include <gold/math/error_function_gold.h>
#include <algorithm>
#include <gold/Errors.h>
#include <gold/BSFormulas_gold.h>
Realv normalDensity(Realv x)
{
    return M_SQRT_OF_ONE_OVER_2PI*std::exp(-0.5*x*x);
}



Realv cumulativeNormal(Realv x)
{
    ErrorFunction eFunction;

    double result = 0.5 * ( 1.0 + eFunction( x*M_SQRT_2 ) );

    return result;
}



Realv inverseCumulativeNormalDerivative(Realv x)
{
    return 1.0/( normalDensity(inverseCumulativeGold<Realv>()(static_cast<Realv>(x))));
}
namespace
{
    const Realv x_[] = {
        0.24840615,
        0.39233107,
        0.21141819,
        0.03324666,
        0.00082485334
    };

    const Realv y_[] = {
        0.10024215,
        0.48281397,
        1.06094980,
        1.77972940,
        2.66976040000
    };
}
Realv BivariateCumulativeNormal(Realv a, Realv b, Realv rho)
{
    if (rho< -1+2e-15)
    { //perfect negative correlation

        if (-b < a)
            return cumulativeNormal(a) - cumulativeNormal(-b);
        else 
            return 0.0;

    }
    if (rho > 1-2e-15) // perfect correlation
    {
        return cumulativeNormal(std::min(a,b));
    }

    Realv CumNormDistA = cumulativeNormal(a);
    Realv CumNormDistB = cumulativeNormal(b);
    Realv MaxCumNormDistAB = std::max(CumNormDistA, CumNormDistB);
    Realv MinCumNormDistAB = std::min(CumNormDistA, CumNormDistB);

    if (1.0-MaxCumNormDistAB<1e-15)
        return MinCumNormDistAB;

    if (MinCumNormDistAB<1e-15)
        return MinCumNormDistAB;

    Realv a1 = a / std::sqrt(2.0 * (1.0 - rho*rho));
    Realv b1 = b / std::sqrt(2.0 * (1.0 - rho*rho));

    Realv result=-1.0;

    if (a<=0.0 && b<=0 && rho<=0) 
    {
        Realv sum=0.0;
        for (int i=0; i<5; i++) 
        {
            for (int j=0;j<5; j++) 
            {
                sum += x_[i]*x_[j]*
                    std::exp(a1*(2.0*y_[i]-a1)+b1*(2.0*y_[j]-b1)
                    +2.0*rho*(y_[i]-a1)*(y_[j]-b1));
            }
        }
        result = std::sqrt(1.0 - rho*rho)/M_PI*sum;
    } 
    else 
        if (a<=0 && b>=0 && rho>=0) 
        {
            //        BivariateCumulativeNormalDistributionDr78 bivCumNormalDist(-rho_);
            result= CumNormDistA - BivariateCumulativeNormal(a, -b,-rho);
        } 
        else 
            if (a>=0.0 && b<=0.0 && rho>=0.0) 
            {
                result= CumNormDistB - BivariateCumulativeNormal(-a, b,-rho);
            } 
            else 
                if (a>=0.0 && b>=0.0 && rho<=0.0) {
                    result= CumNormDistA + CumNormDistB -1.0 + BivariateCumulativeNormal(-a, -b,rho);
                } else if (a*b*rho>0.0) 
                {
                    Realv rho1 = (rho*a-b)*(a>0.0 ? 1.0: -1.0)/
                        std::sqrt(a*a-2.0*rho*a*b+b*b);

                    Realv rho2 = (rho*b-a)*(b>0.0 ? 1.0: -1.0)/
                        std::sqrt(a*a-2.0*rho*a*b+b*b);

                    Realv delta = (1.0-(a>0.0 ? 1.0: -1.0)*(b>0.0 ? 1.0: -1.0))/4.0;

                    result= BivariateCumulativeNormal(a, 0.0,rho1) + BivariateCumulativeNormal(b, 0.0,rho2) - delta;
                } else 
                {
                    GenerateError("case not handled");
                }

                return result;
}

// Moments of a Truncated Bivariate Normal Distribution
// S. Rosenbaum
// Journal of the Royal Statistical Society. Series B (Methodological)
// Vol. 23, No. 2 (1961), pp. 405-408
Realv TruncatedFirstMomentBivariateNormal(Realv lowerFirstVariable, Realv lowerSecondVariable, Realv rho)
{
    
    Realv h = lowerFirstVariable;
    Realv k = lowerSecondVariable;


    if (rho > 1-2e-15)
    {
        Realv hstar = std::max(lowerFirstVariable,lowerSecondVariable);
        // perfect correlation
        return normalDensity(hstar);
    }
    if (rho < -1+2e-15)
    {
           // perfect anti-correlation
        if (-k < h)
            return 0.0;
        else
            return normalDensity(h) - normalDensity(-k);

    }

//    Realv L_hkrho =  BivariateCumulativeNormal(-h, -k,rho);

    Realv z= (k-rho*h)/sqrt(1-rho*rho);
    Realv zprime = -z;

    Realv w = (h-rho*k)/sqrt(1-rho*rho);
    Realv wprime = -w;

    Realv term1 = normalDensity(h)*cumulativeNormal(zprime);
    Realv term2 = normalDensity(k)*cumulativeNormal(wprime);
    return term1+rho*term2;

}
Realv ConditionalTruncatedFirstMomentBivariateNormal(Realv lowerFirstVariable, Realv lowerSecondVariable, Realv rho)
{
    Realv h = lowerFirstVariable;
    Realv k = lowerSecondVariable;

    Realv L_hkrho =  BivariateCumulativeNormal(-h, -k,rho);
    Realv integral = TruncatedFirstMomentBivariateNormal(lowerFirstVariable,  lowerSecondVariable,  rho);

    return integral/L_hkrho;

}
Realv LowerUpperTruncatedFirstMomentBivariateNormal(Realv lowerFirstVariable, Realv upperSecondVariable, Realv rho)
{
    Realv h = lowerFirstVariable;
    Realv k = upperSecondVariable;

    return TruncatedFirstMomentBivariateNormal(h,-k,-rho);

}
Realv UpperTruncatedFirstMomentBivariateNormal(Realv upperFirstVariable, Realv upperSecondVariable, Realv rho)
{
    Realv h = upperFirstVariable;
    Realv k = upperSecondVariable;

    Realv term1 = -normalDensity(k)*rho;
    Realv term2 = TruncatedFirstMomentBivariateNormal(h,-k,-rho);
    return term1-term2;

}


Realv UpperTruncatedFirstMomentBivariateNormal(Realv upperFirstVariable, Realv upperSecondVariable, 
                                               Realv mux, Realv sigmax, Realv muy, Realv sigmay, Realv sigmaxy)
{
    bool xConst = sigmax < 1e-15;
    bool yConst = sigmay < 1e-15;

    if (xConst && yConst)
        return (mux < upperFirstVariable) && (muy < upperSecondVariable) ? mux : 0.0;

    if (xConst)
        return mux < upperFirstVariable ? mux*cumulativeNormal((upperSecondVariable-muy)/sigmay): 0.0;
  
    
    Realv h = (upperFirstVariable-mux)/sigmax;
  

    if (yConst)
        return muy < upperSecondVariable ?  mux*cumulativeNormal(h)- sigmax*normalDensity(h)   : 0.0;

   
    Realv k = (upperSecondVariable-muy)/sigmay;

    Realv rho = sigmaxy/(sigmax*sigmay);

    Realv term1 = mux* BivariateCumulativeNormal(h,  k, rho);
    Realv term2 = sigmax*UpperTruncatedFirstMomentBivariateNormal(h, k, rho);

    return  term1+term2;

}


Realv LowerTruncatedFirstMomentBivariateNormal(Realv lowerFirstVariable, Realv lowerSecondVariable, 
                                               Realv mux, Realv sigmax, Realv muy, Realv sigmay, Realv sigmaxy)
{
   
    bool xConst = sigmax < 1e-15;
    bool yConst = sigmay < 1e-15;

    if (xConst && yConst)
        return (mux > lowerFirstVariable) && (muy > lowerSecondVariable) ? mux : 0.0;

    if (xConst)
        return mux > lowerFirstVariable ? mux*cumulativeNormal(-(lowerSecondVariable-muy)/sigmay): 0.0;
  
    Realv h = (lowerFirstVariable-mux)/sigmax;
 

    if (yConst)
        return muy > lowerSecondVariable ?  mux*cumulativeNormal(-h)+ sigmax*normalDensity(h)   : 0.0;

     Realv k = (lowerSecondVariable-muy)/sigmay;

    Realv rho = sigmaxy/(sigmax*sigmay);

    Realv term1 = mux* BivariateCumulativeNormal(-h, -k, rho);
    Realv term2 = sigmax*TruncatedFirstMomentBivariateNormal(h, k, rho);

    return  term1+term2;

}

Realv threeMax(int i,int j, int k)
{
      return std::max(i,std::max(j,k));
}

Realv expectedMaximumOfGaussianAndAConstant(Realv mu, Realv sigma, Realv c)
{
    if (sigma < 1e-15)
        return std::max(mu,c);

    Realv f = std::min(mu,c);
    Realv k = std::max(mu,c);

    return   k+  NormalBlackFromSd(f, k,sigma, 1.0);
}

Realv ExpectedMaximumOfTwoGausssiansAndAConstant(Realv mux, Realv muy, Realv c, Realv sigmax2, Realv sigmaxy, Realv sigmay2)
{

    Realv tolerance = 1E-10;

    Realv sigmax = sqrt(sigmax2);
    Realv sigmay = sqrt(sigmay2);

    if (sigmax < 1e-15) // x is effectively constant
    {
        Realv c2 = std::max(mux,c);
        return expectedMaximumOfGaussianAndAConstant(muy,sigmay,c2);
    }

    if (sigmay < 1e-15)
    {
        Realv c2 = std::max(muy,c);
        return expectedMaximumOfGaussianAndAConstant(mux,sigmax,c2);
    }


    // term1 is E of  X I_{X>Y} I_{X>c} = X I_{X-Y>0} I_{ X>c}

    Realv muxminusy = mux-muy;
    Realv sigmaxminusysquared = sigmax2 - 2*sigmaxy+sigmay2;
    Realv covxminusyx = sigmax2-sigmaxy;

    if (fabs(sigmaxminusysquared) < tolerance) // X and Y are effectively the same variable up to a constant
    {
        if (fabs(sigmax) < tolerance) // X and Y are constant
            return std::max(c,std::max(mux,muy));
        else // effectively one-d
        {
            
            Realv mu = std::max(mux,muy);

            Realv f = std::min(mu,c);
            Realv k = std::max(mu,c);

            Realv correction = NormalBlackFromSd(f, k,sqrt(sigmax2), 1.0);
            return correction+k;
        }
    }
    else
    {
        if (fabs(sigmax) < tolerance) // x is constant but y is not
        {
            Realv mu = std::max(mux,muy);

            Realv f = std::min(mu,c);
            Realv k = std::max(mu,c);

            Realv correction = NormalBlackFromSd(f, k,sqrt(sigmay2), 1.0);
            return correction+k;
        }
        else
        {
        Realv term1 = LowerTruncatedFirstMomentBivariateNormal(c,0.0,mux,sqrt(sigmax2),muxminusy,sqrt(sigmaxminusysquared),covxminusyx);

        // term2 is E of  Y I_{Y>X} I_{Y>c} = Y I_{Y-X>0} I_{ Y>c}
        Realv covyminusxy = sigmay2-sigmaxy;
        Realv term2 = LowerTruncatedFirstMomentBivariateNormal(c,0.0,muy,sqrt(sigmay2),-muxminusy,sqrt(sigmaxminusysquared),covyminusxy);

        // term3 is C P ( X<c, Y < c)


        Realv a = (c-mux)/sigmax;
        Realv b = (c-muy)/sigmay;
        Realv rho = sigmaxy/(sigmax*sigmay);
        Realv term3;

        if (rho > 1.0 - tolerance)
        {
            // perfect correlation
            term3 = c*cumulativeNormal(std::min(a,b));
        }
        else 
             term3 = c*BivariateCumulativeNormal(a, b,rho);

        return term1+term2+term3;
        }
    }
}

Realv ExpectedMaximumOfThreeIndependentGausssians(Realv mux, Realv muy, Realv muz, Realv sigmax2, Realv sigmay2, Realv sigmaz2)
{
    Realv sigmaprimex2= sigmax2+sigmaz2;
    Realv sigmaprimey2= sigmay2+sigmaz2;
    Realv cov = sigmaz2;
    return  ExpectedMaximumOfTwoGausssiansAndAConstant( mux,  muy,  muz, sigmaprimex2, cov, sigmaprimey2);
}
