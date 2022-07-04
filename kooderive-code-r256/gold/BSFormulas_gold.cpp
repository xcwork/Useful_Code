//
//
//                                      BSFormulas_gold.cpp
//
//
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3
#include <gold/BSFormulas_gold.h>
#include <gold/math/normals_gold.h>
#include <gold/Errors.h>

Realv Compute_d2(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv SigmaRootT = Sigma*sqrt(T);
    Realv d2 = ((r-d-0.5*Sigma*Sigma)*T + log(S0/K) )/SigmaRootT;
    return d2;
}

Realv Compute_d1(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv SigmaRootT = Sigma*sqrt(T);
    Realv d1 = ((r-d+0.5*Sigma*Sigma)*T + log(S0/K) )/SigmaRootT;
    return d1;
}

void Compute_d1d2(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K,Realv& d1, Realv& d2)
{
    Realv SigmaRootT = Sigma*sqrt(T);
    d2 = ((r-d-0.5*Sigma*Sigma)*T + log(S0/K) )/SigmaRootT;
    d1 = d2 + SigmaRootT;

}

Realv BlackScholesCall(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv d1,d2;
    Compute_d1d2( S0,  r, d,  T,  Sigma,  K,d1,d2);
    return S0*exp(-d*T)*cumulativeNormal(d1) - K*exp(-r*T)*cumulativeNormal(d2);
}

Realv BlackScholesPut(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv d1,d2;
    Compute_d1d2( S0,  r, d,  T,  Sigma,  K,d1,d2);
    return -S0*exp(-d*T)*cumulativeNormal(-d1) + K*exp(-r*T)*cumulativeNormal(-d2);
}


Realv BlackScholesDigitalCall(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv d2 =  Compute_d2( S0,  r, d,  T,  Sigma,  K);
    return exp(-r*T)*cumulativeNormal(d2) ;
}

Realv BlackScholesDigitalPut(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv d2 =  Compute_d2( S0,  r, d,  T,  Sigma,  K);
    return exp(-r*T)*(1.0-cumulativeNormal(d2)) ;
}

Realv BlackScholesCallDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv d1 =  Compute_d1( S0,  r, d,  T,  Sigma,  K);
    return exp(-d*T)*cumulativeNormal(d1);
}

Realv BlackScholesPutDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv d1 =  Compute_d1( S0,  r, d,  T,  Sigma,  K);
    return exp(-d*T)*(cumulativeNormal(d1)-1.0);
}
Realv BlackScholesDigitalCallDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv d2 =  Compute_d2( S0,  r, d,  T,  Sigma,  K);
    Realv SigmaRootT = Sigma*sqrt(T);
    return exp(-r*T)*normalDensity(d2)/(S0*SigmaRootT);
}
Realv BlackScholesDigitalPutDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    return - BlackScholesDigitalCallDelta(S0,r,d,T,Sigma,K);
}

Realv BlackScholesCallVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    Realv d1 =  Compute_d1( S0,  r, d,  T,  Sigma,  K);
    return S0*sqrt(T)*exp(-d*T)*normalDensity(d1);
}

Realv BlackScholesPutVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    return BlackScholesCallVega(S0, r,  d, T,  Sigma,  K);
}

Realv BlackScholesDigitalCallVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
      Realv d2 =  Compute_d2( S0,  r, d,  T,  Sigma,  K);
      return -exp(-r*T)*normalDensity(d2)*(sqrt(T)+d2/Sigma);
}

Realv BlackScholesDigitalPutVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K)
{
    return - BlackScholesDigitalCallVega( S0,  r, d,  T, Sigma, K);
}


Realv BlackScholesPowerCall(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K, int Power)
{
    if (Power ==0)
        return BlackScholesDigitalCall(S0,r,d,T,Sigma,K);

    if (Power ==1)
        return BlackScholesCall(S0,r,d,T,Sigma,K);

    GenerateError("Power options are not yet fully implemented.");

    
    return 0;
}
Realv BlackScholesPowerCallDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K, int Power)
{
    if (Power ==0)
        return BlackScholesDigitalCallDelta(S0,r,d,T,Sigma,K);

    if (Power ==1)
        return BlackScholesCallDelta(S0,r,d,T,Sigma,K);

    GenerateError("Power options are not yet fully implemented.");

    return 0;

}


Realv BlackScholesPowerCallVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K, int Power)
{
    if (Power ==0)
        return BlackScholesDigitalCallVega(S0,r,d,T,Sigma,K);

    if (Power ==1)
        return BlackScholesCallVega(S0,r,d,T,Sigma,K);

    GenerateError("Power options are not yet fully implemented.");

    
    return 0;

}

Realv BlackScholesDoubleDigital(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K1, Realv K2)
{
    return BlackScholesDigitalCall(S0,r,d,T,Sigma,K1)-BlackScholesDigitalCall(S0,r,d,T,Sigma,K2);
}

Realv BlackScholesDoubleDigitalDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K1, Realv K2)
{
    return BlackScholesDigitalCallDelta(S0,r,d,T,Sigma,K1)-BlackScholesDigitalCallDelta(S0,r,d,T,Sigma,K2);
}

Realv BlackScholesDoubleDigitalVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K1, Realv K2)
{
    return BlackScholesDigitalCallVega(S0,r,d,T,Sigma,K1)-BlackScholesDigitalCallVega(S0,r,d,T,Sigma,K2);

}


Realv NormalBlackFromSd(Realv f0, Realv Strike,Realv Sd, Realv Annuity)
{
      double z = (Strike-f0)/Sd; // normal draw to get in the money

      double pInTheMoney = cumulativeNormal(-z);

      double term2 = (f0-Strike)*pInTheMoney;
      double term1 = Sd*normalDensity(z);
      return term1+term2;
}
Realv MargrabeBlackScholes(Realv S1, Realv S2, Realv T, Realv rho, Realv sigma1, Realv sigma2,Realv d1, Realv d2)
{
       Realv sigmabar = sqrt(sigma1*sigma1-2*rho*sigma1*sigma2+sigma2*sigma2);


       return BlackScholesCall(S2*exp(-d2*T),0,0,T,sigmabar,S1*exp(-d1*T));
}

// See Haug 2d Edition page 211
double BSCallTwoMax(double S1 , double S2 , double K , double T, double r , 
       double q1 ,double q2 ,double v1 ,double v2 , double rho )
{
    double b1 = r-q1;
    double b2 = r-q2;
    double v ,rho1 , rho2 , d , y1 , y2 ;
    
    v = sqrt(v1 *v1+ v2 *v2 - 2 * rho * v1 * v2);
    rho1 = (v1 - rho * v2) / v;
    rho2 = (v2 - rho * v1) / v;
    d = (log(S1 / S2) + (b1 - b2 + v*v / 2) * T) / (v * sqrt(T));
    y1 = (log(S1 / K) + (b1 + v1*v1 / 2) * T) / (v1 * sqrt(T));
    y2 = (log(S2 / K) + (b2 + v2*v2 / 2) * T) / (v2 * sqrt(T));
  
    return S1 * exp((b1 - r) * T) * BivariateCumulativeNormal(y1, d, rho1) 
        + S2 * exp((b2 - r) * T) * BivariateCumulativeNormal(y2, -d + v * sqrt(T), rho2) 
        - K * exp(-r * T) * (1 - BivariateCumulativeNormal(-y1 + v1 * sqrt(T), -y2 + v2 * sqrt(T), rho));
}

// See Haug 2d Edition page 211
double BSCallTwoMaxDrezner(double S1 , double S2 , double K , double T, double r , 
       double q1 ,double q2 ,double v1 ,double v2 , double rho )
{
    double b1 = r-q1;
    double b2 = r-q2;
    double v ,rho1 , rho2 , d , y1 , y2 ;
    
    v = sqrt(v1 *v1+ v2 *v2 - 2 * rho * v1 * v2);
    rho1 = (v1 - rho * v2) / v;
    rho2 = (v2 - rho * v1) / v;
    d = (log(S1 / S2) + (b1 - b2 + v*v / 2) * T) / (v * sqrt(T));
    y1 = (log(S1 / K) + (b1 + v1*v1 / 2) * T) / (v1 * sqrt(T));
    y2 = (log(S2 / K) + (b2 + v2*v2 / 2) * T) / (v2 * sqrt(T));
  
    return S1 * exp((b1 - r) * T) * BivariateCumulativeNormalDrezner(y1, d, rho1) 
        + S2 * exp((b2 - r) * T) * BivariateCumulativeNormalDrezner(y2, -d + v * sqrt(T), rho2) 
        - K * exp(-r * T) * (1 - BivariateCumulativeNormalDrezner(-y1 + v1 * sqrt(T), -y2 + v2 * sqrt(T), rho));
}
