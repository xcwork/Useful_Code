//
//
//                                  CosMethod_gold.cpp
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence 

/*
Formula for 
\int_{c}^{d} e^{\lambda y} \cos(\gamma y + \beta) dy
*/
#include <cmath>

double GetCosCoefficient(double c, double d, double lambda, double gamma, double beta)
{

    double scale = gamma*gamma+ lambda*lambda;

    if (scale < 1e-12)
    {
        // gamma = lambda =0.0

        return (d-c)*cos(beta);

    }
    else
    {

        double eld = exp(lambda*d);
        double elc = exp(lambda*c);

        double gammadplusbeta = gamma*d+beta;
        double gammacplusbeta = gamma*c + beta;

        double Iscaled = gamma*eld*sin(gammadplusbeta) -gamma*elc*sin(gammacplusbeta) + lambda*eld*cos(gammadplusbeta) - lambda*elc*cos(gammacplusbeta);

        return Iscaled/scale;
    }
}
