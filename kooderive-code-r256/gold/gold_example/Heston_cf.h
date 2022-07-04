//
//
//                                  HestonCF_example.h
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence 

#ifndef HESTON_CF_EXAMPLE
#define HESTON_CF_EXAMPLE


// this is intended to be comprehensive rather than efficient
void HestonCFDumper(double alpha0 // Real u
                    , double beta0 // Imaginary u
                    , int numberPoints
                    , double alphaStep
                    , double betaStep
                    ,  double a, double b, double c, double rho, double t, double v0, double x0);
void DigitalOptionBSExample();
void OptionCosHestonExample(
                    //, double beta0 // Imaginary u
                    //, 
                    int numberPoints
                     ,  double a, double b,double c,  double rho, double T, double v0, 
                    double S0,
                    double K,
                    double r,
                    double q
                    );


void OptionCosVGExample(
                    //, double beta0 // Imaginary u
                    //, 
                    int numberPoints,
                    int samples
                     ,  double theta, 
                     double nu, double T, double sigma, 
                    double S0,
                    double K,
                    double r,
                    double q
                    );

#endif
