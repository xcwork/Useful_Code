#include <gold/VG_cf_gold.h>

std::complex<double>
 VG_cf(std::complex<double> u,std::complex<double> omegaT, double nu, double theta, double sigma,  double Tbynu,  double x0)
 {
     std::complex<double> i(0.0,1.0);
     
     std::complex<double> term0 = exp(i*x0*u);
     std::complex<double> term1 = exp(i*omegaT*u);
     std::complex<double> term2Denom = 1.0 - i * theta*nu*u + sigma*sigma*nu*0.5*u*u;
     std::complex<double> term2 = pow(term2Denom,-Tbynu);
     
     std::complex<double> res = term0*term1*term2;

     return res;

 }

VarianceGammaCF::VarianceGammaCF(double nu, double theta, double sigma,  double T, double x0) :
    nu_(nu),
    theta_(theta) ,
    sigma_(sigma),  
    Tbynu_(T/nu),
    omegaT_(-1.0),
    x0_(x0)
{
        
  //  std::complex<double> i(0.0,1.0);
    omegaT_=T*log( 1.0 - theta_*nu_ - sigma_*sigma_*0.5*nu)/nu_;
}
