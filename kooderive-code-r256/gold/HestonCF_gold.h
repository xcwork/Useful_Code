//
//
//                                  HestonCF.h
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence 

#ifndef HESTONCF_H
#define HESTONCF_H
#include <complex>

std::complex<double> Heston_cf_d(std::complex<double> u, double a, double c, double rho);

std::complex<double> Heston_cf_xi(std::complex<double> u, double a, double c, double rho);


std::complex<double> Heston_cf_g(std::complex<double> u, double a, double c, double rho);;


// argument of middle exp in equation (6)
std::complex<double> Heston_cf_term1(std::complex<double> u, double a, double b, double c, double rho, double t);

std::complex<double> Heston_cf_term1a(std::complex<double> u, double a, double b, double c, double rho, double t);
std::complex<double> Heston_cf_term1b(std::complex<double> u, double a, double b, double c, double rho, double t);

// argument of final exp in equation (6)
std::complex<double> Heston_cf_term2(std::complex<double> u, double a, double b, double c, double rho, double t, double v0);

std::complex<double> Heston_cf_log(std::complex<double> u, double a, double b, double c, double rho, double t, double v0, double x0);
std::complex<double> Heston_cf_zero_rho_approx_log(std::complex<double> u, double a, double b, double c,  double t, double v0, double x0);

std::complex<double> Heston_cf_general_rho_approx_log(std::complex<double> u, double a, double b, double c,  double rho, double t, double v0, double x0);

std::complex<double> Heston_cf_g_approx(std::complex<double> u, double a, double b, double c,  double rho, double t, double v0, double x0);

std::complex<double> Heston_cf(std::complex<double> u, double a, double b, double c, double rho, double t, double v0, double x0);

class HestonCF
{
public:
    HestonCF(double a, double b, double c,  double rho, double t, double v0, double x0);

    std::complex<double> operator()(std::complex<double>u) const
    {
       std::complex<double> hcf = Heston_cf(u, a_, b_, c_,  rho_, t_, v0_, x0_);  
       return hcf;
    }

private:
     double a_;
     double b_; 
     double c_;  
     double rho_; 
     double t_; 
     double v0_; 
     double x0_;
};

class BlackScholesCF
{
    public:
    BlackScholesCF(double sigma, double T, double x0);

    std::complex<double> operator()(std::complex<double>u) const
    {
       std::complex<double> i(0.0,1.0);
       std::complex<double> lcf = (u*u+i*u)*multiplier_;
       std::complex<double> cf = exp(lcf);
       std::complex<double> basePoint =exp(i*x0_*u);

       std::complex<double> res = cf*basePoint;

       return res;
    }

private:
    double multiplier_;
    double x0_;
 
};


#endif
