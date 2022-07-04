//
//
//                                  VG_cf_gold.h
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence 
// see page 245, More Mathematical Finance by Mark Joshi, PWP 2011
#ifndef VGCF_H
#define VGCF_H
#include <complex>

std::complex<double>
     VG_cf(std::complex<double> u, std::complex<double> omegaT, double nu, double theta, double sigma,  double Tbynu,  double x0);

class VarianceGammaCF
{
public:
    VarianceGammaCF(double nu, double theta, double sigma,  double T, double x0);

    std::complex<double> operator()(std::complex<double>u) const
    {
       std::complex<double> hcf = VG_cf(u, omegaT_,nu_, theta_, sigma_,  Tbynu_,  x0_);  
       return hcf;
    }

private:
    double nu_;
    double theta_; 
    double sigma_;  
    double Tbynu_;
    std::complex<double> omegaT_;
    double x0_;

};


#endif
