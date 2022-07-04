//
//
//                                  HestonCF.cpp
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence 


#include <gold/HestonCF_gold.h>

typedef  std::complex<double> complex;

// cf page 5, An new look at the Heston Characteristic Function, del Bano Rollin et al

std::complex<double> Heston_cf_d(complex u, double a, double c, double rho)
{
   complex i(0.0,1.0);

   complex inner_val = (a- c*i*rho*u)*(a- c*i*rho*u) +c*c*(i*u+u*u);

   return sqrt(inner_val);
}

std::complex<double> Heston_cf_xi(complex u, double a, double c, double rho)
{
   complex i(0.0,1.0);

   complex inner_val = (a- c*i*rho*u);

   return inner_val;
}

complex Heston_cf_g(complex u, double a, double c, double rho)
{
   complex xi = Heston_cf_xi(u,a,c,rho);
   complex d = Heston_cf_d(u,a,c,rho);

   return (xi-d)/(xi+d);

}

// argument of middle exp in equation (6)
complex Heston_cf_term1a(complex u, double a, double b, double c, double rho, double t)
{
   complex xi = Heston_cf_xi(u,a,c,rho);
   complex d = Heston_cf_d(u,a,c,rho);

 
   complex innerA = (xi-d)*t;
 
   complex inner = a*b*(innerA)/(c*c);

   return inner;

}
// argument of middle exp in equation (6)
complex Heston_cf_term1b(complex u, double a, double b, double c, double rho, double t)
{
   complex d = Heston_cf_d(u,a,c,rho);

   complex g= Heston_cf_g(u,  a,  c,  rho);

   complex innerB = -2.0*log((1.0-g*exp(-d*t))/(1.0-g));

   complex inner = a*b*(innerB)/(c*c);

   return inner;

}

// argument of middle exp in equation (6)
complex Heston_cf_term1(complex u, double a, double b, double c, double rho, double t)
{
   complex xi = Heston_cf_xi(u,a,c,rho);
   complex d = Heston_cf_d(u,a,c,rho);

   complex g= Heston_cf_g(u,  a,  c,  rho);

   complex innerA = (xi-d)*t;
   complex innerB = -2.0*log((1.0-g*exp(-d*t))/(1.0-g));

   complex inner = a*b*(innerA+innerB)/(c*c);

   return inner;

}

// argument of final exp in equation (6)
complex Heston_cf_term2(complex u, double a, double b, double c, double rho, double t, double v0)
{
   complex xi = Heston_cf_xi(u,a,c,rho);
   complex d = Heston_cf_d(u,a,c,rho);

   complex g= Heston_cf_g(u,  a,  c, rho);

   complex innerA = (xi-d)*v0/(c*c);
   complex innerB = (1.0 - exp(-d*t))/(1.0-g*exp(-d*t));

   complex inner = innerA*innerB;

   return inner;

}

complex Heston_cf_log(complex u, double a, double b, double c, double rho, double t, double v0, double x0)
{
    complex i(0.0,1.0);
    complex t1 = i*x0*u;

    complex t2 = Heston_cf_term1( u,  a,  b,  c,  rho,  t);
    
    complex t3 = Heston_cf_term2( u,  a,  b,  c,  rho,  t,v0);

    complex res = t1+t2+t3;

    return res;


}
complex Heston_cf_zero_rho_approx_log(complex u, double a, double b, double c,  double t, double v0, double x0)
{
    complex i(0.0,1.0);
    complex t1 = i*x0*u;

    complex t2 = u*(-a*b*t/c) +a*a*b*t/(c*c) + 2 * log(2)*a*b/(c*c) - i * a*b*t/(2.0*c);
    
    complex t3 = v0*(-u/c+a/(c*c)-i*0.5/c);

    complex res = t1+t2+t3;

    return res;


}


complex Heston_cf_general_rho_approx_log(complex u, double a, double b, double c,  double rho, double t, double v0, double x0)
{
    complex i(0.0,1.0);
  
    complex t1 = i*x0*u;

    double rhop = sqrt(1-rho*rho);

    double alpha = u.real();
    double beta = u.imag();

    complex firstLargeTerm = -alpha*c*rhop-alpha*c*rho*i+a+c*rho*beta-i*c*0.5/rhop-i*beta*c*rhop+i*a*rho;
    complex firstLargeMult = (v0+a*b*t)*firstLargeTerm;

    complex logarg = 1.0 - rho*rho+rho*rhop*i;

    complex secondLargeTerm = a*b*(2.0* log(2) + 2.0*log(logarg));

    complex mainTerm = (firstLargeMult+secondLargeTerm)/(c*c);

    complex res = t1+mainTerm;

    return res;


}


complex Heston_cf_g_approx(complex u, double a, double b, double c,  double rho, double t, double v0, double x0)
{
    complex i(0.0,1.0);
  
    double rhop = sqrt(1-rho*rho);


    complex g= -1.0 + 2.0*rho*rho-2.0*rho*rhop*i;

    return g;


}




complex Heston_cf(complex u, double a, double b, double c, double rho, double t, double v0, double x0)
{
    
    complex res = exp(Heston_cf_log(u, a, b,  c, rho,  t,  v0, x0));

    return res;


}



 HestonCF::HestonCF(double a, double b, double c,  double rho, double t, double v0, double x0)
     : a_(a), b_(b), c_(c), rho_(rho), t_(t), v0_(v0), x0_(x0)
 {

 }


BlackScholesCF::BlackScholesCF(double sigma, double T, double x0)
    : multiplier_(-0.5*sigma*sigma*T), x0_(x0)
{
}


