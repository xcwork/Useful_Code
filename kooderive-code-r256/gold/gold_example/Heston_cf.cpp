//
//
//                                  HestonCF_example.cpp
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence 

#include <iostream>
#include <complex>
#include <gold/HestonCF_gold.h>
#include <gold/CosMethod_gold.h>
#include <gold/BSFormulas_gold.h>
#include <iomanip> 
#include <gold/VG_cf_gold.h>


typedef  std::complex<double> complex;



// this is intended to be comprehensive rather than efficient
void HestonCFDumper(double alpha0 // Real u
                    , double beta0 // Imaginary u
                    , int numberPoints
                    , double alphaStep
                    , double betaStep
                    ,  double a, double b, double c, double rho, double t, double v0, double x0)
{
    std::cout  <<  "alpha0," <<  alpha0 // Real u
        <<  ",beta0,"    <<  beta0 // Imaginary u
        <<  ",numberPoints,"    <<  numberPoints
        <<  ",alphaStep,"    <<  alphaStep
        <<  ",betaStep,"     <<  betaStep
        <<  ",a," <<  a <<  
        ",b," <<  b << 
        ",c," <<  c << 
        ",rho," <<  rho<<  
        ",t," <<  t << 
        ",v0," <<  v0 << 
        ",x0," <<   x0 << "\n";

    for (int i=0; i < numberPoints; ++i)
    {
        double alpha = alpha0+i * alphaStep;
        double beta = beta0+i * betaStep;

        complex u(alpha,beta);

        std::complex<double> d=Heston_cf_d(u,  a,  c,  rho);

        std::complex<double> xi = Heston_cf_xi( u,  a,  c,  rho);


        std::complex<double> g= Heston_cf_g( u,  a,  c,  rho);

        std::complex<double> t1 = Heston_cf_term1( u,  a,  b,  c,  rho,  t);
        std::complex<double> t1a = Heston_cf_term1a( u,  a,  b,  c,  rho,  t);
        std::complex<double> t1b = Heston_cf_term1b( u,  a,  b,  c,  rho,  t);


        std::complex<double> t2 = Heston_cf_term2( u,  a,  b,  c,  rho,  t,  v0);
        std::complex<double> cflog = Heston_cf_log( u,  a,  b,  c,  rho,  t,  v0,  x0);
        std::complex<double> cflogapprox = Heston_cf_general_rho_approx_log( u,  a,  b,  c, rho,  t,  v0,  x0);
   
        std::complex<double> gapprox = Heston_cf_g_approx( u,  a,  b,  c, rho,  t,  v0,  x0);


        std::complex<double> cf = Heston_cf( u,  a,  b,  c,  rho,  t,  v0,  x0);


        std::cout  <<  "alpha," <<  alpha // Real u
            <<  ",beta,"    <<  beta // Imaginary u
            << ",d," << d.real() <<"," << d.imag()<<","  
            << ",xi," << xi.real() <<"," << xi.imag()<<"," 
            << ", g, " << g.real() <<"," << g.imag()<<"," 
            << ", gapprox, " << gapprox.real() <<"," << gapprox.imag()<<"," 
            << ",t1," << t1.real() <<"," << t1.imag()<<"," 
          << ",t1a," << t1a.real() <<"," << t1a.imag()<<"," 
          << ",t1b," << t1b.real() <<"," << t1b.imag()<<"," 
            << ",t2," << t2.real() <<"," << t2.imag()<<"," 
            << ",cflog," << cflog.real() <<"," << cflog.imag()<<"," 
           << ",cflogapprox," << cflogapprox.real() <<"," << cflogapprox.imag()<<"," 
            << ",cf," << cf.real() <<"," << cf.imag()<<"," 
            << "\n";


    }

}

void DigitalOptionBSExample()
{

    double K1 = 0.8;
    double K2 = 1.1;
    double S0 = 1.05;
    double T = 2.0;
    double sigma = 0.1;
    double r=0.05;

    int N=1000;

    double df = exp(-r*T);

    double x0 = log(S0/df);

    BlackScholesCF cf( sigma, T, x0);

    double c = log(K1);
    double d = log(K2);

    double a =  -10.0;
    double b =  10.0;


   

    for (int i =1; i <=N; ++i)
    {

       double valueI = CosDigitalPriceZeroShift( a ,  b, i,  c,  d, df, cf);
       std::cout << i << "," << valueI << "\n"; 
    }

    double ddPrice = BlackScholesDoubleDigital(S0,r,0.0,T,sigma,K1,K2);
    
    std::cout << "\n\n BSDD , " << ddPrice << "\n\n";

   for (int i =1; i <=N; ++i)
    {

       double valueAD = CosAssetDigitalPriceZeroShift( a ,  b, i,  c,  b, df, cf);
       std::cout << i << "," << valueAD << "\n"; 
    }

    double asPrice = BlackScholesCall(S0,r,0.0,T,sigma,K1);
    asPrice += BlackScholesDigitalCall(S0,r,0.0,T,sigma,K1)*K1;

    std::cout << "\n BSDD , " << asPrice << "\n";

}

void OptionCosHestonExample(
                    //, double beta0 // Imaginary u
                    //, 
                    int numberPoints
                     ,  double a, double b,double c,  double rho, double T, double v0, 
                    double S0,
                    double K,
                    double r,
                    double q
                    )
{
    std::cout  
      //  <<  ",beta0,"    <<  beta0 // Imaginary u
        <<  ",numberPoints,"    <<  numberPoints
         <<  ",a," <<  a <<  
        ",b," <<  b << 
        ",c," <<  c << 
        ",rho," <<  rho<<  
        ",T," <<  T << 
        ",v0," <<  v0 << 
        ",S0," <<   S0 <<
        ",K," << K <<
        ",r," << r <<
        "\n";




    double df = exp(-r*T);

    double x0 = log(S0/df)-q*T;

    HestonCF cf(a,  b, c,   rho, T, v0, x0);


    double cc = log(K);
    

    double ac =  -15.0;
    double bc =  15.0;

    double dc=bc; // since pricing a call option
   

    for (int i =1; i <=numberPoints; ++i)
    {

  
       double valueDigCall = CosDigitalPriceZeroShift( ac ,  bc, i,  cc,  dc, df, cf);
       double valueAOrNCall = CosAssetDigitalPriceZeroShift( ac ,  bc, i,  cc,  bc, df, cf);

       std::cout << std::setprecision(15)<< i << ",call," <<valueDigCall << "," <<valueAOrNCall << ","<< valueAOrNCall-K*valueDigCall;

        double valueDigPut = CosDigitalPriceZeroShift( ac ,  bc, i,  ac,  cc, df, cf); // lower limit is density bottom, upper limit is strike
       double valueAOrNPut = CosAssetDigitalPriceZeroShift( ac ,  bc, i,  ac,  cc, df, cf);

       std::cout << std::setprecision(15) << ",put," <<valueDigPut << "," <<valueAOrNPut << ","<< -valueAOrNPut+K*valueDigPut ; 

       for (int j=0; j < 3;++j)
       {
           double shift = static_cast<double>(j);
     
            double aOrNShiftedCall = CosPricePowerDigitalShift(ac , // truncation lower limit for density
                                      bc, // truncation upper limit for density
                                     i,  // no of terms in the series 
                                     cc,  // lower digital limit in log space
                                     dc,// upper digital limit in log space
                                     1, // pay-off is e^{\lambda x} I_{c,d}(x) in log space
                                      shift, // power of numeraire 
                                     df, // discount factor 
                                     cf // functor for characteristic function
                                     );

              double digShiftedCall = CosPricePowerDigitalShift(ac , // truncation lower limit for density
                                      bc, // truncation upper limit for density
                                     i,  // no of terms in the series 
                                     cc,  // lower digital limit in log space
                                     dc,// upper digital limit in log space
                                     0, // pay-off is e^{\lambda x} I_{c,d}(x) in log space
                                      shift, // power of numeraire 
                                     df, // discount factor 
                                     cf // functor for characteristic function
                                     );

                     std::cout << std::setprecision(15) << ",shift call,"<< shift <<"," <<digShiftedCall << "," <<aOrNShiftedCall << ","<< aOrNShiftedCall-K*digShiftedCall ; 
    
             double aOrNShiftedPowerCall = CosPricePowerDigitalShift(ac , // truncation lower limit for density
                                      bc, // truncation upper limit for density
                                     i,  // no of terms in the series 
                                     cc,  // lower digital limit in log space
                                     dc,// upper digital limit in log space
                                     2, // pay-off is e^{\lambda x} I_{c,d}(x) in log space
                                      shift, // power of numeraire 
                                     df, // discount factor 
                                     cf // functor for characteristic function
                                     );
             std::cout << std::setprecision(15) << ",shift power 2 call,"<< shift <<"," <<aOrNShiftedCall <<"," ; 
    


       }
       std::cout << "\n";

  }

  

}


void OptionCosVGExample(
                    //, double beta0 // Imaginary u
                    //, 
                    int numberPoints,
                    int samples,
                      double theta, 
                     double nu, double T, double sigma, 
                    double S0,
                    double K,
                    double r,
                    double q
                    )
{
    std::cout  
      //  <<  ",beta0,"    <<  beta0 // Imaginary u
        <<  ",numberPoints,"    <<  numberPoints
         << ",nu," << nu <<
          ",sigma," << sigma <<
        ",theta," << theta <<
         ",T," <<  T << 
         ",S0," <<   S0 <<
        ",K," << K <<
        ",r," << r <<
        "\n";




    double df = exp(-r*T);

    double x0 = log(S0/df)-q*T;
     VarianceGammaCF cf( nu,  theta,  sigma,  T, x0);


    double cc = log(K);
    

    double ac =  -15.0;
    double bc =  15.0;

    double dc=bc; // since pricing a call option
   

    for (int i =1; i <=numberPoints; i=i+numberPoints/samples)
    {

  
       double valueDigCall = CosDigitalPriceZeroShift( ac ,  bc, i,  cc,  dc, df, cf);
       double valueAOrNCall = CosAssetDigitalPriceZeroShift( ac ,  bc, i,  cc,  bc, df, cf);

       std::cout << std::setprecision(15)<< i << ",call," <<valueDigCall << "," <<valueAOrNCall << ","<< valueAOrNCall-K*valueDigCall;

        double valueDigPut = CosDigitalPriceZeroShift( ac ,  bc, i,  ac,  cc, df, cf); // lower limit is density bottom, upper limit is strike
       double valueAOrNPut = CosAssetDigitalPriceZeroShift( ac ,  bc, i,  ac,  cc, df, cf);

       std::cout << std::setprecision(15) << ",put," <<valueDigPut << "," <<valueAOrNPut << ","<< -valueAOrNPut+K*valueDigPut ; 

       for (int j=0; j < 3;++j)
       {
           double shift = static_cast<double>(j);
     
            double aOrNShiftedCall = CosPricePowerDigitalShift(ac , // truncation lower limit for density
                                      bc, // truncation upper limit for density
                                     i,  // no of terms in the series 
                                     cc,  // lower digital limit in log space
                                     dc,// upper digital limit in log space
                                     1, // pay-off is e^{\lambda x} I_{c,d}(x) in log space
                                      shift, // power of numeraire 
                                     df, // discount factor 
                                     cf // functor for characteristic function
                                     );

              double digShiftedCall = CosPricePowerDigitalShift(ac , // truncation lower limit for density
                                      bc, // truncation upper limit for density
                                     i,  // no of terms in the series 
                                     cc,  // lower digital limit in log space
                                     dc,// upper digital limit in log space
                                     0, // pay-off is e^{\lambda x} I_{c,d}(x) in log space
                                      shift, // power of numeraire 
                                     df, // discount factor 
                                     cf // functor for characteristic function
                                     );

                     std::cout << std::setprecision(15) << ",shift call,"<< shift <<"," <<digShiftedCall << "," <<aOrNShiftedCall << ","<< aOrNShiftedCall-K*digShiftedCall ; 
    
             double aOrNShiftedPowerCall = CosPricePowerDigitalShift(ac , // truncation lower limit for density
                                      bc, // truncation upper limit for density
                                     i,  // no of terms in the series 
                                     cc,  // lower digital limit in log space
                                     dc,// upper digital limit in log space
                                     2, // pay-off is e^{\lambda x} I_{c,d}(x) in log space
                                      shift, // power of numeraire 
                                     df, // discount factor 
                                     cf // functor for characteristic function
                                     );
             std::cout << std::setprecision(15) << ",shift power 2 call,"<< shift <<"," <<aOrNShiftedCall <<"," ; 
    


       }
       std::cout << "\n";

  }

  

}
