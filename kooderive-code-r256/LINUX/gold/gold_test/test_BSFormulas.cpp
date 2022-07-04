//
//
//                                             test_BS_formulas.cpp
//
//

//
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3
/*
Test BS formulas against Monte Carlo
Test BS Greek formulas against Finite Differencing
*/

#include <gold/BSFormulas_gold.h>
#include <gold/Mersenne_gold.h>

#include <gold/math/Normals_gold.h>
#include <gold/MonteCarloStatistics_concrete_gold.h>
#include <iostream>



int BSFormulasTest(bool verbose)
{
    if (verbose)
        std::cout << "testing BS formulas by MC\n";

    Realv tolerance = 3.5;

    int result=5;

    Realv S0=100;
    Realv K=110;
    Realv K2 = 120;
    Realv r =0.05;
    Realv d=0.02;
    Realv Sigma = 0.2;
    Realv T = 2.5;

    Realv  s= Sigma*sqrt(T);
    Realv drift = (r- d- 0.5*Sigma*Sigma)*T;
    Realv median = S0*exp(drift);
    Realv df = exp(-r*T);

    int paths =10000000;

#ifdef _DEBUG
    paths = 10000;

#endif

    std::vector<std::string> names(5);
    names[0] = "call";
    names[1] = "put";
    names[2] = "digitalcall";
    names[3] = "digitalput";
    names[4] = "doubledigital";

    MonteCarloStatisticsSimple stats(5,names);

    std::vector<Realv> results(5);
    MersenneTwisterUniformRng rng(1); // seed =1

    for (int i=0; i < paths; ++i)
    {
        Realv z = inverseCumulativeGold<double>()(rng.next());

        Realv ST = median*exp(s*z);

        results[0] = std::max<Realv>(ST-K,0)*df;
        results[1] = std::max<Realv>(K-ST,0)*df;
        results[2] = ST > K ? df : 0.0;
        results[3] = ST < K ? df : 0.0;
        results[4] = ( ST > K && ST < K2) ? df : 0.0;

        stats.AddDataVector(results);

    }

    Realv BSCallprice = BlackScholesCall( S0,  r,  d,  T, Sigma, K);
    Realv BSPutprice = BlackScholesPut( S0,  r,  d,  T, Sigma, K);
    Realv BSDigitalCallprice = BlackScholesDigitalCall( S0,  r,  d,  T, Sigma, K);
    Realv BSDigitalPutprice = BlackScholesDigitalPut( S0,  r,  d,  T, Sigma, K);
    Realv BSDoubleDigitalprice = BlackScholesDoubleDigital( S0,  r,  d,  T, Sigma, K,K2);

    std::vector<std::vector<Realv> > MCres(stats.GetStatistics());

    Realv callError = (MCres[0][0] - BSCallprice)/MCres[1][0];
    Realv putError = (MCres[0][1] - BSPutprice)/MCres[1][1];
    Realv digitalCallError = (MCres[0][2] - BSDigitalCallprice)/MCres[1][2];
    Realv digitalPutError = (MCres[0][3] - BSDigitalPutprice)/MCres[1][3];
    Realv doubleDigitalError = (MCres[0][4] - BSDoubleDigitalprice)/MCres[1][4];


    if (fabs(callError) < tolerance)
        --result;
    else
        std::cout << " Call formula test against MC failed " << BSCallprice << " " << MCres[0][0] << " " << MCres[1][0] << "\n";

    if (fabs(putError) < tolerance)
        --result;
    else
        std::cout << " put formula test against MC failed " << BSPutprice << " " << MCres[0][1] << " " << MCres[1][1] << "\n";

    if (fabs(digitalCallError) < tolerance)
        --result;
    else
        std::cout << " digital call formula test against MC failed " << BSDigitalCallprice << " " << MCres[0][2] << " " << MCres[1][2] << "\n";

    if (fabs(digitalPutError) < tolerance)
        --result;
    else
        std::cout << " digital put formula test against MC failed " << BSDigitalPutprice << " " << MCres[0][3] << " " << MCres[1][3] << "\n";

    if (fabs(doubleDigitalError) < tolerance)
        --result;
    else
        std::cout << " double digital  formula test against MC failed " << BSDoubleDigitalprice << " " << MCres[0][4] << " " << MCres[1][4] << "\n";

    if (verbose && result ==0)
        std::cout << "test passed\n";


    return result;

}


int TestBSGreekFormulas(bool verbose)
{

    if (verbose)
        std::cout << " testing Greek BS formulas ";

    Realv tolerance = 1E-6;
    Realv eps = 1E-4;

    int result=14;

    Realv S0=100;
    Realv K=110;
    Realv K2 = 120;
    Realv r =0.05;
    Realv d=0.02;
    Realv Sigma = 0.2;
    Realv T = 2.5;

    Realv bscallDelta = BlackScholesCallDelta(S0,  r,  d, T,  Sigma, K);
    Realv bscallDeltaFD = (BlackScholesCall(S0+eps,  r,  d, T,  Sigma, K) - BlackScholesCall(S0-eps,  r,  d, T,  Sigma, K) )/(2.0*eps) ;
    if (fabs(bscallDelta -bscallDeltaFD)> tolerance)
        std::cout << "BS Call Delta wrong " << bscallDelta << " " << bscallDeltaFD << "\n";
    else
        --result;

    Realv bsputDelta = BlackScholesPutDelta(S0,  r,  d, T,  Sigma, K);
    Realv bsputDeltaFD = (BlackScholesPut(S0+eps,  r,  d, T,  Sigma, K) - BlackScholesPut(S0-eps,  r,  d, T,  Sigma, K) )/(2.0*eps) ;
    if (fabs(bsputDelta -bsputDeltaFD)> tolerance)
        std::cout << "BS put Delta wrong " << bsputDelta << " " << bsputDeltaFD << "\n";
    else
        --result;


    Realv bsDcallDelta = BlackScholesDigitalCallDelta(S0,  r,  d, T,  Sigma, K);
    Realv bsDcallDeltaFD = (BlackScholesDigitalCall(S0+eps,  r,  d, T,  Sigma, K) - BlackScholesDigitalCall(S0-eps,  r,  d, T,  Sigma, K) )/(2.0*eps) ;
    if (fabs(bsDcallDelta -bsDcallDeltaFD)> tolerance)
        std::cout << "BS D Call Delta wrong " << bsDcallDelta << " " << bsDcallDeltaFD << "\n";
    else
        --result;


    Realv bsDputDelta = BlackScholesDigitalPutDelta(S0,  r,  d, T,  Sigma, K);
    Realv bsDputDeltaFD = (BlackScholesDigitalPut(S0+eps,  r,  d, T,  Sigma, K) - BlackScholesDigitalPut(S0-eps,  r,  d, T,  Sigma, K) )/(2.0*eps) ;
    if (fabs(bsDputDelta -bsDputDeltaFD)> tolerance)
        std::cout << "BS D put Delta wrong " << bsDputDelta << " " << bsDputDeltaFD << "\n";
    else
        --result;


    Realv bscallVega = BlackScholesCallVega(S0,  r,  d, T,  Sigma, K);
    Realv bscallVegaFD = (BlackScholesCall(S0,  r,  d, T,  Sigma+eps, K) - BlackScholesCall(S0,  r,  d, T,  Sigma-eps, K) )/(2.0*eps) ;
    if (fabs(bscallVega -bscallVegaFD)> tolerance)
        std::cout << "BS Call Vega wrong " << bscallVega << " " << bscallVegaFD << "\n";
    else
        --result;

    Realv bsputVega = BlackScholesPutVega(S0,  r,  d, T,  Sigma, K);
    Realv bsputVegaFD = (BlackScholesPut(S0,  r,  d, T,  Sigma+eps, K) - BlackScholesPut(S0,  r,  d, T,  Sigma-eps, K) )/(2.0*eps) ;
    if (fabs(bsputVega -bsputVegaFD)> tolerance)
        std::cout << "BS Put Vega wrong " << bsputVega << " " << bsputVegaFD << "\n";
    else
        --result;

    Realv bsDcallVega = BlackScholesDigitalCallVega(S0,  r,  d, T,  Sigma, K);
    Realv bsDcallVegaFD = (BlackScholesDigitalCall(S0,  r,  d, T,  Sigma+eps, K) - BlackScholesDigitalCall(S0,  r,  d, T,  Sigma-eps, K) )/(2.0*eps) ;
    if (fabs(bsDcallVega -bsDcallVegaFD)> tolerance)
        std::cout << "BS D Call Vega wrong " << bsDcallVega << " " << bsDcallVegaFD << "\n";
    else
        --result;

    Realv bsDputVega = BlackScholesDigitalPutVega(S0,  r,  d, T,  Sigma, K);
    Realv bsDputVegaFD = (BlackScholesDigitalPut(S0,  r,  d, T,  Sigma+eps, K) - BlackScholesDigitalPut(S0,  r,  d, T,  Sigma-eps, K) )/(2.0*eps) ;
    if (fabs(bsDputVega -bsDputVegaFD)> tolerance)
        std::cout << "BS D put Vega wrong " << bsDputVega << " " << bsDputVegaFD << "\n";
    else
        --result;

    for (int power =0; power < 2; ++power)
    {

        Realv bsPowerCallDelta = BlackScholesPowerCallDelta(S0,  r,  d, T,  Sigma, K,power);
        Realv bsPowerCallDeltaFD = (BlackScholesPowerCall(S0+eps,  r,  d, T,  Sigma, K,power) - BlackScholesPowerCall(S0-eps,  r,  d, T,  Sigma, K,power) )/(2.0*eps) ;
        if (fabs(bsPowerCallDelta -bsPowerCallDeltaFD)> tolerance)
            std::cout << "BS PowerCall Delta wrong " << bsPowerCallDelta << " " << bsPowerCallDeltaFD << "\n";
        else
            --result;
    }

    for (int power =0; power < 2; ++power)
    {

        Realv bsPowerCallVega = BlackScholesPowerCallVega(S0,  r,  d, T,  Sigma, K,power);
        Realv bsPowerCallVegaFD = (BlackScholesPowerCall(S0,  r,  d, T,  Sigma+eps, K,power) - BlackScholesPowerCall(S0,  r,  d, T,  Sigma-eps, K,power) )/(2.0*eps) ;
        if (fabs(bsPowerCallVega -bsPowerCallVegaFD)> tolerance)
            std::cout << "BS PowerCall Vega wrong " << bsPowerCallVega << " " << bsPowerCallVegaFD << "\n";
        else
            --result;
    }



    Realv bsDoubleDigitalDelta = BlackScholesDoubleDigitalDelta(S0,  r,  d, T,  Sigma, K,K2);
    Realv bsDoubleDigitalDeltaFD = (BlackScholesDoubleDigital(S0+eps,  r,  d, T,  Sigma, K,K2) - BlackScholesDoubleDigital(S0-eps,  r,  d, T,  Sigma, K,K2) )/(2.0*eps) ;
    if (fabs(bsDoubleDigitalDelta -bsDoubleDigitalDeltaFD)> tolerance)
        std::cout << "BS Double digital Delta wrong " << bsDoubleDigitalDelta << " " << bsDoubleDigitalDeltaFD << "\n";
    else
        --result;


    Realv bsDoubleDigitalVega = BlackScholesDoubleDigitalVega(S0,  r,  d, T,  Sigma, K,K2);
    Realv bsDoubleDigitalVegaFD = (BlackScholesDoubleDigital(S0,  r,  d, T,  Sigma+eps, K,K2) - BlackScholesDoubleDigital(S0,  r,  d, T,  Sigma-eps, K,K2) )/(2.0*eps) ;
    if (fabs(bsDoubleDigitalVega -bsDoubleDigitalVegaFD)> tolerance)
        std::cout << "BS Double digital Vega wrong " << bsDoubleDigitalVega << " " << bsDoubleDigitalVegaFD << "\n";
    else
        --result;


    if (result>0)
        std::cout << result << " fails\n";
    else
        if (verbose)
            std::cout << " test passed " << "\n";

    return result;


}

int NormalFormulasTest(bool verbose)
{
    if (verbose)
        std::cout << "testing Normal formulas by MC\n";

    Realv tolerance = 3.5;

    int result=1;

    Realv S0=100;
    Realv K=110;
    Realv Sigma = 10;
    Realv T = 1;

    Realv  s= Sigma*sqrt(T);

    int paths =10000000;

#ifdef _DEBUG
    paths = 10000;

#endif

    std::vector<std::string> names(1);
    names[0] = "call";
    MonteCarloStatisticsSimple stats(1,names);

    std::vector<Realv> results(1);
    MersenneTwisterUniformRng rng(1); // seed =1

    for (int i=0; i < paths; ++i)
    {
        Realv z = inverseCumulativeGold<double>()(rng.next());

        Realv ST = S0+Sigma*z;

        results[0] = std::max<Realv>(ST-K,0);

        stats.AddDataVector(results);

    }

    Realv NormalCallprice = NormalBlackFromSd(S0,K,Sigma,1.0);

    std::vector<std::vector<Realv> > MCres(stats.GetStatistics());

    Realv callError = (MCres[0][0] - NormalCallprice)/MCres[1][0];


    if (fabs(callError) < tolerance)
        --result;
    else
        std::cout << " Call formula test for normal against MC failed " << NormalCallprice << " " << MCres[0][0] << " " << MCres[1][0] << "\n";


    if (verbose && result ==0)
        std::cout << "test passed\n";


    return result;

}

int TestBivariateNormals(bool verbose)
{
    if (verbose)
        std::cout << "testing TestBivariateNormals formulas by MC\n";

    Realv tolerance = 3.5;

    int cases=7;

    double h =0.1;
    double k =-0.2;

    double mux=0.1;
    double muy=-0.5;
   
    double c =0.15;


    int rhosize = 5;
    double rhos[]={-1.0,-0.5,0,0.25,1.0};

    int hsize =3;
    double hs[] = {-1,0.1,1};

    int sigmaxsize=3;
    int sigmaysize=2;

    double sigmaxs[]={0.0, 0.723, 1.5};
    double sigmays[]={0.0, 0.723};


    int paths = 100000;
    int result=rhosize*cases*hsize*sigmaxsize*sigmaysize;

    for (int sigmaxtest=0; sigmaxtest < sigmaxsize; ++sigmaxtest)
           for (int sigmaytest=0; sigmaytest < sigmaysize; ++sigmaytest)
    for (int htest = 0; htest< hsize;++htest)
    {
        double sigmax = sigmaxs[sigmaxtest];
        double sigmay = sigmays[sigmaytest];

        double h = hs[htest];
        for (int test=0; test<rhosize; ++test)
        {
            double rho = rhos[test];
            double rho2 = sqrt(1-rho*rho);

            double sigmaxy = rho*sigmax*sigmay;

            std::vector<std::string> names(cases);
            names[0] = "cumulative";
            names[1] = "truncated below";
            names[2] = "truncated above and below";
            names[3] = "truncated above";
            names[4] = "truncated above general case";
            names[5] = "truncated below general case";
            names[6] = "max general case";


            MonteCarloStatisticsSimple stats(cases,names);

            std::vector<Realv> results(cases);
            MersenneTwisterUniformRng rng(1); // seed =1

            for (int i=0; i < paths; ++i)
            {
                Realv z1 = inverseCumulativeGold<double>()(rng.next());
                Realv z2 = inverseCumulativeGold<double>()(rng.next());

                Realv w1= z1;
                Realv w2 = rho*z1+rho2*z2;

                Realv x = mux+sigmax*w1;
                Realv y = muy+sigmay*w2;

                double inDomain = w1 < h && w2 < k ? 1.0 : 0.0;
                double m1 = w1 > h && w2 > k ? w1 : 0.0;
                double m2= w1 > h && w2 < k ? w1 :0.0;
                double m3= w1 < h && w2 < k ? w1 :0.0;
                double m4 = x < h && y < k ? x :0.0;
                double m5 = x > h && y > k ? x :0.0;

                double maxv =std::max( std::max(x,y),c);

                results[0] = inDomain;
                results[1] = m1;
                results[2] = m2;
                results[3] = m3;
                results[4] = m4;
                results[5] = m5;
                results[6] = maxv;

                stats.AddDataVector(results);

            }


            Realv bivAnalytic  = BivariateCumulativeNormal(h,k,rho);
            Realv truncNormalBiv = TruncatedFirstMomentBivariateNormal(h,k,rho);
            Realv truncNormalBiv2 =  LowerUpperTruncatedFirstMomentBivariateNormal(h,k,rho);
            Realv truncNormalBiv3 =  UpperTruncatedFirstMomentBivariateNormal(h,k,rho);
            Realv truncNormalBiv4=  UpperTruncatedFirstMomentBivariateNormal(h, k, 
                mux,  sigmax,  muy,  sigmay,  sigmaxy);
            Realv truncNormalBiv5=  LowerTruncatedFirstMomentBivariateNormal(h, k, 
                mux,  sigmax,  muy,  sigmay,  sigmaxy);


            Realv  expectedMax= ExpectedMaximumOfTwoGausssiansAndAConstant(mux, muy, c, sigmax*sigmax,  sigmax*sigmay*rho, sigmay*sigmay);

      
            std::vector<std::vector<Realv> > MCres(stats.GetStatistics());

            Realv bivAnalyticError = (MCres[0][0] - bivAnalytic)/(MCres[1][0]+1e-12);
            Realv truncNormalBivError = (MCres[0][1] - truncNormalBiv)/(MCres[1][1]+1e-12);
            Realv truncNormalBivError2 = (MCres[0][2] - truncNormalBiv2)/(MCres[1][2]+1e-12);
            Realv truncNormalBivError3 = (MCres[0][3] - truncNormalBiv3)/(MCres[1][3]+1e-12);
            Realv truncNormalBivError4 = (MCres[0][4] - truncNormalBiv4)/(MCres[1][4]+1e-12);
            Realv truncNormalBivError5 = (MCres[0][5] - truncNormalBiv5)/(MCres[1][5]+1e-12);
            Realv truncNormalBivError6 = (MCres[0][6] - expectedMax)/(MCres[1][6]+1e-12);
       
            if (fabs(bivAnalyticError) < tolerance)
                --result;
            else
                std::cout << " BivariateCumulativeNormal test failed " << rho << "\n";

            if (fabs(truncNormalBivError) < tolerance)
                --result;
            else
                std::cout << " truncNormalBivError test failed. " << rho << " " << MCres[0][1] << " " << truncNormalBiv << " " << MCres[1][1]+1e-12<<"\n";

            if (fabs(truncNormalBivError2) < tolerance)
                --result;
            else
                std::cout << " truncNormalBivError2 test failed." << rho << "\n";

            if (fabs(truncNormalBivError3) < tolerance)
                --result;
            else
                std::cout << " truncNormalBivError3 test failed." << rho << "\n";

            if (fabs(truncNormalBivError4) < tolerance)
                --result;
            else
                std::cout << " truncNormalBivError4 test failed." << rho << "\n";

            if (fabs(truncNormalBivError5) < tolerance)
                --result;
            else
                std::cout << " truncNormalBivError5 test failed." << rho << "\n";


            if (fabs(truncNormalBivError6) < tolerance)
                --result;
            else
                std::cout << " max test failed." << rho << "\n";

          

        }
    }
    if (verbose && result ==0)
        std::cout << "test passed\n";



    return result;

}
