
//
//                                  CosMethod_gold.h
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence 

#ifndef COSMETHOD_GOLD_H
#define COSMETHOD_GOLD_H
#include <gold/math/constants_gold.h>

/*
Formula for 
\int_{c}^{d} e^{\lambda y} \cos(\gamma y + \beta) dy
*/
template<class T> 
double CosDigitalPrice(double a , double b, int N, double c, double d, const T& charFunction);


/* what is 

$$
\int \limits_{c}^{d} e^{\lambda y} \cos (\gamma y + \beta) dy 
$$

?

*/


double GetCosCoefficient(double c, double d, double lambda, double gamma, double beta);


template<class T> 
class CosCharacteristicFunctionComputer
{
public:
    CosCharacteristicFunctionComputer(const T & charFunction, double a,
                                                                 double b,
                                                                 double scale,
                                                                 double imaginaryShift);

    double GetCharacteristicFunctionCosCoefficient(int j) const;

private:
    const T  charFunction_;
    double a_;
    double b_;
    double scale_;
    double imaginaryShift_;
    std::complex<double> pibybminusa_;
    double outsideCoeff_;

};

template<class T> 
CosCharacteristicFunctionComputer<T>::CosCharacteristicFunctionComputer(const T & charFunction, double a,
                                                                        double b,
                                                                        double scale,
                                                                        double imaginaryShift)
                                                                        :
charFunction_(charFunction),
    a_(a),
    b_(b), scale_(scale), imaginaryShift_(imaginaryShift)
{
    pibybminusa_= M_PI/(b_-a_);
    outsideCoeff_ = scale *2.0/(b_-a_);
}

template<class T> 
double
CosCharacteristicFunctionComputer<T>::GetCharacteristicFunctionCosCoefficient(int j) const
{
    std::complex<double> i(0.0,1.0);

    std::complex<double> j_complex(j,0.0);

    std::complex<double> cFunctionValue = charFunction_(j_complex*pibybminusa_ - i *imaginaryShift_);
    std::complex<double> multiplier = exp(-i*a_*j_complex*pibybminusa_);
    std::complex<double> complexCoeff = cFunctionValue*multiplier;
    double realPart = complexCoeff.real();

    double res = outsideCoeff_*realPart;

    return res;
}


template<class T> 
double CosDigitalPriceZeroShift(double a , double b, int N, double c, double d,double df, const T& charFunction)
{

    CosCharacteristicFunctionComputer<T> computer(charFunction, a,
                                                                         b,
                                                                        df,
                                                                        0.0);

    double total =0.0;

    double lengthMult =1.0 ;// 0.5*(b-a);

    double psi0 = d-c;

    double a0 = lengthMult*psi0;

    double d0 = computer.GetCharacteristicFunctionCosCoefficient(0);

    total += a0*d0*0.5; // first term is different

    double theta1 = M_PI*(d-a)/(b-a);
    double theta2 = M_PI*(c-a)/(b-a);


    for (int j=1; j <N; ++j)
    {
        double dj = computer.GetCharacteristicFunctionCosCoefficient(j);

        double sin1arg = j*theta1;
        double sin2arg = j*theta2;

        

        double psij = ((b-a)/(j*M_PI))*(sin(sin1arg) - sin(sin2arg));
        double psitildej = lengthMult*psij;

        double term = dj*psitildej;
        total+= term;

    }

    return total;
}


template<class T> 
double CosAssetDigitalPriceZeroShift(double a , // truncation lower limit for density
                                     double b, // truncation upper limit for density
                                     int N,  // no of terms in the series 
                                     double c,  // lower digital limit in log space
                                     double d,// upper digital limit in log space
                                     double df, // discount factor 
                                     const T& charFunction // functor for characteristic function
                                     )
{

    CosCharacteristicFunctionComputer<T> computer(charFunction, a,
                                                                         b,
                                                                        df,
                                                                        0.0);
    double lambda =1; // i.e. e^x

    double total =0.0;

    double lengthMult =1.0 ;// 0.5*(b-a);

    double psi0 = d-c;

    double a0 = GetCosCoefficient(c, d,  lambda, 0, 0);

    double d0 = computer.GetCharacteristicFunctionCosCoefficient(0);

    total += a0*d0*0.5; // first term is different

  

    for (int j=1; j <N; ++j)
    {
        double dj = computer.GetCharacteristicFunctionCosCoefficient(j);

     
        double beta = j*M_PI/(b-a);
        double gamma =-a*beta;
        double psij = GetCosCoefficient(c, d,  lambda, beta, gamma);

       
        double psitildej = lengthMult*psij;

    //    std::cout << ", "<< j << "," << dj << "," << psij ;

        double term = dj*psitildej;
        total+= term;

    }

    return total;
}


template<class T> 
double CosPricePowerDigitalShift(double a , // truncation lower limit for density
                                     double b, // truncation upper limit for density
                                     int N,  // no of terms in the series 
                                     double c,  // lower digital limit in log space
                                     double d,// upper digital limit in log space
                                     double lambda, // pay-off is e^{\lambda x} I_{c,d}(x) in log space
                                     double shift, // power of numeraire 
                                     double df, // discount factor 
                                     const T& charFunction // functor for characteristic function
                                     )
{

    CosCharacteristicFunctionComputer<T> computer(charFunction, a,
                                                                         b,
                                                                        df,
                                                                        shift);
    double lambdaEffective =lambda-shift; 

    double total =0.0;

    double lengthMult =1.0 ;// 0.5*(b-a);

    double psi0 = d-c;

    double a0 = GetCosCoefficient(c, d,  lambdaEffective, 0, 0);

    double d0 = computer.GetCharacteristicFunctionCosCoefficient(0);

    total += a0*d0*0.5; // first term is different

  

    for (int j=1; j <N; ++j)
    {
        double dj = computer.GetCharacteristicFunctionCosCoefficient(j);

     
        double beta = j*M_PI/(b-a);
        double gamma =-a*beta;
        double psij = GetCosCoefficient(c, d,  lambdaEffective, beta, gamma);

       
        double psitildej = lengthMult*psij;

    //    std::cout << ", "<< j << "," << dj << "," << psij ;

        double term = dj*psitildej;
        total+= term;

    }

    return total;
}



#endif
