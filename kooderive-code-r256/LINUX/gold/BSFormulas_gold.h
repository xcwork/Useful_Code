//
//
//                                              BSFormulas_gold.h
//
//
// (c) Mark Joshi 2012,2014
// This code is released under the GNU public licence version 3

#ifndef BSFORMULAS_GOLD_H
#define BSFORMULAS_GOLD_H

#include <gold/math/typedefs_math_gold.h>

Realv Compute_d2(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv Compute_d1(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

void Compute_d1d2(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K,Realv& d1, Realv& d2);

Realv BlackScholesCall(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesPut(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesDigitalCall(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesDigitalPut(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesCallDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesPutDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesDigitalCallDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesDigitalPutDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesCallVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesPutVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesDigitalCallVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesDigitalPutVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K);

Realv BlackScholesPowerCall(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K, int Power);
Realv BlackScholesPowerCallDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K, int Power);
Realv BlackScholesPowerCallVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K, int Power);

Realv BlackScholesDoubleDigital(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K1, Realv K2);
Realv BlackScholesDoubleDigitalDelta(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K1, Realv K2);
Realv BlackScholesDoubleDigitalVega(Realv S0, Realv r, Realv d, Realv T, Realv Sigma, Realv K1, Realv K2);

Realv NormalBlackFromSd(Realv f0, Realv Strike,Realv Sd, Realv Annuity);

#endif
