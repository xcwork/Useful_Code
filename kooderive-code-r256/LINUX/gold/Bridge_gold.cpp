// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3


#include <gold/Bridge_gold.h>

#include <cmath>

int intPower(int x, int n)
{
    if (n <0)
        throw("neg powers not supported");

    int r=1;

    for (int j=0; j < n; ++j)
        r*=x;

    return r;

}

