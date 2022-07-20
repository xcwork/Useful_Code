#ifndef _MULTIDIMCOUNTER_H_
#define _MULTIDIMCOUNTER_H_
#include "Types.h"

class MultiDimCounter :public NonCopyable
{
private:
    const vector<int> positionCounts;
    const int dim;
    const int count;
    mutable vector<int> coordinate;

public:
    MultiDimCounter(const vector<int>& posCounts);

    int size() const;
    void reset() const;
    bool moveNext() const;
    const vector<int>& currentCount() const;



};

#endif //_MULTIDIMCOUNTER_H_
