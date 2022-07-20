#pragma once
#ifndef _AUTOTIMER_H_
#define _AUTOTIMER_H_
#include "Types.h"

class AutoTimer : public NonCopyable
{
public:
    AutoTimer();
    void RecordTime();
    double CurrentTime();
    double ShowCurrentTime();
    double TimeConsumption();
    int NumberOfTimeRecorded() const;

private:
    long _iniTime;
    long _lastTime;
    vector<long> _recordTimes;
};

#endif //_AUTOTIMER_H_