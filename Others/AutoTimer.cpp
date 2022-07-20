#include "pch.h"
#include <ctime>
#include "AutoTimer.h"


AutoTimer::AutoTimer()
{
    _iniTime = clock();
    _lastTime = _iniTime;
    _recordTimes.push_back(_iniTime);
}
void AutoTimer::RecordTime()
{
    _lastTime = clock();
    _recordTimes.push_back(_lastTime);
}
double AutoTimer::CurrentTime()
{
    long tmp = clock();
    double theRet = static_cast<double>((tmp - _iniTime + 0.0) / CLOCKS_PER_SEC);
    _lastTime = tmp;
    _recordTimes.push_back(_lastTime);
    return theRet;
}
double AutoTimer::ShowCurrentTime()
{
    double theRet = CurrentTime();
    return theRet;
}
double AutoTimer::TimeConsumption()
{
    long tmp = clock();
    double theRet = static_cast<double>((tmp - _lastTime + 0.0) / CLOCKS_PER_SEC);
    _lastTime = tmp;
    _recordTimes.push_back(_lastTime);
    return theRet;
}
int AutoTimer::NumberOfTimeRecorded() const
{
    return static_cast<int>(_recordTimes.size());
}