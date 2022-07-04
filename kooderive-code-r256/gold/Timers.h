//
//
//
//
//

#ifndef TIMERS_H
#define TIMERS_H
#include <windows.h>
#include <iostream>
class Timer
{
public:
	Timer();

	void StartTimer()
	{
       LARGE_INTEGER res;
       QueryPerformanceCounter(&res);
       startTick = res.QuadPart;
	}

	double timePassed()
	{
		   LARGE_INTEGER res;
		  QueryPerformanceCounter(&res);
		  return double(res.QuadPart-startTick)*tickSize;

	}

private:

double tickSize;
__int64 startTick;

};

#endif

	

