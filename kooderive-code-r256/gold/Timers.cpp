#include <gold/Timers.h>
Timer::Timer()
	{
	    LARGE_INTEGER res;
   
		if(!QueryPerformanceFrequency(&res))
			std::cout << "QueryPerformanceFrequency failure!\n";

		tickSize = 1.0/static_cast<double>(res.QuadPart);

       QueryPerformanceCounter(&res);
       startTick = res.QuadPart;
	}
