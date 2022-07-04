//
//
//
//
//

#ifndef TIMERS_H
#define TIMERS_H

#include <time.h>
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define BILLION 1000000000L;

class Timer
{
public:
	Timer();

	void StartTimer()
	{
	    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
	      perror( "clock gettime" );
	      exit( EXIT_FAILURE );
	    }
	}

	double timePassed()
	{
	    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
	      perror( "clock gettime" );
	      exit( EXIT_FAILURE );
	    }
		return ( stop.tv_sec - start.tv_sec )
		          + ( stop.tv_nsec - start.tv_nsec )
		            / BILLION;
	}

private:
	struct timespec start, stop;

};

#endif

	

