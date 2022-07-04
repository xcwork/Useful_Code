//
//
//                               smallFunctions.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3
// no cpp file at the moment

#ifndef SMALL_FUNCTIONS_H
#define SMALL_FUNCTIONS_H


class squareof
{
public:
	__host__ __device__ float operator()(float x)
	{
		return x*x;
	}
};

class inInterval
{
private:
	float lower_,upper_;
public:
	__host__ __device__ inInterval(float lower,float upper) : lower_(lower), upper_(upper)
	{
	}

	__host__ __device__ int operator()(float val)
	{
		return (val > lower_) && (val < upper_) ? 1 : 0;

	}

};

class shiftAndMult
{
private:
	float shift_;
	float mult_;
	
public:
	shiftAndMult(float shift,float mult):shift_(shift), mult_(mult) {}

	__host__ __device__ float operator()(float val)
	{
		return (val+shift_)*mult_;

	}

};

#endif
