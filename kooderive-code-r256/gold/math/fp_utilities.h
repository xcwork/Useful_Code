//
//
//                                  fp_utilities.h
//
//
//
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3

#ifndef FP_UTILITIES_H
#define FP_UTILITIES_H
#include <iostream>

template<class T>
bool fp_isnan(T f)
{
    return f != f;
}

template<class it>
bool checkVectorForIsNan(it start, it end, bool output)
{
	bool found = false;
	it loc = start;
	while (loc != end && !found)
	{
		found = found || fp_isnan(*loc);
		++loc;

	}

	if (found && output)
		std::cout << " first isnan found at " << loc-start << "\n";

	return found;


}
#endif
