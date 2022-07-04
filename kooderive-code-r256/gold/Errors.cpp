// (c) Mark Joshi 2010,2011
// This code is released under the GNU public licence version 3

#include <gold/Errors.h>
#include <iostream>
void GenerateError(const char* errorMessage)
{
    std::cout << errorMessage << "\n";
    throw(errorMessage);
}
