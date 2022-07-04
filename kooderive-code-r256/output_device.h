//
//
//                       output_device.h
//
//

#ifndef OUTPUT_DEVICE_H
#define OUTPUT_DEVICE_H
#include <iostream>

template<class T>
void DumpDeviceVector(T& out)
{
    for (size_t i=0;  i < out.size(); ++i)
        std::cout << i << "," << out[i] << "\n";

}

template<class T>
void DumpDeviceMatrix(T& out,int rows, int columns)
{
    for (int i=0;  i < rows; ++i)
    {
        for (int j=0; j < columns; ++j)
         std::cout  << out[i*columns+j] << ",";
        std::cout << "\n";
    }

}

#endif

