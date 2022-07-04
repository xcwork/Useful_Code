//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

/*

The purpose of this file is to automate creation and deletion of pinned memory vectors.
All the work is in the header file. A source file allows us to make
sure it compiles. 
*/

#include <cudaWrappers/vector_pinned.h>

namespace
{

    int test()
    {
        vector_pinned<float> data(0);
        vector_pinned<float> data1(10000);
        vector_pinned<float> data2;
        data2.resize(100000);

        data1[0]=data2[0];
        int i = data.end() - data.begin();
        i-= data.size();

        return 0;
    }

}
