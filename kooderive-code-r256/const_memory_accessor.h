//
//
//
//                                                                                                                                 const memory accessor.h
//
//

// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3

#ifndef CONST_MEMORY_ACCESOR_H
#define CONST_MEMORY_ACCESOR_H
#include <vector>
#include "Const_memory_buffer.h"
#include "cudaMacros.h"
#include <cuda_runtime.h>
#include <cutil.h>
#include <cudaMacros.h>
#include <cutil_inline.h>


extern "C"
void SetConstMemory(const std::vector<float>& v, int offset);

extern "C"
void GetConstMemoryContents( std::vector<float>& v);


        class ConstMemoryMirror
        {
        
            public:
                
                static ConstMemoryMirror& instance()
                {
                    static ConstMemoryMirror unique;
                    return unique;
                 }
                 
                 void SetMemory(const std::vector<float>& v, int offset)
                 {
                        if (offset+v.size() > BUFFER_SIZE)
                            std::cout << "attempt to set buffer off end ignored ";
                        else
                        { 
                             std::copy(v.begin(),v.end(), memoryMirror.begin()+offset);
                 
                             COPYCONSTANTMEMORYFLOAT(dev_const_memory_buffer,memoryMirror);
                             
                            COPYFROMCONSTANTMEMORYTOFLOAT(dev_const_memory_buffer,memoryDoubleMirror);
                             
                             for (int i=0; i < BUFFER_SIZE; ++i)
                                if ( memoryMirror[i] != memoryDoubleMirror[i] )
                                    std::cout << "failure, " << i << "," << memoryMirror[i]  << "," << memoryDoubleMirror[i] << "\n";
                             
                        }
                 }
                 
                void GetMemory(std::vector<float>& v)
                 {
                      
                         v.resize(BUFFER_SIZE);
                 
                             
                         COPYFROMCONSTANTMEMORYTOFLOAT(dev_const_memory_buffer,v);
                                            
                        
                 }
                 
                 
                 private:
                     ConstMemoryMirror() : memoryMirror(BUFFER_SIZE), memoryDoubleMirror(BUFFER_SIZE)
                     {
                            for (int i=0; i < memoryMirror.size(); ++i)
                                memoryMirror[i] = i+1.5f;
                     
                     }
                     
                    std::vector<float> memoryMirror;
                    std::vector<float> memoryDoubleMirror; 
                
        
        };



extern "C"
void SetConstMemory(const std::vector<float>& v, int offset)
{
    ConstMemoryMirror::instance().SetMemory(v,offset);

}

extern "C"
void GetConstMemoryContents( std::vector<float>& v)
{
        ConstMemoryMirror::instance().GetMemory(v);
}


#endif
