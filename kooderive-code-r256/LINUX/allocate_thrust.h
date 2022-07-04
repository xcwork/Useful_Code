
//
//
//                                              Allocate_thrust.h
//
//


#ifndef ALLOCATE_THRUST_H
#define ALLOCATE_THRUST_H

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

// very dumb device array
// purpose is to allow allocation of global memory on GPU using thrust
// without initializing the memory

template<class T>
class allocate_device_thrust_pointer
{
public:

    allocate_device_thrust_pointer<T>(int N) : ptr( thrust::device_malloc<T>(N)), N_(N)
    {
      
    }

     ~allocate_device_thrust_pointer<T>()
    {
        thrust::device_free(ptr);
    }


    T* getGlobal() const
    {
        return thrust::raw_pointer_cast(&ptr[0]);  
    }
        
    thrust::device_ptr<T> getDevicePtr() const
    {
        return ptr;
    }

    thrust::device_ptr<T> getEndPtr() const
    {
        return ptr+N_;
    }

    int getN()
    {
           return N_;
    }

private:
    thrust::device_ptr<T> ptr;
    int N_;

    allocate_device_thrust_pointer(const allocate_device_thrust_pointer<T>&){}

};

#endif
