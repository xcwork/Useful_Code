//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

/*

The purpose of this file is to automate creation and deletion of pinned memory vectors
*/

#ifndef VECTOR_PINNED_H
#define VECTOR_PINNED_H
#include <cstddef>
#include <vector>
#include <cutil_inline.h>

template<class T>
class vector_pinned
{
public:
    vector_pinned();

    ~vector_pinned();
    
    explicit vector_pinned(size_t length);

    void resize(size_t newLength);

    T* begin()
    {
        return start;
    }

    T* end()
    {
        return endPtr;
    }

    T& operator[](size_t i)
    {   
        return start[i];
    }

    const  T& operator[](size_t i) const
    {   
        return start[i];
    }

    size_t size() const
    {
        return endPtr-start;
    }

private:
    bool allocated;
    bool align;



    T* start;
    T* endPtr;
    size_t length;
    size_t bufferSize;

    T& operator=(const  vector_pinned<T>& ){ return *this;}

    vector_pinned(const vector_pinned<T>& ){}

};

template<class T>
vector_pinned<T>::vector_pinned()
: allocated(false), start(0), endPtr(0), length(0), bufferSize(0)
{
}

template<class T>
 vector_pinned<T>::vector_pinned(size_t lengthVal) : length(lengthVal),
                                                             bufferSize(lengthVal)
{
    if (lengthVal ==0)
    {
        allocated = false; 
        start=0;
        endPtr=0;
        return;
    }

    cutilSafeCall(
        cudaHostAlloc( (void**)(&start),length*sizeof(T),
        cudaHostAllocDefault)
        );
  
    endPtr = start+length;

    allocated= true;
}

template<class T>
void vector_pinned<T>::resize(size_t newLength)
{
    if (newLength <= bufferSize)
    {
        length = newLength;
        endPtr = start+length;
        return;
    }
    
    if (allocated)
    {
        cutilSafeCall(cudaFreeHost( start));
    }

    length = newLength;
    bufferSize = newLength;
    
    cutilSafeCall(
        cudaHostAlloc( (void**)(&start),length*sizeof(T),
        cudaHostAllocDefault));
    
    endPtr = start+length;

}

template<class T>
vector_pinned<T>::~vector_pinned()
{
    if (allocated)
         cutilSafeCall(cudaFreeHost( start));
}


#endif
