// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3

//
//
//                                                                                                          Bridge_gold.h
//
//

#ifndef BRIDGE_GOLD_H
#define BRIDGE_GOLD_H
#include <vector>
#include <valarray>

template<class D>
class BrownianBridge
{

public:
    BrownianBridge(int PowerOfTwo_);

    void GenerateBridge(const std::vector<D>& variates,  std::vector<D>& bridgedVariates) const;

     void GenerateBridge(D* variates,  std::vector<D>& bridgedVariates) const;

     void GenerateBridge(D* variates, D* bridgedVariates) const;

// normally i hate get methods but we need them here for the GPU version to use

    int dimensions() const
    {
        return Points;
    }

     D getFirstMidScalar() const
     {
         return FirstMidScalar;
     }

    const std::vector<D>& getrightNotLeftScalars() const
        {
            return rightNotLeftScalars;
    }

    const std::vector<D>& getmidNotLeftScalars() const
        {
            return midNotLeftScalars;
    }

   const  std::vector<int>& getindexRightOneSided() const
        {
            return indexRightOneSided;
    }

    const std::vector<int>& getindexToDoOneSided() const
        {
            return indexToDoOneSided;
    }
 
   const  std::vector<D>& getrightScalars() const
        {
            return rightScalars;
    }
    
    const std::vector<D>& getmidScalars() const
        {
            return midScalars;
    }
    
    const std::vector<D>& getleftScalars() const
        {
            return leftScalars;
    }

    const std::vector<int>& getindexTodo() const
        {
            return indexTodo;
    }
    
    
    const std::vector<int>& getindexLeft() const
    {
        return indexLeft;
    }

    const std::vector<int>& getindexRight() const
     {
         return indexRight;
    }

    const std::vector<int>& getvariateToUseForThisIndex()
    {
        return variateToUseForThisIndex;
    }

private:
    int PowerOfTwo;
    int Points;

    D FirstMidScalar;

    std::vector<D> rightNotLeftScalars;
    std::vector<D> midNotLeftScalars;
    std::vector<int> indexRightOneSided;
    std::vector<int> indexToDoOneSided;
 
    std::vector<D> rightScalars;
    std::vector<D> midScalars;
    std::vector<D> leftScalars;

    std::vector<int> indexTodo;
    std::vector<int> indexLeft;
    std::vector<int> indexRight;

    std::vector<int> variateToUseForThisIndex;

};

int intPower(int x, int n);


template<class D>
class BrownianBridgeMultiDim
{
public:

    enum ordering {row, column, triangular};

    BrownianBridgeMultiDim(int powerOfTwo_, int factors_, ordering allocator_);

    void GenerateBridge(D* variates, D* output) const;

    void reorder(D*variates, D* output) const;

    const std::vector<int>& reordering() const;
    
    const std::vector<int>& reorderingDimension() const;
    const std::vector<int>& reorderingFactor() const;

    int consistencyCheck() const;

    const BrownianBridge<D>& innerBridge() const
    {
        return oneDBridge;
    }
    
private:

    int powerOfTwo;
    int factors;
    int points;
    int N; // factors*points


    std::vector<int> indexOrdering;
    std::vector<int> dimensionOrdering;
    std::vector<int> factorOrdering;
    mutable std::vector<D> tmp;
    BrownianBridge<D> oneDBridge;

    ordering allocator;

};



template<class D>
BrownianBridge<D>::BrownianBridge(int PowerOfTwo_) 
: 
PowerOfTwo(PowerOfTwo_)
{
    if (PowerOfTwo < 0)
        throw("We must have a positive power of two.");

    Points = static_cast<int>(pow(2.0,PowerOfTwo));

    FirstMidScalar =  static_cast<D>(pow(2.0, 0.5*PowerOfTwo));

    rightNotLeftScalars.resize(PowerOfTwo);
    midNotLeftScalars.resize(PowerOfTwo);
    indexRightOneSided.resize(PowerOfTwo);
    indexToDoOneSided.resize(PowerOfTwo);

    for (int i=0; i < PowerOfTwo; ++i)
    {
        D t = static_cast<D>(pow(2.0, PowerOfTwo-i-1.0));
        rightNotLeftScalars[i] =0.5;
        midNotLeftScalars[i] = static_cast<D>(sqrt(0.5*t));
        indexToDoOneSided[i] =static_cast<int>( pow(2.0,PowerOfTwo-i-1)) -1;       
        indexRightOneSided[i] = static_cast<int>( pow(2.0,PowerOfTwo-i)) -1;       
    }

    int pointsLeft = Points - PowerOfTwo-1;

    rightScalars.resize(pointsLeft);
    leftScalars.resize(pointsLeft);
    midScalars.resize(pointsLeft);
    indexRight.resize(pointsLeft);
    indexLeft.resize(pointsLeft);
    indexTodo.resize(pointsLeft);

    D leftScalar = 0.5;
    D rightScalar = 0.5;

 //   int thisIndex=0;
    for (int i=2, thisIndex=0; i <= PowerOfTwo; ++i)
    {
        int thisPower = PowerOfTwo - i;
        int thisMultiple = intPower(2,thisPower);

        int maxMultiple = intPower(2,i);

        D midScalar = static_cast<D>(sqrt(thisMultiple*0.5));

        for (int j=3; j < maxMultiple; j=j+2)
        {
            int index = j*thisMultiple-1;
            int indexBelow = (j-1)*thisMultiple-1;
            int IndexAbove = (j+1)*thisMultiple-1;

            indexLeft[thisIndex] = indexBelow;
            indexRight[thisIndex] = IndexAbove;
            indexTodo[thisIndex] = index;

            rightScalars[thisIndex] = rightScalar;
            leftScalars[thisIndex] = leftScalar;
            midScalars[thisIndex] = midScalar;

            ++thisIndex;
        }

    }

    variateToUseForThisIndex.resize(Points);


    for (int i=0, thisIndex=0; i <= PowerOfTwo; ++i)
    {
        int thisPower = PowerOfTwo - i;
        int thisMultiple = intPower(2,thisPower);
        int maxMultiple = intPower(2,i);

        for (int j=1; j <= maxMultiple; j=j+2)
        {
            int index = j*thisMultiple-1;
            variateToUseForThisIndex[index] = thisIndex;
            ++thisIndex;

        }
    }

}
template<class D>
void  BrownianBridge<D>::GenerateBridge(const std::vector<D>& variates,  std::vector<D>& bridgedVariates) const
{
    if (bridgedVariates.size() != Points || variates.size() != Points)
        throw("size mismatch");

    *(bridgedVariates.rbegin()) = variates[0]*FirstMidScalar;

    for (int i=0; i < PowerOfTwo; ++i)
    {
        int index = indexToDoOneSided[i];
        D rscalar= rightNotLeftScalars[i];
        D bv = bridgedVariates[indexRightOneSided[i]];
        D variate = variates[variateToUseForThisIndex[index]];
        D volscalar = midNotLeftScalars[i];
        bridgedVariates[index] = rscalar* bv+ volscalar*variate;                  
    }

    for (int i=0; i < Points- PowerOfTwo-1; ++i)
    {
        int index = indexTodo[i];

        bridgedVariates[index] = rightScalars[i]*bridgedVariates[indexRight[i]] 
        + leftScalars[i]*bridgedVariates[indexLeft[i]] 
        + midScalars[i]*variates[variateToUseForThisIndex[index]];                  
    }

}

template<class D>
void  BrownianBridge<D>::GenerateBridge(D* variates,  std::vector<D>& bridgedVariates) const
{
    if (bridgedVariates.size() !=  Points)
        throw("size mismatch");

    GenerateBridge(variates, &bridgedVariates[0]);
}

template<class D>
void  BrownianBridge<D>::GenerateBridge(D* variates,  D* bridgedVariates) const
{

    //    *(bridgedVariates.rbegin()) = variates[0]*FirstMidScalar;

    bridgedVariates[ Points-1] = variates[0]*FirstMidScalar;


    for (int i=0; i < PowerOfTwo; ++i)
    {
        int index = indexToDoOneSided[i];
        D rscalar= rightNotLeftScalars[i];
        D bv = bridgedVariates[indexRightOneSided[i]];
        D variate = variates[variateToUseForThisIndex[index]];
        D volscalar = midNotLeftScalars[i];
        bridgedVariates[index] = rscalar* bv+ volscalar*variate;                  
    }

    for (int i=0; i < Points- PowerOfTwo-1; ++i)
    {
        int index = indexTodo[i];

        bridgedVariates[index] = rightScalars[i]*bridgedVariates[indexRight[i]] 
        + leftScalars[i]*bridgedVariates[indexLeft[i]] 
        + midScalars[i]*variates[variateToUseForThisIndex[index]];                  
    }

    for (int i=1; i < Points; ++i)
    {

        int index = Points-i;

        bridgedVariates[index]  -=     bridgedVariates[index-1];
    }

}





template<class D>
BrownianBridgeMultiDim<D>::BrownianBridgeMultiDim(int powerOfTwo_, int factors_, ordering allocator) : powerOfTwo(powerOfTwo_), factors(factors_), points(intPower(2,powerOfTwo)), 
N(factors_*points), indexOrdering(N),  dimensionOrdering(N), factorOrdering(N),tmp(N), oneDBridge(powerOfTwo_)
{
    if (allocator == column)
    {
        for (int i =0; i < N; ++i)
        {
            indexOrdering[i] = i;
            int column = i / points; 
            int row = i % points; 
            dimensionOrdering[i] = row;    
            factorOrdering[i] =column;
        }
    }
    else
        if (allocator ==row)
        {
            for (int i =0; i < N; ++i)
            {
                int column = i % factors; 
                int row = i / factors; 
                indexOrdering[i] = column*points + row;
                dimensionOrdering[i] = row;
                factorOrdering[i] =column;
            }
        }
        else 
            if  (allocator == triangular)
            {
                int ccol =0;
                int crow =0;


                for (int i =0; i < N; ++i)
                {
                    int index = ccol*points + crow;
                    indexOrdering[i] = index;
                    dimensionOrdering[i] = crow;
                    factorOrdering[i] =ccol;

                    if (ccol > 0 && crow < points-1)
                    {
                        --ccol;
                        ++crow;
                    }
                    else 
                    {
                        int rowPlusCol = crow+ccol+1;

                        ccol =std::min(factors-1,rowPlusCol);

                        crow = rowPlusCol - ccol;
                    }
                }
            }

            else 
                throw("unknown allocator type");

}
template<class D>
void BrownianBridgeMultiDim<D>::GenerateBridge(D* variates, D* output) const
{
  //  for (int i =0; i < N; ++i)
    //    tmp[indexOrdering[i]] = variates[i];

    reorder(variates, &tmp[0]);

    for (int j=0; j < factors; ++j)
        oneDBridge.GenerateBridge(&tmp[j*points],&output[j*points]);
}
template<class D>
 void BrownianBridgeMultiDim<D>::reorder(D*variates, D* output) const
 {
         for (int i =0; i < N; ++i)
            output[indexOrdering[i]] = variates[i];
  }
 template<class D>
const std::vector<int>& BrownianBridgeMultiDim<D>::reordering() const
{
    return indexOrdering;
}
template<class D>
const std::vector<int>& BrownianBridgeMultiDim<D>::reorderingDimension() const
{
    return dimensionOrdering;
}
template<class D>
const std::vector<int>& BrownianBridgeMultiDim<D>::reorderingFactor() const
{
    return factorOrdering;
}


template<class D>
int BrownianBridgeMultiDim<D>::consistencyCheck() const
{
    std::vector<bool> tmp(N, false);

    for (int i=0; i < N;++i)
    {
        tmp[indexOrdering[i]]=true;

    }

    int fails = 0;

    for (int i=0; i < N;++i)
    {
        if (!tmp[i])
            ++fails;

    }

    return fails;

}




#endif

