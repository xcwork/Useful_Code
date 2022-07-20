#include "pch.h"
#include "MultiDimCounter.h"

int CheckAndCount(const vector<int>& posMaxValues)
{
    int total = 1;
    for (const auto& maxVal : posMaxValues)
    {
        if (maxVal < 1)
        {
            throw("Counter requires each pos max>0");
        }
        total *= maxVal;
    }
    return total;
}

MultiDimCounter::MultiDimCounter(const vector<int>& posCounts)
    :positionCounts(posCounts), dim(static_cast<int>(posCounts.size())), count(CheckAndCount(posCounts)), coordinate(posCounts.size(), 0ul)
{
    coordinate[0] = -1;
}

int MultiDimCounter::size() const
{
    return count;
}

void MultiDimCounter::reset() const
{
    vector<int>(dim, 0).swap(coordinate);
    coordinate[0] = -1;
}

bool MultiDimCounter::moveNext() const
{
    if (coordinate.back() == positionCounts.back())
    {
        return false;
    }
    int currentPos = 0;
    while (true)
    {
        coordinate[currentPos]++;
        if (coordinate[currentPos] == positionCounts[currentPos])
        {
            if (currentPos + 1 == dim)
            {
                return false;
            }
            coordinate[currentPos] = 0;
            currentPos++;
            continue;
        }
        else
        {
            return true;
        }
    }
}

const vector<int>& MultiDimCounter::currentCount() const
{
    if (coordinate[0] == -1)
    {
        throw runtime_error("Count does not begin.");
    }
    return coordinate;
}


