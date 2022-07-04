/*
 * Copyright (c) 2010 Oh Kang Kwon. Released under GNU Public Licence Version 3.
 (c) Mark Joshi (2010)
 */


#include <gold/MatrixFacade.h> 
#include <vector>
void bridge_allocate_factors(std::vector<int>& indices,int factors, int steps)
{
    MatrixFacade<int> indexMatrix(&indices[0],factors,steps);
    int counter(0);
    for (int i = 0; i < factors; ++i)
    {
        for (int j = 0; j < steps; ++j)
        {
            indexMatrix(i,j) = counter;
            ++counter;
        }
    }
}

void bridge_allocate_step(std::vector<int>& indices,int factors, int steps)
{
    MatrixFacade<int> indexMatrix(&indices[0],factors,steps);
  
    int counter(0);
    for (int j = 0; j < steps; ++j)
    {
        for (int i = 0; i < factors; ++i)
        {
            indexMatrix(i,j) = counter;
            ++counter;
        }
    }
}

void bridge_allocate_diagonal(std::vector<int>& indices,int factors, int steps)
{
    MatrixFacade<int> indexMatrix(&indices[0],factors,steps);
  
    // starting position of the current diagonal
    int i0 = 0, j0 = 0;
    // current position
    int i = 0, j = 0;
    int counter = 0;
    while (counter < factors * steps) 
    {
        indexMatrix(i,j) = counter;
        ++counter;
        if (i == 0 || j == steps - 1) 
        {
            // we completed a diagonal and have to start a new one
            if (i0 < factors - 1) 
            {
                // we start the path of the next factor
                i0 = i0 + 1;
                j0 = 0;
            } 
            else 
            {
                // we move along the path of the last factor
                i0 = factors - 1;
                j0 = j0 + 1;
            }
            i = i0;
            j = j0;
        } 
        else 
        {
            // we move along the diagonal
            i = i - 1;
            j = j + 1;
        }
    }
}
