/*
 * Copyright (c) 2010 Oh Kang Kwon. Released under GNU Public Licence Version 3.
 (c) Mark Joshi (2010)
 */

#ifndef BRIDGE_ALLOCATOR_GOLD_H
#define BRIDGE_ALLOCATOR_GOLD_H


#include <vector>

void bridge_allocate_factors(std::vector<int>& indices,int factors, int steps);

void bridge_allocate_step(std::vector<int>& indices,int factors, int steps);

void bridge_allocate_diagonal(std::vector<int>& indices,int factors, int steps);
     

#endif 

