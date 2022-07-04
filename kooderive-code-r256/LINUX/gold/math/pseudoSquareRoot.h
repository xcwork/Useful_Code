
////                     pseudoSquareRoot.h

//  Copyright (C) 2012, 2013 Mark Joshi

// Release under GNU public licence version 3
/*
 Copyright (C) 2003, 2004 Ferdinando Ametrano

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file pseudosqrt.hpp
    \brief pseudo square root of a real symmetric matrix
*/

#ifndef KOO_PSEUDO_SQRT_H
#define KOO_PSEUDO_SQRT_H

#include <gold/MatrixFacade.h>

#include <gold/math/typedefs_math_gold.h>

void pseudoSqrtSchur(const MatrixConstFacade<Real_d>& input, MatrixFacade<Real_d>& output, int factors, bool normalise );

    //! Returns the pseudo square root of a real symmetric matrix
   
    void pseudoSqrtSpectral(
                        const MatrixConstFacade<Real_d>& input,
                        MatrixFacade<Real_d>& pseudoRoot);

    




#endif
