// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

/*
Kevin Chang
August 26, 2002
Senior Research Project
Torrence

Center for Theoretical and Computational Materials Science
National Institute of Standards and Technology

gabor.h

This file contains a function used by SWIG to start the Gabor function.
*/

#include <oofconfig.h>

#ifndef GABOR_H
#define GABOR_H

class OOFImage;
// class grayImage;

void makeGabor(OOFImage&, double, double, int);

#endif // GABOR_H
