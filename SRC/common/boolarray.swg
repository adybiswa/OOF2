// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#ifndef BOOLARRAY_SWG
#define BOOLARRAY_SWG

%module(package="ooflib.SWIG.common") boolarray

%include "common/typemaps.swg"
%import "common/coord.swg"

%{
#include "common/boolarray.h"
%}

%nodefaultctor BoolArray;

class BoolArray {
public:
  ~BoolArray();
  BoolArray clone();
  void set(ICoord *iPoint);
  void reset(ICoord *iPoint);
  bool get(ICoord *iPoint);
  void toggle(ICoord *iPoint);
  void invert();
  %newobject pixels;
  std::vector<ICoord> *pixels(bool);
};

#endif // BOOLARRAY_SWG
