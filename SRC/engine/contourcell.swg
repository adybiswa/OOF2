// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#ifndef CONTOURCELL_SWG
#define CONTOURCELL_SWG

%module(package="ooflib.SWIG.engine") contourcell

%include "common/typemaps.swg"

%{
#include "engine/contourcell.h"
%}

class ContourCellCoord {
public:
  double x, y;
  int ix, iy;
};

%nodefaultctor ContourCellSkeleton;

class ContourCellSkeleton {
};

%extend ContourCellSkeleton {
  ContourCellCoord getitem(int i) {
    return self->corner[i];
  }
  int __len__() { return 3; }
};

%pythoncode "engine/contourcell.spy"

#endif // CONTOURCELL_SWG
