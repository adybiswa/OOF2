// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#ifndef PIXELATTRIBUTE_SWG
#define PIXELATTRIBUTE_SWG

%module(package="ooflib.SWIG.common") pixelattribute

%include "common/typemaps.swg"
%{
#include "common/pixelattribute.h"
%}

%nodefaultctor PxlAttributeRegistration;

class PxlAttributeRegistration {
public:
  const std::string &name();
};

PYTHONEXPORTABLE(PxlAttributeRegistration);

int nAttributes();
PxlAttributeRegistration *getRegistration(int);

%pythoncode %{

def _writeGlobalData(self, datafile, ms):
    pass

PxlAttributeRegistration.writeGlobalData = _writeGlobalData

%}

#endif // PIXELATTRIBUTE_SWG
