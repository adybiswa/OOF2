// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov.
 */

#ifndef MASSDENSITY_SWG
#define MASSDENSITY_SWG

%module(package="ooflib.SWIG.engine.properties.massdensity") massdensity

%include "common/typemaps.swg"
%import "engine/property.swg"

%{
#include "engine/properties/massdensity/massdensity.h"
%}

class MassDensityProp : public Property {
public:
  MassDensityProp(PyObject *self, char *, double);
};

%pythoncode "engine/properties/massdensity/massdensity.spy"

#endif // MASSDENSITY_SWG
