// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#ifndef GEOMETRY_SWG
#define GEOMETRY_SWG

%module(package="ooflib.SWIG.common") geometry

%include "common/typemaps.swg"

%{
#include "common/geometry.h"
typedef class CPolygon<double, Coord> CPolygonCoord;
%}

%nodefaultctor CPolygonCoord;
class CPolygonCoord {};

ADD_REPR_GENERIC(CRectangle);

class CRectangle : public CPolygonCoord {
public:
  CRectangle(const Coord&, const Coord&);
  double xmin();
  double xmax();
  double ymin();
  double ymax();
  double height();
  double width();
  void expand(double factor);
};

#endif // GEOMETRY_SWG
