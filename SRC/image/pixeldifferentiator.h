// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#ifndef PIXELDIFFERENTIATORI_H
#define PIXELDIFFERENTIATORI_H

#include <oofconfig.h>
#include <vector>

#include "common/array.h"
#include "common/burn.h"

class OOFImage;

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

class CColorDifferentiator3 : public CPixelDifferentiator3 {
private:
  const OOFImage *image;
  double local_flammability;
  double global_flammability;
  bool useL2norm;
public:
  CColorDifferentiator3(const OOFImage *image, double lf, double gf, bool l2);
  virtual bool operator()(const ICoord&, const ICoord&, const ICoord&) const;
};

class CColorDifferentiator2 : public CPixelDifferentiator2 {
private:
  const OOFImage *image;
  double color_delta;
  bool useL2norm;
public:
  CColorDifferentiator2(const OOFImage *image, double cd, bool l2);
  virtual bool operator()(const ICoord&, const ICoord&) const;
};


#endif // PIXELDIFFERENTIATORI_H
