// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#include <oofconfig.h>

#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include "mask.h"
#include "common/doublearray.h"

class Laplacian : public MASK
{
public:
  Laplacian();
  virtual ~Laplacian() {}
};

#endif //LAPLACIAN_H
