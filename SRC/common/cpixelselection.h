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

#ifndef CPIXELSELECTION_H
#define CPIXELSELECTION_H

// Base class for python PixelSelection and related objects.  

#include "common/pixelselectioncourier.h"
#include "common/timestamp.h"
#include "common/pixelgroup.h"
#include "common/IO/bitoverlay.h"

class ActiveArea;

class CPixelSelection {
protected:
  // These can be set by the ActiveArea subclass.
  PixelSet pixset;
  BitmapOverlay bitmap;
private:
  TimeStamp timestamp;
  const ICoord isize_;
  const Coord size_;
  const ActiveArea *getActiveArea() const;
  const std::vector<ICoord> *getActivePixels() const;
public:
  CPixelSelection(const ICoord *pxlsize, const Coord *size, CMicrostructure*);
  CPixelSelection(const CPixelSelection&);
  CPixelSelection *clone() const { return new CPixelSelection(*this); }
  virtual ~CPixelSelection() {}

  CMicrostructure *getCMicrostructure() const;

  const Coord &size() const { return size_; }
  const ICoord &sizeInPixels() const { return isize_; }
  bool checkpixel(const ICoord *pixel) const;

  void clear();
  void clearWithoutCheck();
  void invert();
  void invertWithoutCheck();
  
  void select(PixelSelectionCourier*);
  void unselect(PixelSelectionCourier*);
  void toggle(PixelSelectionCourier*);
  void selectSelected(PixelSelectionCourier*);

  void selectWithoutCheck(PixelSelectionCourier*);
  void unselectWithoutCheck(PixelSelectionCourier*);
  void toggleWithoutCheck(PixelSelectionCourier*);

  bool isSelected(const ICoord*) const;
  TimeStamp *getTimeStamp() { return &timestamp; }
  const std::vector<ICoord> *members() const;
  const PixelSet *getPixelGroup() const { return &pixset; }
  const BitmapOverlay *getBitmap() const { return &bitmap; }
  void setFromGroup(const PixelSet*);
  std::size_t len() const;
};


#endif // CPIXELSELECTION_H
