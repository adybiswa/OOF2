// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#ifndef OOFCANVASIMAGE_H
#define OOFCANVASIMAGE_H

#include "canvasitem.h"
#include "utility.h"
#include <string>

#ifdef OOFCANVAS_USE_IMAGEMAGICK
#include <Magick++.h>
#endif // OOFCANVAS_USE_IMAGEMAGICK

namespace OOFCanvas {

  // TODO? Arbitrary rotation.

  class CanvasImage : public CanvasItem {
  protected:
    Coord location;	      // lower-left corner in user coordinates
    Coord size;
    ICoord pixels;
    double opacity;
    bool pixelScaling;
    bool drawPixelByPixel;
    unsigned char *buffer;	// points to data owned by Cairo::ImageSurface
    int stride;
    Cairo::RefPtr<Cairo::ImageSurface> imageSurface;
    virtual void drawItem(Cairo::RefPtr<Cairo::Context>) const;
    virtual bool containsPoint(const OffScreenCanvas*, const Coord&) const;
    void setUp(Cairo::RefPtr<Cairo::ImageSurface>,
	       double, double);	// displayed size
    void setSurface(Cairo::RefPtr<Cairo::ImageSurface>, int);
  public:
    CanvasImage(const Coord &pos, const Coord &size, const ICoord &npixels);
    virtual const std::string &classname() const;
    friend std::ostream &operator<<(std::ostream&, const CanvasImage&);
    virtual std::string print() const;

    virtual void pixelExtents(double&, double&, double&, double&) const;

    static CanvasImage *newBlankImage(double, double, // position
				 int, int,	 // no. of pixels
				 double, double, // displayed size
				 double, double, double, double); //color,alpha

    static CanvasImage *newFromPNGFile(double, double,	   // position
				       const std::string&, // filename
				       double, double);	   // displayed size
#ifdef OOFCANVAS_USE_IMAGEMAGICK
    static CanvasImage *newFromImageMagickFile(double, double, // position
					       const std::string&, // filename
					       double, double);	// disp. size

    static CanvasImage *newFromImageMagick(double, double,	// position
					   Magick::Image,
					   double, double); // displayed size
#endif // OOFCANVAS_USE_IMAGEMAGICK

    void setPixelSize(); 
    void setDrawIndividualPixels() { drawPixelByPixel = true; }
    void set(int x, int y, double r, double g, double b);
    void set(int x, int y, double r, double g, double b, double a);
    void set(int x, int y, unsigned char r, unsigned char g, unsigned char b);
    void set(int x, int y, unsigned char r, unsigned char g, unsigned char b,
	     unsigned char a);
    void setOpacity(double alpha) { opacity = alpha; }

    Color get(int x, int y) const;
  };
  
  std::ostream &operator<<(std::ostream&, const CanvasImage&);

};				// namespace OOFCanvas


#endif // OOFCANVASIMAGE_H