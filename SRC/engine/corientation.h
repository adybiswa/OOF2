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

#ifndef CORIENTATION_H
#define CORIENTATION_H

class COrientation;
class COrientABG;
class COrientBunge;
class COrientQuaternion;
class COrientX;
class COrientXYZ;
class COrientAxis;
class COrientRodrigues;

#include "common/ooferror.h"
#include "engine/fieldindex.h"
// NO, I don't know why outputval.h is in engine but propertyoutput.h
// is in engine/IO.
#include "engine/IO/propertyoutput.h"
#include "engine/outputval.h"

#include <vector>
#include <string>
#include <iostream>

template <class ORIENT> class COrientationSubClass;
template <class ORIENT> class OIndex;
template <class ORIENT> class OComponents;
template <class ORIENT> class OIterator;

class SmallMatrix;
class LatticeSymmetry;

class COrientation : public NonArithmeticOutputVal {
protected:
  mutable SmallMatrix *cachedrot;
  void copyMatrix(const COrientation&);
  virtual const COrientation &copyFrom(const COrientation&) = 0;
public:
  COrientation();
  COrientation(const COrientation&);
  virtual ~COrientation();

  virtual const COrientation &operator=(const OutputVal &ov) {
    return operator=(dynamic_cast<const COrientation&>(ov));
  }
  virtual const COrientation &operator=(const COrientation &other) {
    copyFrom(other);
    return *this;
  }

  // As used by the Cijkl's "transform" method, these matrices
  // yield lab-frame vectors when right-multiplied by crystal-frame vectors.
  virtual SmallMatrix *rotation_() const = 0; // returns new'd pointer
  const SmallMatrix &rotation() const;

  // Any orientation can be transformed into another one via the
  // rotation matrix.  These are virtual functions so that subclass
  // objects can override the trivial conversion into their own type.
  virtual COrientABG abg() const;
  virtual COrientBunge bunge() const;
  virtual COrientQuaternion quaternion() const;
  virtual COrientX X() const;
  virtual COrientXYZ XYZ() const;
  virtual COrientAxis axis() const;
  virtual COrientRodrigues rodrigues() const;

  double misorientation(const COrientation&, const LatticeSymmetry&) const;
  double misorientation(const COrientation&, const std::string&) const;

  COrientAxis weightedAverage(double, double, const COrientation&) const;
  COrientAxis weightedAverage(double, double, const SmallMatrix&) const;
  COrientAxis weightedAverage(double, double, const COrientation&,
			      const LatticeSymmetry&) const;

  virtual void print(std::ostream&) const = 0;
  virtual FieldIndex *getIndex(const std::string&) const = 0;
};

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

// Because the COrientation classes are derived from OutputVal, they
// need a mechanism to fetch the names and values of their parameters
// using index and iterator classes derived from FieldIndex and
// FieldIterator.  This has to be done separately for each
// COrientation subclass, but fortunately can be done with templates.
// See the TODO in outputval.h about misusing FieldIndex.

// The indices are used in the __getitem__ methods defined in
// corientation.spy.  TODO: Add C++ operator[](OIndex&) ?

template <class ORIENT>
class COrientationSubClass : public COrientation {
protected:
  OComponents<ORIENT> comps;
public:
  COrientationSubClass()
    : comps(dynamic_cast<const ORIENT*>(this))
  {}
  virtual FieldIndex *getIndex(const std::string &s) const {
    return new OIndex<ORIENT>(dynamic_cast<const ORIENT*>(this), s);
  }
  virtual ComponentsP components() const {
    return ComponentsP(&comps);
  }
};

template <class ORIENT>
class OIndex : public FieldIndex {
protected:
  std::vector<std::string>::size_type which;
  const std::vector<std::string> &args;
public:
  OIndex(const ORIENT *o) : which(0), args(o->arguments) {}
  OIndex(const ORIENT *o, int i) : which(i), args(o->arguments) {}
  OIndex(const ORIENT *o, const std::string &s)
    : args(o->arguments)
  {
    for(std::vector<std::string>::size_type i=0; i<args.size(); i++) {
      if(args[i] == s) {
	which = i;
	return;
      }
    }
    throw ErrProgrammingError("Bad arg to OIndex: " + s, __FILE__, __LINE__);
  }
  OIndex(const OIndex &o) : which(o.which), args(o.args) {}
  const std::string &classname() const;
  virtual FieldIndex *clone() const { return new OIndex(*this); }
  virtual int integer() const { return which; }
  virtual void print(std::ostream &os) const {
    os << classname() << "('" << args[which] << "')";
  }
  virtual const std::string &shortrepr() const {
    return args[which];
  }
};

template <class ORIENT>
class OIterator : public ComponentIterator {
protected:
  int v;
  const ORIENT *orient;
public:
  OIterator(const ORIENT *o, int i) : v(i), orient(o) {}
  virtual bool operator!=(const ComponentIterator &othr) const {
    const OIterator<ORIENT> &other =
      dynamic_cast<const OIterator<ORIENT>&>(othr);
    return orient != other.orient || v != other.v;
  }
  virtual ComponentIterator &operator++() {
    v++;
    return *this;
  }
  virtual FieldIndex *fieldindex() const {
    return new OIndex<ORIENT>(orient, v);
  }
  virtual ComponentIterator *clone() const {
    return new OIterator<ORIENT>(orient, v);
  }
  virtual void print(std::ostream &os) const {
    os << "OIterator<" << orient->classname() << ">" << std::endl;
  }
};

template <class ORIENT>
class OComponents : public Components {
protected:
  const ORIENT *orient;
public:
  OComponents(const ORIENT *o) : orient(o) {}
  virtual ComponentIteratorP begin() const {
    return ComponentIteratorP(new OIterator<ORIENT>(orient, 0));
  }
  virtual ComponentIteratorP end() const {
    return ComponentIteratorP(
		      new OIterator<ORIENT>(orient, orient->arguments.size()));
  }
};

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

class COrientABG : public COrientationSubClass<COrientABG> {
private:
  double alpha_, beta_, gamma_;
protected:
  const COrientation &copyFrom(const COrientation&);
public:
  COrientABG(double alpha, double beta, double gamma)
    : alpha_(alpha), beta_(beta), gamma_(gamma)
  {}
  COrientABG(const SmallMatrix&);
  // Default constructor is needed so that an array of these can be created.
  COrientABG() : alpha_(0.0), beta_(0.0), gamma_(0.0) {}
  
  virtual SmallMatrix *rotation_() const;
  virtual COrientABG abg() const { return *this; }
  // COrientABG has special status among the COrientation subclasses,
  // because it's the default representation.  It needs some extra
  // methods because of that.
  COrientABG operator-() const {
    return COrientABG(-alpha_, -gamma_, -beta_);
  }
  double alpha() const { return alpha_; }
  double beta() const { return beta_; }
  double gamma() const { return gamma_; }
  bool operator==(const COrientABG&) const;
  bool operator!=(const COrientABG&) const;
  virtual void print(std::ostream&) const;

  // Methods required by OutputVal
  virtual const std::string &classname() const;
  virtual unsigned int dim() const { return 3; }
  virtual COrientABG *clone() const;
  virtual COrientABG *zero() const;
  virtual std::vector<double> *value_list() const;
  static const std::vector<std::string> arguments;
};

class COrientBunge : public COrientationSubClass<COrientBunge> {
private:
  double phi1_, theta_, phi2_;
protected:
  const COrientation &copyFrom(const COrientation&);
public:
  COrientBunge(double phi1, double theta, double phi2)
    : phi1_(phi1), theta_(theta), phi2_(phi2)
  {}
  COrientBunge(const SmallMatrix&);
  virtual SmallMatrix *rotation_() const;
  virtual COrientBunge bunge() const { return *this; }
  double phi1() const { return phi1_; }
  double theta() const { return theta_; }
  double phi2() const { return phi2_; }
  virtual void print(std::ostream&) const;

  // Methods required by OutputVal
  virtual const std::string &classname() const;
  virtual unsigned int dim() const { return 3; }
  virtual COrientBunge *clone() const;
  virtual COrientBunge *zero() const;
  virtual std::vector<double> *value_list() const;
  static const std::vector<std::string> arguments;
};

class COrientQuaternion : public COrientationSubClass<COrientQuaternion> {
private:
  double q[4];
protected:
  const COrientation &copyFrom(const COrientation&);
public:
  COrientQuaternion(double e0, double e1, double e2, double e3);
  COrientQuaternion(const SmallMatrix&);
  virtual SmallMatrix *rotation_() const;
  virtual COrientQuaternion quaternion() const { return *this; }
  virtual COrientAxis axis() const;
  double e0() const { return q[0]; }
  double e1() const { return q[1]; }
  double e2() const { return q[2]; }
  double e3() const { return q[3]; }
  double norm2() const;
  virtual void print(std::ostream&) const;

  // Methods required by OutputVal
  virtual const std::string &classname() const;
  virtual unsigned int dim() const { return 4; }
  virtual COrientQuaternion *clone() const;
  virtual COrientQuaternion *zero() const;
  virtual std::vector<double> *value_list() const;
  static const std::vector<std::string> arguments;
};

// Goldstein's "X" convention.  This may have some other more
// descriptive name, but I don't know what it is.  Rotations are z,x,z.

class COrientX : public COrientationSubClass<COrientX> {
private:
  double phi_, theta_, psi_;
protected:
  const COrientation &copyFrom(const COrientation&);
public:
  COrientX(double phi, double theta, double psi)
    : phi_(phi), theta_(theta), psi_(psi)
  {}
  COrientX(const SmallMatrix&);
  virtual SmallMatrix *rotation_() const;
  virtual COrientX X() const { return *this; }
  double phi() const { return phi_; }
  double theta() const { return theta_; }
  double psi() const { return psi_; }
  virtual void print(std::ostream&) const;

  // Methods required by OutputVal
  virtual const std::string &classname() const;
  virtual unsigned int dim() const { return 3; }
  virtual COrientX *clone() const;
  virtual COrientX *zero() const;
  virtual std::vector<double> *value_list() const;
  static const std::vector<std::string> arguments;
};

// The "aerodynamic" XYZ convention, with each rotation about a
// different principal axis.  Again the name is from Goldstein.

class COrientXYZ : public COrientationSubClass<COrientXYZ> {
private:
  double phi_, theta_, psi_;
protected:
  const COrientation &copyFrom(const COrientation&);
public:
  COrientXYZ(double phi, double theta, double psi)
    : phi_(phi), theta_(theta), psi_(psi)
  {}
  COrientXYZ(const SmallMatrix&);
  virtual SmallMatrix *rotation_() const;
  virtual COrientXYZ XYZ() const { return *this; }
  double phi() const { return phi_; }
  double theta() const { return theta_; }
  double psi() const { return psi_; }
  virtual void print(std::ostream&) const;

  // Methods required by OutputVal
  virtual const std::string &classname() const;
  virtual unsigned int dim() const { return 3; }
  virtual COrientXYZ *clone() const;
  virtual COrientXYZ *zero() const;
  virtual std::vector<double> *value_list() const;
  static const std::vector<std::string> arguments;
};

class COrientAxis : public COrientationSubClass<COrientAxis> {
private:
  double angle_, x_, y_, z_;
protected:
  const COrientation &copyFrom(const COrientation&);
public:
  COrientAxis(double angle, double x, double y, double z)
    : angle_(angle), x_(x), y_(y), z_(z)
  {}
  COrientAxis(const SmallMatrix&);
  virtual SmallMatrix *rotation_() const;
  virtual COrientAxis axis() const { return *this; }
  virtual COrientQuaternion quaternion() const;
  double angle() const { return angle_; }
  double x() const { return x_; }
  double y() const { return y_; }
  double z() const { return z_; }
  bool operator==(const COrientAxis&) const;
  bool operator!=(const COrientAxis&) const;
  virtual void print(std::ostream&) const;

  // Methods required by OutputVal
  virtual const std::string &classname() const;
  virtual unsigned int dim() const { return 4; }
  virtual COrientAxis *clone() const;
  virtual COrientAxis *zero() const;
  virtual std::vector<double> *value_list() const;
  static const std::vector<std::string> arguments;
};

// Rodrigues vector. Another way of describing crystal orientations.
// This form is quite popular in the texture community; it is
// particularly useful to describe fiber-texture and poling in
// ferroelectrics. --REG

class COrientRodrigues : public COrientationSubClass<COrientRodrigues> {
private:
  double r1_, r2_, r3_;
protected:
  const COrientation &copyFrom(const COrientation&);
public:
  COrientRodrigues(double r1, double r2, double r3)
    : r1_(r1), r2_(r2), r3_(r3)
  {}
  COrientRodrigues(const SmallMatrix&);
  virtual SmallMatrix *rotation_() const;
  virtual COrientRodrigues rodrigues() const { return *this; }
  double r1() const { return r1_; }
  double r2() const { return r2_; }
  double r3() const { return r3_; }
  virtual void print(std::ostream&) const;

  // Methods required by OutputVal
  virtual const std::string &classname() const;
  virtual unsigned int dim() const { return 3; }
  virtual COrientRodrigues *clone() const;
  virtual COrientRodrigues *zero() const;
  virtual std::vector<double> *value_list() const;
  static const std::vector<std::string> arguments;
};

std::ostream &operator<<(std::ostream&, const COrientation&);

COrientation *orientationFactory(const std::string*);

class OrientationPropertyOutputInit : public NonArithmeticPropertyOutputInit {
public:
  COrientation *operator()(const NonArithmeticPropertyOutput*,
			   const FEMesh*,
			   const Element*, const MasterCoord&) const;
};



#endif // CORIENTATION_H
