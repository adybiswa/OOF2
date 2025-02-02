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

// Symmetric matrix storage class
// may not be best for linear algebra!

class SymmMatrix;

#ifndef SYMMMATRIX_H
#define SYMMMATRIX_H

#include "engine/IO/propertyoutput.h"
#include "engine/eigenvalues.h"
#include "engine/outputval.h"

#include <iostream>
#include <math.h>
#include <string>
#include <vector>

class COrientation;
class FieldIndex;
class DoubleVec;
class IndexP;
class SmallMatrix;
class SymTensorIndex;

class SymmMatrix {
protected:
//   friend double SymmMatrix_get(SymmMatrix*, int, int);
  double **m;
  unsigned int nrows;
  void allocate();
  void free();
  virtual void copy(double**);
public:
  SymmMatrix() : m(0), nrows(0) {}
  SymmMatrix(int);		// specifies size
  SymmMatrix(const SymmMatrix&); // copy constructor
  virtual ~SymmMatrix();
  SymmMatrix &operator=(const SymmMatrix&);
  SymmMatrix &operator*=(double);
  SymmMatrix &operator/=(double);
  bool operator==(const SymmMatrix&);
  double &operator()(int i, int j);
  double operator()(int i, int j) const;
  SymmMatrix &operator+=(const SymmMatrix&);
  SymmMatrix &operator-=(const SymmMatrix&);
  void resize(unsigned int);
  unsigned int size() const { return nrows; }
  void clear(double x=0);
  int badindex(int i) const { return i < 0 || i >= int(nrows); }
	
  SymmMatrix transform(const COrientation*) const; // A^T B A
	
  friend class Debug;
  friend class SymmMatrix3;
  friend std::ostream& operator<<(std::ostream&, const SymmMatrix&);
  friend SmallMatrix operator*(const SymmMatrix&, const SymmMatrix&);
  friend SymmMatrix operator*(double, const SymmMatrix&);
  friend SymmMatrix operator*(const SymmMatrix&, double);
  friend DoubleVec operator*(const SymmMatrix&, const DoubleVec&);
};

class SymmMatrix3 : public ArithmeticOutputVal, public SymmMatrix {
// OutputVal is a PythonExportable class, and must be the first base
// class listed so that the PythonExportable dynamic classes work.
// This doesn't feel right...
private:
  mutable EigenValues eigenvalues; // cached
  mutable bool dirtyeigs_;	// are eigenvalues up-to-date?
  void findEigenvalues() const;
  static std::string classname_; // OutputVal is PythonExportable
public:
  SymmMatrix3() : SymmMatrix(3), dirtyeigs_(true) {}
//   virtual ~SymmMatrix3();
  SymmMatrix3(double, double, double, double, double, double); // voigt order
  SymmMatrix3(const SymmMatrix3&);
  SymmMatrix3(const SymmMatrix&);
  virtual const SymmMatrix3 &operator=(const OutputVal &other);
  virtual unsigned int dim() const { return 6; }
  virtual OutputVal *clone() const;
  virtual OutputVal *zero() const;
  virtual SymmMatrix3 *one() const;
  virtual const std::string &classname() const { return classname_; }
  SymmMatrix3 &operator=(const SymmMatrix3 &x) {
    dirtyeigs_ = x.dirtyeigs_;
    eigenvalues = x.eigenvalues;
    return dynamic_cast<SymmMatrix3&>(SymmMatrix::operator=(x));
  }

  virtual void component_pow(int p) {
    dirtyeigs_ = true;
    double *data = m[0];
    for(int i=0;i<6;i++)  // SymmMatrix3 guaranteed to have 6 entries.
      data[i] = pow(data[i], p);
  }
  virtual void component_square() {
    dirtyeigs_ = true;
    double *data = m[0];
    for(int i=0;i<6;i++) 
      data[i] *= data[i];
  }
  virtual void component_sqrt() {
    dirtyeigs_ = true;
    double *data = m[0];
    for(int i=0;i<6;i++)
      data[i] = sqrt(data[i]);
  }
  virtual std::vector<double>* value_list() const;

  ArithmeticOutputVal &operator*=(double x) {
    dirtyeigs_ = true;
    SymmMatrix::operator*=(x);
    return *this;
  }
  ArithmeticOutputVal &operator+=(const ArithmeticOutputVal &x) {
    dirtyeigs_ = true;
    const SymmMatrix3 &sm = dynamic_cast<const SymmMatrix3&>(x);
    SymmMatrix::operator+=(sm);
    return *this;
  }
  ArithmeticOutputVal &operator-=(const ArithmeticOutputVal &x) {
    dirtyeigs_ = true;
    const SymmMatrix3 &sm = dynamic_cast<const SymmMatrix3&>(x);
    SymmMatrix::operator-=(sm);
    return *this;
  }
  SymmMatrix3 &operator/=(double x) {
    dirtyeigs_ = true;
    SymmMatrix::operator/=(x);
    return *this;
  }
  SymmMatrix3 &operator+=(const SymmMatrix3 &x) {
    dirtyeigs_ = true;
    SymmMatrix::operator+=(x);
    return *this;
  }
  SymmMatrix3 &operator-=(const SymmMatrix3 &x) {
    dirtyeigs_ = true;
    SymmMatrix::operator-=(x);
    return *this;
  }
  virtual double operator[](const FieldIndex&) const;
  virtual double &operator[](const FieldIndex&);
  double operator[](const SymTensorIndex&) const;
  double &operator[](const SymTensorIndex&);
  double trace() const;
  double determinant() const;
  double secondInvariant() const;
  double deviator() const;
  double vonMises() const;
  virtual double magnitude() const;
  double maxEigenvalue() const;
  double midEigenvalue() const;
  double minEigenvalue() const;
  double contract(const SymmMatrix3&) const;

  virtual FieldIndex *getIndex(const std::string&) const;
  virtual ComponentsP components() const;
  virtual void print(std::ostream&) const;
};

SymmMatrix3 operator+(const SymmMatrix3&, const SymmMatrix3&);
SymmMatrix3 operator-(const SymmMatrix3&, const SymmMatrix3&);
SymmMatrix3 operator*(const SymmMatrix3&, double);
SymmMatrix3 operator*(double, const SymmMatrix3&);
SymmMatrix3 operator/(SymmMatrix3&, double);


ArithmeticOutputValue *newSymTensorOutputValue();

class SymmMatrix3PropertyOutputInit : public ArithmeticPropertyOutputInit {
public:
  SymmMatrix3 *operator()(const ArithmeticPropertyOutput*,
				  const FEMesh*,
				  const Element*, const MasterCoord&) const;
};

void copyOutputVals(const SymmMatrix3&, ListOutputVal*,
		    const std::vector<std::string>&);

std::ostream& operator<<(std::ostream&, const SymmMatrix&);
// SymmMatrix3 needs operator<< to disambiguate the base class
// operators.
std::ostream& operator<<(std::ostream&, const SymmMatrix3&);



#endif
