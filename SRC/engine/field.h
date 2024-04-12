// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#ifndef FIELD_H
#define FIELD_H

class CompoundField;
class Field;
class FieldIndex;
class ScalarField;
class ScalarFieldBase;
class TwoVectorField;
class TwoVectorFieldBase;
class ThreeVectorField;
class VectorFieldBase;

#include "common/coord.h"
#include "freedom.h"
#include "common/identification.h"
#include "common/pythonexportable.h"
#include "engine/fieldeqnlist.h"
#include "engine/fieldindex.h"
#include "engine/indextypes.h"
#include <oofconfig.h>
#include <iostream>
#include <string>
#include <vector>

class ArithmeticOutputValue;
class CSubProblem;
class ElementFuncNodeIterator;
class FEMesh;
class FuncNode;
class OutputValue;
class Property;

// There is one Field for each physical field.  There should be only
// one instance of each derived Field class.  These are just
// bookkeeping classes, and don't actually store the field values at
// each node. That's done by DegreeOfFreedom. So there's no problem
// with creating fields that aren't used. Thus, every possible field
// is constructed (in problem.py), and they're never deleted.

// Fields are *active* if they're being solved for.
// Fields are *defined* if the nodes contain values for them.

// Because a Field may be defined, or active, or in-plane, on one
// CSubProblem, but not on another CSubProblem, much of the data for a Field is
// stored in a CSubProblem::FieldData class inside the CSubProblem class.  This
// is meant to be completely transparent, but it does mean that a
// CSubProblem pointer has to be passed to some Field functions.

// Access to degrees of freedom at Nodes (specifically, FuncNodes) is
// done via routines like DegreeOfFreedom *Field::operator()(const
// FuncNode&), but these aren't defined in the base class, because
// they can act very differently depending on the type of
// field. (Since the Fields are global variables, there's no need to
// use virtual functions -- any Property needing to get the value of a
// Field has access to the derived class functions.)

// ARRGH! That's not quite right! CompoundField::out_of_plane()
// returns a pointer to a Field (base class). So we need at least one
// way of accessing DegreeOfFreedom objects through the base
// class. Therefore the function
//    DegreeOfFreedom *Field::operator()(const FuncNode&, int component)
// is virtual, even though the component argument doesn't make sense
// for some fields.

class Field : public IdentifiedObject
{
private:
  const std::string name_;
  const unsigned int index_;
protected:
  const int dim;
  Field *time_derivative_;
public:
  Field(const std::string &, int);

  const std::string &name() const;
  unsigned int index() const { return index_; }

  static Field *getField(const std::string &name);

  // where a given component lives in the dof lists in a Node
  int localindex(const FuncNode*, const FieldIndex &component) const;

  int ndof() const { return dim; } // number of degrees of freedom

  Field *time_derivative() const { return time_derivative_; }

  void set_time_derivative(Field *f) { time_derivative_ = f; }

  virtual void activate(CSubProblem*) const;
  virtual void deactivate(CSubProblem*) const;
  bool is_active(const CSubProblem*) const;
  
  virtual void define(CSubProblem*) const;
  virtual void undefine(CSubProblem*) const;
  bool is_defined(const CSubProblem*) const;

  virtual void registerProperty(Property*) const;

  // all() needs to be a function, because it is used during the
  // construction of global Field objects.  Making it a function
  // guarantees that the vector that it returns is constructed before
  // it's needed.
  static std::vector<Field*> &all();

  double value(const FEMesh*, const FuncNode*, int component) const;
  double value(const FEMesh*, const ElementFuncNodeIterator&, int component)
    const;

  // TODO: Add
  // double value(const FEMesh*, const Element*, const MasterPosition&);
  // double value(const FEMesh*, const Coord&);
  // which will interpolate to the given position.  The second form
  // will find the Element and MasterCoord and call the first
  // form. This will be useful in Outputs, and maybe elsewhere.  See
  // ThermalExpansion::output.
  
  virtual DegreeOfFreedom *operator()(const FuncNode*, int component) const=0;
  DegreeOfFreedom *operator()(const FuncNode &n, int component) const
  {
    return operator()(&n, component);
  }
  virtual DegreeOfFreedom *operator()(const ElementFuncNodeIterator&,
				      int component)
    const = 0;
  DegreeOfFreedom *operator()(const ElementFuncNodeIterator &n,
			      const IndexP &i) const
  {
    return operator()(n, i.integer());
  }

  virtual ArithmeticOutputValue newOutputValue() const = 0;

  virtual ArithmeticOutputValue output(const FEMesh*,
				       const ElementFuncNodeIterator&)
    const = 0;
  virtual ArithmeticOutputValue output(const FEMesh*, const FuncNode&)
    const = 0;
  virtual void setValueFromOutputValue(FEMesh*, const FuncNode&,
				       const OutputValue*) = 0;

  friend bool operator==(const Field &f1, const Field &f2) {
    return f1.index() == f2.index();
  }

  // getIndex converts the string representation of a FieldIndex (eg,
  // "x" for a vector component) to an actual FieldIndex.  It's only
  // used in python.
  virtual FieldIndex *getIndex(const std::string&) const = 0;

  virtual ComponentsP components(Planarity p=ALL_INDICES) const = 0;
  virtual ComponentsP outOfPlaneComponents() const = 0;

  // Stuff required by fieldeqnlist.h templates, which handle Field
  // data that varies from Node to Node, such as specifying which
  // fields are defined and where they live in the Node's list of
  // values. These classes must be typedef'd so that they can be
  // located by the templates.
  class FieldData : public FieldEqnData {
  public:
    FieldData() : active(false) {}
    bool active;		// is a Field active at this node?
  };
  typedef FieldData FEData;
  // FindAllFieldWrappers is an object that fetches a Mesh's
  // dictionary of Field data.
  class FindAllFieldWrappers {
  private:
    FEMesh *mesh;
  public:
    FindAllFieldWrappers(FEMesh *mesh) : mesh(mesh) {}
    // Using the typedef FEWrapper<Field>::AllWrappers here doesn't
    // work.  The compiler can't untangle the templates.  Is this a
    // compiler bug, or just due to a failure to define everything
    // first?
    std::map<const std::vector<FieldData>*, FEWrapper<Field>*,
	     FEvectorCompare<Field> >
    &operator()();
  };
  typedef FindAllFieldWrappers GetWrappers;

  virtual const std::string &classname() const = 0;


protected:
  virtual ~Field();
};

bool operator<(const Field::FieldData&, const Field::FieldData&);

std::ostream &operator<<(std::ostream &, const Field&);


// Compound fields have two parts: the in-plane part, and the
// out-of-plane part.  The Fields that the rest of the program uses
// are derived from the Compound Fields (eg displacement,
// temperature).  CompoundField needs to be derived from Field to
// override the virtual functions for defining and activating the
// field.  Since the in- and out-of-plane fields also need to be
// derived from Field, we have to use virtual inheritance.

class CompoundField : public virtual Field {
private:
  Field * const zfield_;	// the out-of-plane field
  int cfield_indx;
protected:
  CompoundField(const std::string &name, int dim, Field *outofplane,
		Field *timederiv, Field *outofplanetimederiv);
  virtual ~CompoundField();
public:
  Field *out_of_plane() const {
    return zfield_;
  }
  Field *out_of_plane_time_derivative() const {
    return zfield_->time_derivative();
  }
  bool in_plane(const FEMesh*) const;
  bool in_plane(const CSubProblem*) const;
  virtual void define(CSubProblem*) const;
  virtual void undefine(CSubProblem*) const;
  virtual void activate(CSubProblem*) const;
  virtual void deactivate(CSubProblem*) const;
  virtual void registerProperty(Property*) const;
  static std::vector<CompoundField*> &allcompoundfields();
  // For identification in python...
  virtual const std::string &classname() const = 0;
};

//-\\-//-\\-//-\\-//-\\-//-\\-//-\\-//-\\-//-\\-//-\\-//-\\-//-\\-//

// Different varieties of Fields

// Since CompoundField and ScalarFieldBase are both derived from
// Field, the inheritance must be virtual.

class ScalarFieldBase : public virtual Field {
private:
  static const std::string classname_;
public:
  ScalarFieldBase(const std::string &name) : Field(name, 1) {}
  virtual ~ScalarFieldBase() {}
  DegreeOfFreedom *operator()(const FuncNode*) const;
  DegreeOfFreedom *operator()(const FuncNode &n) const {
    return operator()(&n);
  }
  DegreeOfFreedom *operator()(const ElementFuncNodeIterator&) const;
  virtual DegreeOfFreedom *operator()(const FuncNode*, int) const;
  virtual DegreeOfFreedom *operator()(const FuncNode &n, int i) const {
    return operator()(&n, i);
  }
  virtual DegreeOfFreedom *operator()(const ElementFuncNodeIterator&, int)
    const;
  virtual ArithmeticOutputValue newOutputValue() const;
  virtual ArithmeticOutputValue output(const FEMesh*,
				       const ElementFuncNodeIterator&) const;
  virtual ArithmeticOutputValue output(const FEMesh*, const FuncNode&) const;
  virtual void setValueFromOutputValue(FEMesh*, const FuncNode&,
				       const OutputValue*);

  virtual ComponentsP components(Planarity=ALL_INDICES/*irrelevant*/) const;
  virtual ComponentsP outOfPlaneComponents() const;
  virtual FieldIndex *getIndex(const std::string&) const;
  virtual const std::string &classname() const { return classname_; }
};

class ScalarField : public ScalarFieldBase, public CompoundField 
{
private:
  static const std::string classname_;
public:
  ScalarField(const std::string &name);
  virtual ~ScalarField() {}
  virtual const std::string &classname() const { return classname_; }
};

//------------------------

class TwoVectorFieldBase : public virtual Field {
private:
  static const std::string classname_;
public:
  TwoVectorFieldBase(const std::string &name) : Field(name, 2) {}
  virtual ~TwoVectorFieldBase() {}
  virtual DegreeOfFreedom *operator()(const FuncNode*, int component) const;
  virtual DegreeOfFreedom *operator()(const FuncNode &n, int component) const {
    return operator()(&n, component);
  }
  virtual DegreeOfFreedom *operator()(const ElementFuncNodeIterator&,
				      int component) const;
  virtual ArithmeticOutputValue newOutputValue() const;
  virtual ArithmeticOutputValue output(const FEMesh*,
				       const ElementFuncNodeIterator&) const;
  virtual ArithmeticOutputValue output(const FEMesh*, const FuncNode&) const;
  virtual void setValueFromOutputValue(FEMesh*, const FuncNode&,
				       const OutputValue*);
  virtual ComponentsP components(Planarity /*irrelevant*/) const;
  virtual ComponentsP outOfPlaneComponents() const;
  virtual FieldIndex *getIndex(const std::string&) const;
  virtual const std::string &classname() const { return classname_; }
};

class TwoVectorField : public TwoVectorFieldBase, public CompoundField 
{
private:
  static const std::string classname_;
public:
  TwoVectorField(const std::string &name);
  virtual ~TwoVectorField() {}
  virtual const std::string &classname() const { return classname_; }
};

//------------------------

class VectorFieldBase : public virtual Field {
private:
  static const std::string classname_;
public:
  VectorFieldBase(const std::string &name, int dim)
    : Field(name, dim)
  {}
  virtual const std::string &classname() const { return classname_; }
  virtual ~VectorFieldBase() {}
  virtual DegreeOfFreedom *operator()(const FuncNode*, int component) const;
  DegreeOfFreedom *operator()(const FuncNode &n, int component) const {
    return operator()(&n, component);
  }
  virtual DegreeOfFreedom *operator()(const ElementFuncNodeIterator&,
				      int component) const;
  virtual ArithmeticOutputValue newOutputValue() const;
  virtual ArithmeticOutputValue output(const FEMesh*,
				       const ElementFuncNodeIterator&) const;
  virtual ArithmeticOutputValue output(const FEMesh*, const FuncNode&) const;
  virtual void setValueFromOutputValue(FEMesh*, const FuncNode&,
				       const OutputValue*);
  virtual ComponentsP components(Planarity=ALL_INDICES) const;
  virtual ComponentsP outOfPlaneComponents() const;
  virtual FieldIndex *getIndex(const std::string&) const;
};

// ThreeVectorField, provided as a separate class so that it can be
// recognized by the initializer infrastructure.
class ThreeVectorField : public VectorFieldBase {
private:
  static const std::string classname_;
public:
  ThreeVectorField(const std::string &name);
  //    Field(name, 3), VectorFieldBase(name, 3) {}
  virtual ~ThreeVectorField() {}
  virtual const std::string &classname() const { return classname_; }
};


// A symmetric tensor field, for plastic strain and kinematic
// hardening, and whatever else seems appropriate.
class SymmetricTensorField: public Field {
private:
  static const std::string classname_;
public:
  SymmetricTensorField(const std::string &name) : Field(name, 6) {}
  virtual ~SymmetricTensorField() {}

  virtual DegreeOfFreedom *operator()(const FuncNode*, int comp) const;
  DegreeOfFreedom *operator()(const FuncNode &pd, int comp) const {
    return operator()(&pd, comp);
  }
  virtual DegreeOfFreedom *operator()(const ElementFuncNodeIterator&,
				      int component) const;
  DegreeOfFreedom *operator()(const ElementFuncNodeIterator&,
			      SymTensorIndex&) const;
  DegreeOfFreedom *operator()(const FuncNode&, SymTensorIndex&) const;
  virtual ArithmeticOutputValue newOutputValue() const;
  virtual ArithmeticOutputValue output(const FEMesh*, 
			     const ElementFuncNodeIterator&) const;
  virtual ArithmeticOutputValue output(const FEMesh*, const FuncNode&) const;
  virtual void setValueFromOutputValue(FEMesh*, const FuncNode&,
				       const OutputValue*);
  virtual ComponentsP components(Planarity) const;
  virtual ComponentsP outOfPlaneComponents() const;
  virtual FieldIndex *getIndex(const std::string&) const;

  virtual const std::string &classname() const {
    return classname_;
  }
};


Field *getFieldByIndex(int);
int countFields();
CompoundField *getCompoundFieldByIndex(int);
int countCompoundFields();

void testIterators();

#endif	// FIELD_H
