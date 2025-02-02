// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

#ifndef MASTERELEMENT_SWG
#define MASTERELEMENT_SWG

%module(package="ooflib.SWIG.engine") masterelement

%include "common/typemaps.swg"
%import "engine/element.swg"
%import "engine/mastercoord.swg"
%import "engine/material.swg"

%{
#include "common/tostring.h"
#include "engine/cskeleton.h"  
#include "engine/element.h"
#include "engine/masterelement.h"
#include "engine/node.h"
  #include "engine/flux.h"	// SWIGBUG
%}

%nodefaultctor ProtoNode;

class ProtoNode {
public:
  int index();
  bool mapping();
  bool func();
  bool corner();
  int nedges();
  int getedge(int i);
  const MasterCoord &mastercoord();
};

MAKE_LISTVEC_TYPEMAPS(MasterElement);
MAKE_LISTVEC_TYPEMAPS(Node);

// Multiargument typemap to convert a single python SkeletonElement to
// two C++ arguments, a PyObject* and a CSkeletonElement*.  The
// Element constructor wants to know both the SkeletonElement's
// PyObject (so that it can hold a reference to it) and the
// CSkeletonElement (to access its methods).
// MasterElement::buildElement and
// MasterElement::buildInterfaceElement are the swigged C++ functions
// that call the Element constructor.

%typemap(in) (PyObject *skelel, CSkeletonElement *cskelel) {
  // typemap(in) (PyObject *skelel, CSkeletonElement *cskelel)
  $1 = $input;
  void *ptr = 0;
  int res = SWIG_ConvertPtr($input, &ptr, $descriptor(CSkeletonElement*),
			    0);
  if(!SWIG_IsOK(res)) {
    PyErr_SetString(PyExc_TypeError,
		    "in method '$symname', expecting a CSkeletonElement*");
    SWIG_fail;
  }
  $2 = reinterpret_cast<CSkeletonElement*>(ptr);
}



// Do *not* use %newobject here. C++ retains ownership of the list,
// which is static inside the function.
MasterElementRefVec* masterElementList();
// MasterElements are never created or destroyed in Python.
%nodefaultctor MasterElement;
%nodefaultdtor MasterElement;

class MasterElement {
public:
  int id();
  int nnodes();
  int nmapnodes();
  int nfuncnodes();
  int ninteriorfuncnodes();
  int ninteriormapnodes_only();
  int nexteriormapnodes_only();
  int nsides();
  const std::string &name();
  const std::string &description();
  int ncorners();
  double outOfBounds(const MasterCoord&);
  bool onBoundary(const MasterCoord&);
  bool onBoundary2(const MasterCoord&, const MasterCoord&);

  InterfaceElement *buildInterfaceElement(PyObject *skelel,
					  CSkeletonElement *cskelel,
					  PyObject * skelel,
					  CSkeletonElement *cskelel,
					  int,
					  Material*,
					  NodeVec*, NodeVec*,
					  bool,bool,
					  std::vector<std::string>*);
  int map_order();
  int fun_order();

  Element *build(PyObject *skelel, CSkeletonElement *cskelel,
		 Material*, NodeVec*);
  
  // Superconvergent patch recovery
  int nSCpoints();
  MasterCoord getSCpoint(int);
};

%extend MasterElement {
  const ProtoNode *get_protonode(int n) {
    return self->protonode(n);
  }
}

ADD_REPR_GENERIC(MasterElement);

int integration_reduction;

%pythoncode "engine/masterelement.spy"

#endif // MASTERELEMENT_SWG
