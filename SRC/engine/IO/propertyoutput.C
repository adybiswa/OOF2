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

#include "common/printvec.h"	// for debugging
#include "common/pythonlock.h"
#include "common/pyutils.h"
#include "engine/IO/propertyoutput.h"
#include "engine/corientation.h"
#include "engine/element.h"
#include "engine/femesh.h"
#include "engine/material.h"
#include "engine/outputval.h"
#include "engine/property.h"
#include "engine/symmmatrix.h"

// TODO LATER: PropertyOutputs should NOT be computed by the
// Properties directly.  That is, there should not be a
// Property::output function.  This is because a user may want to add
// a PropertyOutput to an existing Property without modifying the
// Property itself.  Instead, PropertyOutputRegistrations should have
// a mechanism for being told which Properties contribute to them, and
// to specify a callback function for each Property.  The callback
// does the actual computation, instead of Property::output. This may
// be harder than it looks...  Presumably the registration of the
// Property and callback should be done from Python, but the callback
// itself must be C++.  Should there be a swigged
// PropertyOutputContribution class?  Each subclass will have to have
// a singleton instance which will be swigged.

const std::string ArithmeticPropertyOutput::classname_(
				       "ArithmeticPropertyOutput");
const std::string NonArithmeticPropertyOutput::classname_(
					  "NonArithmeticPropertyOutput");

std::vector<PropertyOutputRegistration*> &
PropertyOutputRegistration::allPropertyOutputRegs()
{
  static std::vector<PropertyOutputRegistration*> vec;
  return vec;
}

PropertyOutputRegistration::PropertyOutputRegistration(
					       const std::string &name) 
  : name_(name),
    index_(allPropertyOutputRegs().size())
{
  allPropertyOutputRegs().push_back(this);
}

// Count existing PropertyOutputRegistrations.

int nPropertyOutputRegistrations() {
  return PropertyOutputRegistration::allPropertyOutputRegs().size();
}

// Retrieve a PropertyOutputRegistration by name.

PropertyOutputRegistration *getPropertyOutputReg(const std::string &name) {
  std::vector<PropertyOutputRegistration*> &apor =
    PropertyOutputRegistration::allPropertyOutputRegs();
  for(std::vector<PropertyOutputRegistration*>::iterator i=apor.begin();
      i<apor.end(); ++i)
    {
      if(name == (*i)->name())
	return *i;
    }
  return 0;
}

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

PropertyOutput::PropertyOutput(const std::string &name, PyObject *params)
  : name_(name),
    params_(params),
    index_(getPropertyOutputReg(name)->index()),
    initializer(nullptr)
{
  PYTHON_THREAD_BEGIN_BLOCK;
  Py_XINCREF(params_);
}

PropertyOutput::~PropertyOutput() {
  PYTHON_THREAD_BEGIN_BLOCK;
  Py_XDECREF(params_);
}

std::vector<NonArithmeticOutputValue> *
NonArithmeticPropertyOutput::evaluate(
			      FEMesh *mesh, Element *element,
			      const std::vector<MasterCoord*> *mc)
{
  std::vector<NonArithmeticOutputValue> *results =
    new std::vector<NonArithmeticOutputValue>;
  results->reserve(mc->size());
  const Material *material = element->material();
  if(material) {
    const std::vector<Property*> &props = material->outputProperties(this);
    // material->begin_element(mesh, element);

    // Loop over points within the element.
    for(auto i=mc->begin(); i!=mc->end(); ++i) {
      NonArithmeticOutputVal *data = dynamic_cast<NonArithmeticOutputVal*>(
				      (*initializer)(this, mesh, element, **i));
      // Loop over properties that contribute to this output.
      for(std::vector<Property*>::size_type p=0; p<props.size(); p++) {
	// *Don't* check for active properties here; just loop over
	// all of them.  Properties are active or inactive on
	// SubProblems, not FEMeshes, and we don't have a SubProblem
	// here.
	props[p]->output(mesh, element, this, **i, data); // adds to data
      }	// end loop over Properties
      results->emplace_back(data);
      
    } // end loop over points
    // material->end_element(mesh, element);
  }
  else {			// no material!
    for(auto i=mc->begin(); i!=mc->end(); ++i) {
      NonArithmeticOutputVal *data = dynamic_cast<NonArithmeticOutputVal*>(
				      (*initializer)(this, mesh, element, **i));
      results->emplace_back(data);
    }
  }
  return results;
}

std::vector<ArithmeticOutputValue> *
ArithmeticPropertyOutput::evaluate(FEMesh *mesh, Element *element,
				   const std::vector<MasterCoord*> *mc)
{
  std::vector<ArithmeticOutputValue> *results =
    new std::vector<ArithmeticOutputValue>;
  results->reserve(mc->size());
  const Material *material = element->material();
  if(material) {
    const std::vector<Property*> &props = material->outputProperties(this);
    // material->begin_element(mesh, element);

    // Loop over points within the element.
    for(auto i=mc->begin(); i!=mc->end(); ++i) {
      //	ArithmeticOutputVal *data((*init)(this, mesh, element, **i));
      ArithmeticOutputVal *data = dynamic_cast<ArithmeticOutputVal*>(
				     (*initializer)(this, mesh, element, **i));
      // Loop over properties that contribute to this output.
      for(std::vector<Property*>::size_type p=0; p<props.size(); p++) {
	// *Don't* check for active properties here; just loop over
	// all of them.  Properties are active or inactive on
	// SubProblems, not FEMeshes, and we don't have a SubProblem
	// here.
	props[p]->output(mesh, element, this, **i, data); // adds to data
      }	// end loop over Properties
      results->emplace_back(data);
      
    } // end loop over points
    // material->end_element(mesh, element);
  }
  else {			// no material!
    for(auto i=mc->begin(); i!=mc->end(); ++i) {
      ArithmeticOutputVal *data = dynamic_cast<ArithmeticOutputVal*>(
				 (*initializer)(this, mesh, element, **i));
      results->emplace_back(data);
    }
  }
  return results;
}


// TODO: With all the locking and calling to Python, these getXXXParam
// methods are sure to be slow.  They're called by the PropertyOutputs
// at every output point.  The parameter values should be somehow
// cached in C++.

double PropertyOutput::getFloatParam(const char *name) const {
  double x;
  PYTHON_THREAD_BEGIN_BLOCK;
  // params_ is a dictionary
  PyObject *obj = PyMapping_GetItemString(params_, (char*) name);
  if(!obj)
    pythonErrorRelay();
  x = PyFloat_AsDouble(obj);
  Py_XDECREF(obj);
  return x;
}

int PropertyOutput::getIntParam(const char *name) const {
  int x;
  PYTHON_THREAD_BEGIN_BLOCK;
  // params_ is a dictionary
  PyObject *obj = PyMapping_GetItemString(params_, (char*) name);
  if(!obj)
    pythonErrorRelay();
  x = PyInt_AsLong(obj);
  Py_XDECREF(obj);
  return x;
}

std::vector<std::string>*
PropertyOutput::getListOfStringsParam(const char *name) const {
  std::vector<std::string> *x = new std::vector<std::string>;
  PYTHON_THREAD_BEGIN_BLOCK;
  // params_ is a dictionary
  PyObject *obj = PyMapping_GetItemString(params_, (char*) name);
  if(!obj)
    pythonErrorRelay();
  // obj is a Python list or tuple
  assert(PySequence_Check(obj));
  Py_ssize_t n = PySequence_Size(obj);
  for(Py_ssize_t i=0; i<n; i++) {
    PyObject *item = PySequence_GetItem(obj, i);
    if(!item)
      pythonErrorRelay();
    x->emplace_back(pyStringAsString(item));
    Py_XDECREF(item);
  }
  Py_XDECREF(obj);
  return x;
}

const std::string *PropertyOutput::getStringParam(const char *name) const {
  PYTHON_THREAD_BEGIN_BLOCK;
  // params_ is a dictionary
  PyObject *obj = PyMapping_GetItemString(params_, (char*) name);
  if(!obj)
    pythonErrorRelay();
  const std::string *x = new std::string(pyStringAsString(obj));
  Py_XDECREF(obj);
  return x;
}

const std::string *PropertyOutput::getEnumParam(const char *name) const {
   PYTHON_THREAD_BEGIN_BLOCK;
  // params_ is a dictionary
  PyObject *obj = PyMapping_GetItemString(params_, (char*) name);
  if(!obj) {
    pythonErrorRelay();
  }
  PyObject *str = PyObject_GetAttrString(obj, (char*) "name");
  if(!str) {
    Py_XDECREF(obj);
    pythonErrorRelay();
  }
  std::string *x = new std::string(pyStringAsString(str));
  Py_XDECREF(str);
  Py_XDECREF(obj);
  return x;
}

const std::string *PropertyOutput::getRegisteredParamName(const char *name)
  const
{
  PYTHON_THREAD_BEGIN_BLOCK;
  // Get the parameter out of the dictionary
  PyObject *obj = PyMapping_GetItemString(params_, (char*) name);
  if(!obj)
    pythonErrorRelay();
  // Call the parameter's "name" function"
  PyObject *str = PyObject_CallMethod(obj, (char*) "name", 0);
  if(!str) {
    Py_XDECREF(obj);
    pythonErrorRelay();
  }
  const std::string *x = new std::string(pyStringAsString(str));
  Py_XDECREF(str);
  Py_XDECREF(obj);
  return x;
}

//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//
//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//=\\=//

const std::string ArithmeticPropertyOutputInit::classname_(
					   "ArithmeticPropertyOutputInit");
const std::string NonArithmeticPropertyOutputInit::classname_(
					   "NonArithmeticPropertyOutputInit");


// The instantiate() methods are called in PORegBase.opfunc to create
// a PropertyOutput before passing it to the elements for evaluation.
// (PORegBase is a Python class derived from
// PropertyOutputRegistration.)

PropertyOutput *ArithmeticPropertyOutputInit::instantiate(
					  const std::string &name,
					  PyObject *params)
  const
{
  return new ArithmeticPropertyOutput(name, params);
}

PropertyOutput *NonArithmeticPropertyOutputInit::instantiate(
					     const std::string &name,
					     PyObject *params)
  const
{
  return new NonArithmeticPropertyOutput(name, params);
}
