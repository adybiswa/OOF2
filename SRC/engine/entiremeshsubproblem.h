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

#ifndef ENTIREMESHSUBPROBLEM_H
#define ENTIREMESHSUBPROBLEM_H

#include "engine/csubproblem.h"

// A subproblem that includes the entire Mesh.

class CEntireMeshSubProblem : public CSubProblem {
public:
  CEntireMeshSubProblem();
  virtual ~CEntireMeshSubProblem();
  virtual void redefined() {}

  virtual VContainer<Node>* c_nodes() const;
  virtual VContainer<FuncNode>* c_funcnodes() const;
  virtual VContainer<Element>* c_elements() const;
  virtual VContainer<InterfaceElement>* c_interface_elements() const;

  virtual bool contains(const Element*) const;
  virtual bool containsNode(const Node*) const;
  virtual MaterialSet *getMaterials() const;
};

#endif // ENTIREMESHSUBPROBLEM_H
