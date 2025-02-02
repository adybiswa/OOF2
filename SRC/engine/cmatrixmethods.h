// -*- C++ -*-

/* This software was produced by NIST, an agency of the U.S. government,
 * and by statute is not subject to copyright in the United States.
 * Recipients of this software assume all responsibilities associated
 * with its operation, modification and maintenance. However, to
 * facilitate maintenance we ask that before distributing modified
 * versions of this software, you first contact the authors at
 * oof_manager@nist.gov. 
 */

// Swiggable C++ Wrappers for the low level Eigen's matrix solvers.

#ifndef CMATRIXMETHODS_H
#define CMATRIXMETHODS_H

#include <oofconfig.h>
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"
#include "Eigen/SparseQR"
#include "Eigen/OrderingMethods"
#include "engine/sparsemat.h"

// TODO: Progress bars are hacked into Eigen solvers.  Do they work?
// Search for "OOF" in the Eigen source code to find the hacks.

enum class Precond {Uncond=1, Diag=2, ILUT=3, IC=4};

enum Info {
  SUCCESS = Eigen::Success,
  NUMERICAL = Eigen::NumericalIssue,
  NOCONVERG = Eigen::NoConvergence,
  INVALID_INPUT = Eigen::InvalidInput
};

template<Precond P> class CG;
template<Precond P> class BiCGStab;
class SimplicialLLT;
class SimplicialLDLT;
class SparseLU;
class SparseQR;

namespace internal {
// Iterative solver traits

template<typename Derived> struct IterSolverTrait;

// ESMat is an Eigen SparseMatrix typedef'd in sparsemat.h.

template<> struct IterSolverTrait<CG<Precond::Uncond>> {
  typedef Eigen::ConjugateGradient< ESMat,
    Eigen::Lower | Eigen::Upper,
    Eigen::IdentityPreconditioner > Type;
};

template<> struct IterSolverTrait<CG<Precond::Diag>> {
  typedef Eigen::ConjugateGradient< ESMat,
    Eigen::Lower | Eigen::Upper,
    Eigen::DiagonalPreconditioner<double> > Type;
};

template<> struct IterSolverTrait<CG<Precond::ILUT>> {
  typedef Eigen::ConjugateGradient< ESMat,
    Eigen::Lower | Eigen::Upper,
    Eigen::IncompleteLUT<double> > Type;
};

template<> struct IterSolverTrait<CG<Precond::IC>> {
  typedef Eigen::ConjugateGradient< ESMat,
    Eigen::Lower | Eigen::Upper,
    Eigen::IncompleteCholesky<double> > Type;
};

template<> struct IterSolverTrait<BiCGStab<Precond::Uncond>> {
  typedef Eigen::BiCGSTAB< ESMat, Eigen::IdentityPreconditioner > Type;
};

template<> struct IterSolverTrait<BiCGStab<Precond::Diag>> {
  typedef Eigen::BiCGSTAB< ESMat, Eigen::DiagonalPreconditioner<double> > Type;
};

template<> struct IterSolverTrait<BiCGStab<Precond::ILUT>> {
  typedef Eigen::BiCGSTAB< ESMat, Eigen::IncompleteLUT<double> > Type;
};

template<> struct IterSolverTrait<BiCGStab<Precond::IC>> {
  typedef Eigen::BiCGSTAB< ESMat, Eigen::IncompleteCholesky<double> > Type;
};

// Direct solver traits

template<typename Derived> struct DirectSolverTrait;

template<> struct DirectSolverTrait<SimplicialLLT> {
  typedef Eigen::SimplicialLLT<ESMat> Type;
};

template<> struct DirectSolverTrait<SimplicialLDLT> {
  typedef Eigen::SimplicialLDLT<ESMat> Type;
};

template<> struct DirectSolverTrait<SparseLU> {
  typedef Eigen::SparseLU<ESMat> Type;
};

template<> struct DirectSolverTrait<SparseQR> {
  typedef Eigen::SparseQR<ESMat, Eigen::COLAMDOrdering<int>> Type;
};
} // end namespace internal

// Iterative solvers

template<typename Derived>
class IterativeSolver {
private:
  typename internal::IterSolverTrait<Derived>::Type solver_;
public:
  void analyze_pattern(const SparseMat& m) {
    solver_.analyzePattern(m.data);
  }

  void factorize(const SparseMat& m) {
    solver_.factorize(m.data);
  }

  void compute(const SparseMat& m) {
    if(m.ncols() != m.nrows()) {
      std::cerr << "IterativeSolver::compute: rows=" << m.nrows() << " cols="
		<< m.ncols() << std::endl;
      throw ErrSetupError("Matrix is not square!");
    }
    solver_.compute(m.data);
  }

  DoubleVec solve(const DoubleVec& rhs) {
    DoubleVec x;
    x.data = solver_.solve(rhs.data);
    return x;
  }

  DoubleVec solve(const SparseMat& m, const DoubleVec& rhs) {
    if(m.ncols() != m.nrows()) {
      std::cerr << "IterativeSolver::solve: rows=" << m.nrows() << " cols="
		<< m.ncols() << " r=" << rhs.size() << std::endl;
      throw ErrSetupError("Matrix is not square!");
    }
    DoubleVec x;
    solver_.compute(m.data);
    x.data = solver_.solve(rhs.data);
    return x;
  }

  int solve(const SparseMat& m, const DoubleVec& rhs, DoubleVec& x) {
    if(m.ncols() != m.nrows()) {
      std::cerr << "IterativeSolver::solve: rows=" << m.nrows() << " cols="
		<< m.ncols() << " r=" << rhs.size() << std::endl;
      throw ErrSetupError("Matrix is not square!");
    }
    solver_.compute(m.data);
    x.data = solver_.solve(rhs.data);
    return solver_.info();
  }

  void set_max_iterations(int iters) { solver_.setMaxIterations(iters); }
  int max_iterations() const { return solver_.maxIterations(); }
  void set_tolerance(double tol) { solver_.setTolerance(tol); }
  double tolerance() const { return solver_.tolerance(); }
  int iterations() const { return solver_.iterations(); }
  double error() const { return solver_.error(); }
  int info() const { return solver_.info(); }
};

template<Precond P>
class CG : public IterativeSolver<CG<P>> {
  friend class IterativeSolver<CG<P>>;
};
template class CG<Precond::Uncond>;
template class CG<Precond::Diag>;
template class CG<Precond::ILUT>;
template class CG<Precond::IC>;

template <Precond P>
class BiCGStab : public IterativeSolver<BiCGStab<P>> {
  friend class IterativeSolver<BiCGStab<P>>;
};
template class BiCGStab<Precond::Uncond>;
template class BiCGStab<Precond::Diag>;
template class BiCGStab<Precond::ILUT>;
template class BiCGStab<Precond::IC>;

// Direct solvers

template <typename Derived> class DirectSolver {
protected:
  typename internal::DirectSolverTrait<Derived>::Type solver_;
public:
  void analyze_pattern(const SparseMat& m) {
    solver_.analyzePattern(m.data);
  }

  void factorize(const SparseMat& m) { 
    solver_.factorize(m.data);
  }

  void compute(const SparseMat& m) {
    if(m.nrows() != m.ncols()) {
      std::cerr << "DirectSolver::compute: nrows=" << m.nrows()
		<< " ncols=" << m.ncols() << std::endl;
      throw ErrSetupError("Matrix is not square!");
    }
    solver_.compute(m.data);
  }

  DoubleVec solve(const DoubleVec& rhs) {
    DoubleVec x;
    x.data = solver_.solve(rhs.data);
    return x;
  }

  DoubleVec solve(const SparseMat& m, const DoubleVec& rhs) {
    if(m.nrows() != m.ncols()) {
      std::cerr << "DirectSolver::solve: nrows=" << m.nrows()
		<< " ncols=" << m.ncols() << std::endl;
      throw ErrSetupError("Matrix is not square!");
    }
    DoubleVec x;
    solver_.compute(m.data);
    x.data = solver_.solve(rhs.data);
    return x;
  }

  int solve(const SparseMat& m, const DoubleVec& rhs, DoubleVec& x) {
    if(m.nrows() != m.ncols()) {
      std::cerr << "DirectSolver::solve: nrows=" << m.nrows()
		<< " ncols=" << m.ncols() << std::endl;
      throw ErrSetupError("Matrix is not square!");
    }
    solver_.compute(m.data);
    x.data = solver_.solve(rhs.data);
    return solver_.info();
  }

  int info() {
    return solver_.info();
  }
};

class SimplicialLLT : public DirectSolver<SimplicialLLT> {
  friend class DirectSolver<SimplicialLLT>;
};

class SimplicialLDLT : public DirectSolver<SimplicialLDLT> {
  friend class DirectSolver<SimplicialLDLT>;
};

class SparseLU : public DirectSolver<SparseLU> {
  friend class DirectSolver<SparseLU>;
};

class SparseQR : public DirectSolver<SparseQR> {
  friend class DirectSolver<SparseQR>;
};

#endif // CMATRIXMETHODS_H
