# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.common import registeredclass
from ooflib.common.IO import parameter
from ooflib.common.IO import xmlmenudump

## In the Eigen matrix method templates, the preconditioner isn't a
## separate object from the solver.  The Preconditioner subclasses
## here are just placeholders that are used in the
## PreconditionedMatrixMethod classes in matrixmethod.py to choose the
## correct Eigen routine.

class Preconditioner(registeredclass.RegisteredClass):
    registry = []
    tip = "Preconditioners for efficient solution of matrix equations."
    discussion = xmlmenudump.loadFile('DISCUSSIONS/engine/reg/preconditioner.xml')
    xrefs=["RegisteredClass-ConjugateGradient",
           "RegisteredClass-StabilizedBiConjugateGradient"]
    def shortrepr(self):
        return self.__class__.__name__

class UnPreconditioner(Preconditioner):
    pass

class JacobiPreconditioner(Preconditioner):
    pass

class ILUTPreconditioner(Preconditioner):
    pass

class ILUPreconditioner(Preconditioner):
    # The ILU preconditioner is implemented by ILUT.  Eigen doesn't
    # have a separate ILU.  See the solver_maps in matrixmethod.py.
    # ILU is provided for backwards compatibility with old scripts.
    # It's registration is secret so that it doesn't appear in the
    # GUI.
    pass

class ICPreconditioner(Preconditioner):
    pass

registeredclass.Registration(
    "Null",
    Preconditioner,
    UnPreconditioner,
    ordering=2000,
    params=[],
    tip="Be bold (or foolhardy) and attempt to solve the mesh without a preconditioner",
    discussion="""<para>
    Choosing this option for the <varname>preconditioner</varname>
    of a <xref linkend="RegisteredClass-MatrixMethod"/> skips the
    preconditioning step, probably making the solution slower.
    </para>"""
)

registeredclass.Registration(
    "Jacobi",
    Preconditioner,
    JacobiPreconditioner,
    ordering=500,
    params=[],
    tip="A light-weight preconditioner, that simply inverts the diagonal part of the matrix.",
    discussion="""<para>
    This algorithm is provided by <ulink
    url="http://eigen.tuxfamily.org/" role ="external">Eigen</ulink>.
    See <ulink
    url="https://eigen.tuxfamily.org/dox/classEigen_1_1DiagonalPreconditioner.html"
    role="external"/>.
    </para>"""
)

registeredclass.Registration(
    "Incomplete LUT",
    Preconditioner,
    ILUTPreconditioner,
    ordering=101,
    params=[],
    tip="Incomplete LU-factorization with dual thresholding.",
    discussion="""<para>
    This algorithm is provided by <ulink
    url="http://eigen.tuxfamily.org/" role ="external">Eigen</ulink>.
    See <ulink
    url="https://eigen.tuxfamily.org/dox/classEigen_1_1IncompleteLUT.html"
    role="external"/>.
    </para>"""
)

registeredclass.Registration(
    "ILU",
    Preconditioner,
    ILUPreconditioner,
    ordering=101,
    params=[],
    secret=True,
    tip="ILU is not supported. It points to IncompleteLUT instead."
) 

registeredclass.Registration(
    "Incomplete Cholesky",
    Preconditioner,
    ICPreconditioner,
    ordering=100,
    params=[],
    tip="Incomplete Cholesky factorization with dual thresholding.",
    discussion="""<para>
    This algorithm is provided by <ulink
    url="http://eigen.tuxfamily.org/" role ="external">Eigen</ulink>.
    See <ulink
    url="https://eigen.tuxfamily.org/dox/classEigen_1_1IncompleteCholesky.html"
    role="external"/>.
    </para>"""
) 

class PreconditionerParameter(parameter.RegisteredParameter):
    pass
