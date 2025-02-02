# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from . import memorycheck
import unittest
import os

from .UTILS.file_utils import reference_file

class OOF_AMR(unittest.TestCase):
    def setUp(self):
        pass

    @memorycheck.check('el_shape.png')
    def RefineMesh(self):
        from ooflib.engine import mesh

        OOF.File.Load.Data(
            filename=reference_file("mesh_data", "el_shape2.mesh"))
        meshctxt = mesh.meshes['el_shape.png:skeleton:mesh']
        skelctxt = meshctxt.getParent()
        self.assertEqual(meshctxt.nelements(), 400)
        self.assertEqual(meshctxt.nnodes(), 1281)

        OOF.Subproblem.Set_Solver(
            subproblem='el_shape.png:skeleton:mesh:default',
            solver_mode=AdvancedSolverMode(
                nonlinear_solver=NoNonlinearSolver(),
                time_stepper=StaticDriver(),
                symmetric_solver=ConjugateGradient(
                    preconditioner=ILUPreconditioner(),
                    tolerance=1e-13, max_iterations=1000)))
        OOF.Mesh.Solve(
            mesh='el_shape.png:skeleton:mesh', endtime=0.0)

        OOF.Skeleton.Modify(
            skeleton='el_shape.png:skeleton',
            modifier=Refine(
                targets=AdaptiveMeshRefine(
                    subproblem='el_shape.png:skeleton:mesh:default',
                    estimator=ZZ_Estimator(
                        norm=L2ErrorNorm(),flux=Stress,threshold=10)
                    ),
                divider=Trisection(minlength=0.0),
                rules='Quick',
                alpha=0.3))
        # Because adaptive mesh refinement is susceptible to roundoff
        # errors, the number of elements after refinement is not
        # predictable.  This just checks that it's greater the
        # pre-refinement count.
        newNel = len(skelctxt.getObject().elements)
        self.assertTrue(newNel > 400)
        self.assertEqual(meshctxt.nelements(), 400)
        self.assertEqual(meshctxt.nnodes(), 1281)
        self.assertTrue(meshctxt.outOfSync())

        OOF.Mesh.Modify(mesh='el_shape.png:skeleton:mesh',
                        modifier=RebuildMesh())
        self.assertEqual(meshctxt.nelements(), newNel)
        self.assertEqual(len(skelctxt.getObject().elements), newNel)
        # The mesh uses 8 node quads and 6 node triangles, so the new
        # number of Mesh nodes is not equal to the new number of
        # Skeleton nodes. We don't bother checking the number, since
        # it's also subject to roundoff error.
        self.assertTrue(not meshctxt.outOfSync())

        OOF.Skeleton.Undo(skeleton='el_shape.png:skeleton')
        self.assertEqual(meshctxt.nelements(), newNel)
        self.assertEqual(len(skelctxt.getObject().elements), 400)
        self.assertTrue(meshctxt.outOfSync())

        OOF.Mesh.Modify(mesh='el_shape.png:skeleton:mesh',
                        modifier=RebuildMesh())
        self.assertTrue(not meshctxt.outOfSync())
        self.assertEqual(meshctxt.nelements(), 400)
        self.assertEqual(len(skelctxt.getObject().elements), 400)
        self.assertEqual(meshctxt.nnodes(), 1281)

        OOF.Material.Delete(name="green-material")
        OOF.Property.Delete(
            property="Mechanical:Elasticity:Isotropic:green_elasticity")

    @memorycheck.check('el_shape.png')
    def RefineSubproblem(self):
        from ooflib.engine import mesh
        OOF.File.Load.Data(filename=reference_file("mesh_data",
                                                   "el_shape2.mesh"))
        meshctxt = mesh.meshes['el_shape.png:skeleton:mesh']
        skelctxt = meshctxt.getParent()
        OOF.Windows.Graphics.New()
        OOF.Graphics_1.Toolbox.Pixel_Select.Rectangle(
            source='el_shape.png:el_shape.png',
            points=[Point(0.222,9.736), Point(2.728,3.110)],
            shift=0, ctrl=0)
        OOF.PixelGroup.New(name='upperpixels', microstructure='el_shape.png')
        OOF.PixelGroup.AddSelection(microstructure='el_shape.png', 
                                    group='upperpixels')
        OOF.Subproblem.New(name='subproblem',
                           mesh='el_shape.png:skeleton:mesh',
                           subproblem=PixelGroupSubProblem(group='upperpixels'))
        OOF.Subproblem.Set_Solver(
            subproblem='el_shape.png:skeleton:mesh:default',
            solver_mode=AdvancedSolverMode(
                nonlinear_solver=NoNonlinearSolver(),
                time_stepper=StaticDriver(),
                symmetric_solver=ConjugateGradient(
                    preconditioner=ILUPreconditioner(),
                    tolerance=1.e-13, max_iterations=1000)))
        OOF.Mesh.Solve(
            mesh='el_shape.png:skeleton:mesh', endtime=0.0)
        OOF.Subproblem.Copy_Field_State(
            source='el_shape.png:skeleton:mesh:default',
            target='el_shape.png:skeleton:mesh:subproblem')
        OOF.Subproblem.Copy_Equation_State(
            source='el_shape.png:skeleton:mesh:default',
            target='el_shape.png:skeleton:mesh:subproblem')
        OOF.Skeleton.Modify(
            skeleton='el_shape.png:skeleton',
            modifier=Refine(
                targets=AdaptiveMeshRefine(
                    subproblem='el_shape.png:skeleton:mesh:subproblem',
                    estimator=ZZ_Estimator(
                        norm=L2ErrorNorm(),flux=Stress,threshold=10)
                ),
                divider=Trisection(minlength=0.0),
                rules='Quick',
                alpha=0.3))
        self.assertEqual(meshctxt.nelements(), 400)
        newNel = len(skelctxt.getObject().elements)
        self.assertTrue(newNel > 400)

        OOF.Mesh.Modify(mesh='el_shape.png:skeleton:mesh',
                        modifier=RebuildMesh())
        self.assertEqual(meshctxt.nelements(), newNel)
        
        OOF.Graphics_1.File.Close()
        OOF.Material.Delete(name="green-material")
        OOF.Property.Delete(
            property="Mechanical:Elasticity:Isotropic:green_elasticity")

test_set = [
    OOF_AMR("RefineMesh"),
    OOF_AMR("RefineSubproblem")
]
