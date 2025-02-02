# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

# Tests of nonlinear solvers on nonlinear static and quasistatic
# problems with exact solutions.

import unittest, os
from . import memorycheck
import math
from .UTILS import file_utils
reference_file = file_utils.reference_file
from .exact_solns import *
#file_utils.generate = True

class RotatingSquare(unittest.TestCase):
    def setUp(self):
        global utils
        from ooflib.common import utils
        utils.OOFdefine('omega', math.pi/4.)
        utils.OOFdefine('phi0', math.pi/4.)
        OOF.File.Load.Data(filename=reference_file("mesh_data",
                                                 "rotatingsquare1.mesh"))
    def tearDown(self):
        OOF.Material.Delete(name="material")

    @memorycheck.check("microstructure")
    def Static(self):
        OOF.Mesh.Apply_Field_Initializers_at_Time(
            mesh='microstructure:skeleton:mesh', time=1)
        OOF.Mesh.Solve(
            mesh='microstructure:skeleton:mesh', endtime=1.0)
        OOF.Mesh.Analyze.Direct_Output(
            mesh='microstructure:skeleton:mesh',
            time=latest,
            data=getOutput('Field:Value',field=Displacement),
            domain=SinglePoint(point=Point(1,0)),
            sampling=DiscretePointSampleSet(show_x=True,show_y=True),
            destination=OutputStream(filename='rotate.out',mode='w'))
        self.assertTrue(file_utils.compare_last(
                'rotate.out',
                [1, 0, 1/sqrt(2.)-1, 1/sqrt(2.)]))
        OOF.Mesh.Analyze.Direct_Output(
            mesh='microstructure:skeleton:mesh',
            time=latest,
            data=getOutput('Field:Value',field=Displacement),
            domain=SinglePoint(point=Point(0,1)),
            sampling=DiscretePointSampleSet(show_x=True,show_y=True),
            destination=OutputStream(filename='rotate.out',mode='w'))
        self.assertTrue(file_utils.compare_last(
                'rotate.out',
                [0, 1, -1/sqrt(2.), 1/sqrt(2.)-1]))
        file_utils.remove('rotate.out')

    @memorycheck.check("microstructure")
    def QuasiStatic(self):
        OOF.Mesh.Scheduled_Output.New(
            mesh='microstructure:skeleton:mesh',
            name='LRcorner',
            output=BulkAnalysis(
                output_type='Aggregate',
                data=getOutput('Field:Value',field=Displacement),
                operation=DirectOutput(),
                domain=SinglePoint(point=Point(1,0)),
                sampling=DiscretePointSampleSet(show_x=False,show_y=False)))
        OOF.Mesh.Scheduled_Output.Schedule.Set(
            mesh='microstructure:skeleton:mesh',
            output='LRcorner',
            scheduletype=AbsoluteOutputSchedule(),
            schedule=Periodic(delay=0.0,interval=0.1))
        OOF.Mesh.Scheduled_Output.Destination.Set(
            mesh='microstructure:skeleton:mesh',
            output='LRcorner',
            destination=OutputStream(filename='rotate.out',mode='w'))
        OOF.Mesh.Apply_Field_Initializers_at_Time(
            mesh='microstructure:skeleton:mesh', time=0)
        OOF.Mesh.Solve(
            mesh='microstructure:skeleton:mesh', endtime=1.0)
        self.assertTrue(file_utils.fp_file_compare(
            'rotate.out',
            os.path.join('mesh_data', 'rotatingsquare1.out'),
            tolerance=1.e-8))
        file_utils.remove('rotate.out')



class NonlinearPropertyTest(unittest.TestCase):
    def tearDown(self):
        OOF.Material.Delete(name="material")

    def setUp(self):
        global outputdestination
        from ooflib.engine.IO import outputdestination

        self.numX = 32
        self.numY = 32
        self.time = 0.0

        OOF.Microstructure.New(
            name='microstructure',
            width=1.0, height=1.0, width_in_pixels=10, height_in_pixels=10)
        OOF.Material.New(
            name='material', material_type='bulk')
        OOF.Skeleton.New(
            name='skeleton', microstructure='microstructure',
            x_elements=self.numX, y_elements=self.numY,
            skeleton_geometry=QuadSkeleton(
                left_right_periodicity=False,top_bottom_periodicity=False))
        OOF.Mesh.New(
            name='mesh',
            skeleton='microstructure:skeleton',
            element_types=['D2_2', 'T3_3', 'Q4_4'])

        self.heat_solns = exact_solns["scalar"]
        self.elasticity_solns = exact_solns["vector2D"]

        self.boundary_condition_count = 0


    def setBoundaryConditions(self,BC_type,BC_field,BC_equation,BC_list):

        BC_no = 0
        for BC in BC_list:

            BC_no = BC_no + 1

            if BC_type == 'Dirichlet':
                new_BC = DirichletBC(
                    field           = BC_field,
                    field_component = BC[1],
                    equation        = BC_equation,
                    eqn_component   = BC[2],
                    profile         = ContinuumProfileXTd(
                                          function        = BC[3],
                                          timeDerivative  = BC[4],
                                          timeDerivative2 = BC[5]),
                    boundary        = BC[0])

            elif BC_type == 'Neumann':
                new_BC = NeumannBC(
                    field           = BC_field,
                    field_component = BC[1],
                    equation        = BC_equation,
                    eqn_component   = BC[2],
                    profile         = ContinuumProfileXTd(
                                          function        = BC[3],
                                          timeDerivative  = BC[4],
                                          timeDerivative2 = BC[5]),
                    boundary        = BC[0])

            OOF.Mesh.Boundary_Conditions.New(
                name = 'bc<' + str(BC_no) + '>',
                mesh = 'microstructure:skeleton:mesh',
                condition = new_BC )

        self.boundary_condition_count = BC_no


    def removeBoundaryConditions(self):

        for bc_no in range(1, self.boundary_condition_count+1):
             OOF.Mesh.Boundary_Conditions.Delete(
                 mesh='microstructure:skeleton:mesh',
                 name='bc<' + str(bc_no) + '>')

        self.boundary_condition_count = 0


    @memorycheck.check("microstructure")
    def NonlinearHeatSource(self):

        nonlin_heat_source_tests = [ {"test_no":1,"soln_no":0},
                                     {"test_no":3,"soln_no":2} ]

        # define the heat equation related quantities needed for this test
        OOF.Property.Parametrize.Thermal.Conductivity.Isotropic(
            kappa=1.0)
        OOF.Material.Add_property(
            name='material',
            property='Thermal:Conductivity:Isotropic')
        OOF.Subproblem.Field.Define(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Subproblem.Field.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Mesh.Field.In_Plane(
            mesh='microstructure:skeleton:mesh',
            field=Temperature)
        OOF.Subproblem.Equation.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Heat_Eqn)

        # iterate through nonlinear heat source test by alternating
        # between various test examples and nonlinear solvers
        for test in nonlin_heat_source_tests:

            test_no = test["test_no"]
            soln_no = test["soln_no"]

            # add the nonlinear heat source property to the material
            OOF.Property.Parametrize.Thermal.HeatSource.TestNonlinearHeatSource(
                testno=test_no)
            OOF.Material.Add_property(
                name='material',
                property='Thermal:HeatSource:TestNonlinearHeatSource')
            OOF.Material.Assign(
                material='material', microstructure='microstructure', pixels=all)

            # set the boundary conditions for the given test no
            self.setBoundaryConditions( 'Dirichlet', Temperature, Heat_Eqn,
                                        self.heat_solns[soln_no]["DirichletBC"] )

            # compute the solution using Picard iterations
            test_solver = Picard(
                relative_tolerance=1e-08,
                absolute_tolerance=1.0e-13,
                maximum_iterations=20)

            self.nonlinearHeatEqnEngine( soln_no, test_solver )

            # compute the solution using Newton's method
            test_solver = Newton(
                relative_tolerance=1e-08,
                absolute_tolerance=1.0e-13,
                maximum_iterations=20)

            self.nonlinearHeatEqnEngine( soln_no, test_solver )

            # remove the boundary conditions for the given test no
            self.removeBoundaryConditions()

            # remove the current version of the nonlinear heat source property
            OOF.Material.Remove_property(
                name='material',
                property='Thermal:HeatSource:TestNonlinearHeatSource')

        # delete the other properties, fields, equations needed for this test
        OOF.Subproblem.Equation.Deactivate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Heat_Eqn)
        OOF.Subproblem.Field.Undefine(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Material.Remove_property(
            name='material',
            property='Thermal:Conductivity:Isotropic')


    @memorycheck.check("microstructure")
    def NonlinearHeatSourceNoDeriv(self):

        nonlin_heat_source_tests = [ {"test_no":1,"soln_no":0},
                                     {"test_no":3,"soln_no":2} ]

        # define the heat equation related quantities needed for this test
        OOF.Property.Parametrize.Thermal.Conductivity.Isotropic(
            kappa=1.0)
        OOF.Material.Add_property(
            name='material',
            property='Thermal:Conductivity:Isotropic')
        OOF.Subproblem.Field.Define(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Subproblem.Field.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Mesh.Field.In_Plane(
            mesh='microstructure:skeleton:mesh',
            field=Temperature)
        OOF.Subproblem.Equation.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Heat_Eqn)

        # iterate through nonlinear heat source test by alternating
        # between various test examples and nonlinear solvers
        for test in nonlin_heat_source_tests:

            test_no = test["test_no"]
            soln_no = test["soln_no"]

            # add the nonlinear heat source property to the material
            OOF.Property.Parametrize.Thermal.HeatSource.TestNonlinearHeatSourceNoDeriv(
                testno=test_no)
            OOF.Material.Add_property(
                name='material',
                property='Thermal:HeatSource:TestNonlinearHeatSourceNoDeriv')
            OOF.Material.Assign(
                material='material', microstructure='microstructure', pixels=all)

            # set the boundary conditions for the given test no
            self.setBoundaryConditions( 'Dirichlet', Temperature, Heat_Eqn,
                                        self.heat_solns[soln_no]["DirichletBC"] )

            # compute the solution using Newton's method
            test_solver = Newton(
                relative_tolerance=1e-13,
                absolute_tolerance=1e-13,
                maximum_iterations=20)

            self.nonlinearHeatEqnEngine( soln_no, test_solver )

            # remove the boundary conditions for the given test no
            self.removeBoundaryConditions()

            # remove the current version of the nonlinear heat source property
            OOF.Material.Remove_property(
                name='material',
                property='Thermal:HeatSource:TestNonlinearHeatSourceNoDeriv')

        # delete the other properties, fields, equations needed for this test
        OOF.Subproblem.Equation.Deactivate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Heat_Eqn)
        OOF.Subproblem.Field.Undefine(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Material.Remove_property(
            name='material',
            property='Thermal:Conductivity:Isotropic')


    @memorycheck.check("microstructure")
    def NonlinearHeatConductivity(self):

        nonlin_heat_conductivity_tests = [
            {"test_no":1,"heat_source_no":2,"soln_no":0},
            {"test_no":3,"heat_source_no":5,"soln_no":0}
            ]

        # define the heat equation related quantities needed for this test
        OOF.Subproblem.Field.Define(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Subproblem.Field.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Mesh.Field.In_Plane(
            mesh='microstructure:skeleton:mesh',
            field=Temperature)
        OOF.Subproblem.Equation.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Heat_Eqn)
        OOF.Material.Assign(
            material='material', microstructure='microstructure', pixels=all)
        OOF.Material.Add_property(
            name='material',
            property='Thermal:Conductivity:TestNonlinearHeatConductivity')
        OOF.Material.Add_property(
            name='material',
            property='Thermal:HeatSource:TestNonconstantHeatSource')

        # iterate through nonlinear heat conductivity test by alternating
        # between various test examples and nonlinear solvers
        for test in nonlin_heat_conductivity_tests:

            test_no   = test["test_no"]
            source_no = test["heat_source_no"]
            soln_no   = test["soln_no"]

            # add the nonlinear heat conductivity property to the material
            OOF.Property.Parametrize.Thermal.Conductivity.TestNonlinearHeatConductivity(
                testno=test_no)
            # add the nonconstant heat source property to the material
            OOF.Property.Parametrize.Thermal.HeatSource.TestNonconstantHeatSource(
                testno=source_no)

            # set the boundary conditions for the given test no
            self.setBoundaryConditions( 'Dirichlet', Temperature, Heat_Eqn,
                                        self.heat_solns[soln_no]["DirichletBC"] )

            # compute the solution using Newton's method
            test_solver = Newton(
                relative_tolerance=1e-08,
                absolute_tolerance=1e-13,
                maximum_iterations=20)

            self.nonlinearHeatEqnEngine( soln_no, test_solver )

            # remove the boundary conditions for the given test no
            self.removeBoundaryConditions()


        # delete the other properties, fields, equations needed for this test
        OOF.Subproblem.Equation.Deactivate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Heat_Eqn)
        OOF.Subproblem.Field.Undefine(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Material.Remove_property(
            name='material',
            property='Thermal:HeatSource:TestNonconstantHeatSource')
        OOF.Material.Remove_property(
            name='material',
            property='Thermal:Conductivity:TestNonlinearHeatConductivity')


    @memorycheck.check("microstructure")
    def NonlinearHeatConductivityNoDeriv(self):

        nonlin_heat_conductivity_tests = [
            {"test_no":1,"heat_source_no":2,"soln_no":0},
            {"test_no":3,"heat_source_no":5,"soln_no":0}
            ]

        # define the heat equation related quantities needed for this test
        OOF.Subproblem.Field.Define(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Subproblem.Field.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Mesh.Field.In_Plane(
            mesh='microstructure:skeleton:mesh',
            field=Temperature)
        OOF.Subproblem.Equation.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Heat_Eqn)
        OOF.Material.Assign(
            material='material', microstructure='microstructure', pixels=all)
        OOF.Material.Add_property(
            name='material',
            property='Thermal:Conductivity:TestNonlinearHeatConductivityNoDeriv')
        OOF.Material.Add_property(
            name='material',
            property='Thermal:HeatSource:TestNonconstantHeatSource')

        # iterate through nonlinear heat conductivity test by alternating
        # between various test examples and nonlinear solvers
        for test in nonlin_heat_conductivity_tests:

            test_no   = test["test_no"]
            source_no = test["heat_source_no"]
            soln_no   = test["soln_no"]

            # add the nonlinear heat conductivity property to the material
            OOF.Property.Parametrize.Thermal.Conductivity.TestNonlinearHeatConductivityNoDeriv(
                testno=test_no)
            # add the nonconstant heat source property to the material
            OOF.Property.Parametrize.Thermal.HeatSource.TestNonconstantHeatSource(
                testno=source_no)

            # set the boundary conditions for the given test no
            self.setBoundaryConditions( 'Dirichlet', Temperature, Heat_Eqn,
                                        self.heat_solns[soln_no]["DirichletBC"] )

            # compute the solution using Newton's method
            test_solver = Newton(
                relative_tolerance=1e-13,
                absolute_tolerance=1e-13,
                maximum_iterations=20)

            self.nonlinearHeatEqnEngine( soln_no, test_solver )

            # remove the boundary conditions for the given test no
            self.removeBoundaryConditions()


        # delete the other properties, fields, equations needed for this test
        OOF.Subproblem.Equation.Deactivate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Heat_Eqn)
        OOF.Subproblem.Field.Undefine(
            subproblem='microstructure:skeleton:mesh:default',
            field=Temperature)
        OOF.Material.Remove_property(
            name='material',
            property='Thermal:HeatSource:TestNonconstantHeatSource')
        OOF.Material.Remove_property(
            name='material',
            property='Thermal:Conductivity:TestNonlinearHeatConductivityNoDeriv')


    def nonlinearHeatEqnEngine(self,soln_no,test_solver):

        soln_func = self.heat_solns[soln_no]["Solution"]

        OOF.Subproblem.Set_Solver(
            subproblem='microstructure:skeleton:mesh:default',
            solver_mode=AdvancedSolverMode(
                time_stepper=StaticDriver(),
                nonlinear_solver=test_solver,
                symmetric_solver=ConjugateGradient(
                    preconditioner=ICPreconditioner(),
                    tolerance=1e-13,
                    max_iterations=1000),
                asymmetric_solver=BiConjugateGradient(
                    preconditioner=ICPreconditioner(),
                    tolerance=1e-13,
                    max_iterations=1000)))

        OOF.Mesh.Set_Field_Initializer(
            mesh='microstructure:skeleton:mesh',
            field=Temperature,
            initializer=ConstScalarFieldInit(value=0.0))
        OOF.Mesh.Apply_Field_Initializers(
            mesh='microstructure:skeleton:mesh')

        OOF.Mesh.Solve(
            mesh='microstructure:skeleton:mesh',
            endtime=self.time)

        from ooflib.engine import mesh
        from ooflib.SWIG.engine import field

        mesh_obj  = mesh.meshes["microstructure:skeleton:mesh"].getObject()
        field_ptr = field.getField( "Temperature" )

        L2_error = computeScalarErrorL2( soln_func, mesh_obj, field_ptr,
                                         self.numX, self.numY, time=self.time )
        print("L2 error: ", L2_error)

        self.assertTrue( L2_error < 4.e-2 )


    @memorycheck.check("microstructure")
    def NonlinearForceDensity(self):

        nonlin_force_density_tests = [ {"test_no":1,"soln_no":0},
                                       {"test_no":3,"soln_no":2} ]

        # define the force density related quantities needed for this test
        OOF.Property.Parametrize.Mechanical.Elasticity.Isotropic(
            cijkl=IsotropicRank4TensorLame(lmbda=-1,mu=1))
        OOF.Material.Add_property(
            name='material',
            property='Mechanical:Elasticity:Isotropic')
        OOF.Subproblem.Field.Define(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Subproblem.Field.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Mesh.Field.In_Plane(
            mesh='microstructure:skeleton:mesh',
            field=Displacement)
        OOF.Subproblem.Equation.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)

        # iterate through nonlinear force density test by alternating
        # between various test examples and nonlinear solvers
        for test in nonlin_force_density_tests:

            test_no = test["test_no"]
            soln_no = test["soln_no"]

            # add the nonlinear force density property to the material
            OOF.Property.Parametrize.Mechanical.ForceDensity.TestNonlinearForceDensity(
                testno=test_no)
            OOF.Material.Add_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonlinearForceDensity')
            OOF.Material.Assign(
                material='material', microstructure='microstructure', pixels=all)

            # set the boundary conditions for the given test no
            self.setBoundaryConditions( 'Dirichlet', Displacement, Force_Balance,
                                        self.elasticity_solns[soln_no]["DirichletBC"] )

            # compute the solution using Picard iterations
            test_solver = Picard(
                relative_tolerance=1e-08,
                absolute_tolerance=1e-13,
                maximum_iterations=20)

            self.nonlinearElasticityEqnEngine( soln_no, test_solver )

            # compute the solution using Newton's method
            test_solver = Newton(
                relative_tolerance=1e-08,
                absolute_tolerance=1e-13,
                maximum_iterations=20)

            self.nonlinearElasticityEqnEngine( soln_no, test_solver )

            # remove the boundary conditions for the given test no
            self.removeBoundaryConditions()

            # remove the current version of the nonlinear force density property
            OOF.Material.Remove_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonlinearForceDensity')

        # delete the other properties, fields, equations needed for this test
        OOF.Subproblem.Equation.Deactivate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)
        OOF.Subproblem.Field.Undefine(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Material.Remove_property(
            name='material',
            property='Mechanical:Elasticity:Isotropic')


    @memorycheck.check("microstructure")
    def NonlinearForceDensityNoDeriv(self):

        nonlin_force_density_tests = [ {"test_no":1,"soln_no":0},
                                       {"test_no":3,"soln_no":2} ]

        # define the force density related quantities needed for this test
        OOF.Property.Parametrize.Mechanical.Elasticity.Isotropic(
            cijkl=IsotropicRank4TensorLame(lmbda=-1,mu=1))
        OOF.Material.Add_property(
            name='material',
            property='Mechanical:Elasticity:Isotropic')
        OOF.Subproblem.Field.Define(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Subproblem.Field.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Mesh.Field.In_Plane(
            mesh='microstructure:skeleton:mesh',
            field=Displacement)
        OOF.Subproblem.Equation.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)

        # iterate through nonlinear force density test by alternating
        # between various test examples and nonlinear solvers
        for test in nonlin_force_density_tests:

            test_no = test["test_no"]
            soln_no = test["soln_no"]

            # add the nonlinear force density property to the material
            OOF.Property.Parametrize.Mechanical.ForceDensity.TestNonlinearForceDensityNoDeriv(
                testno=test_no)
            OOF.Material.Add_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonlinearForceDensityNoDeriv')
            OOF.Material.Assign(
                material='material', microstructure='microstructure', pixels=all)

            # set the boundary conditions for the given test no
            self.setBoundaryConditions( 'Dirichlet', Displacement, Force_Balance,
                                        self.elasticity_solns[soln_no]["DirichletBC"] )

            # compute the solution using Newton's method
            test_solver = Newton(
                relative_tolerance=1e-13,
                absolute_tolerance=1e-13,
                maximum_iterations=20)

            self.nonlinearElasticityEqnEngine( soln_no, test_solver )

            # remove the boundary conditions for the given test no
            self.removeBoundaryConditions()

            # remove the current version of the nonlinear force density property
            OOF.Material.Remove_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonlinearForceDensityNoDeriv')

        # delete the other properties, fields, equations needed for this test
        OOF.Subproblem.Equation.Deactivate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)
        OOF.Subproblem.Field.Undefine(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Material.Remove_property(
            name='material',
            property='Mechanical:Elasticity:Isotropic')


    @memorycheck.check("microstructure")
    def NonlinearElasticity(self):

        nonlin_elasticity_tests = [ {"test_no":1,"force_no":3,"soln_no":3} ]

        # define the elasticity eqn related quantities needed for this test
        OOF.Subproblem.Field.Define(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Subproblem.Field.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Mesh.Field.In_Plane(
            mesh='microstructure:skeleton:mesh',
            field=Displacement)
        OOF.Subproblem.Equation.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)

        # iterate through nonlinear elasticity test by alternating
        # between various test examples and nonlinear solvers
        for test in nonlin_elasticity_tests:

            test_no  = test["test_no"]
            force_no = test["force_no"]
            soln_no  = test["soln_no"]

            # add the nonlinear elasticity property to the material
            OOF.Property.Parametrize.Mechanical.Elasticity.TestGeneralNonlinearElasticity(
                testno=test_no)
            OOF.Material.Add_property(
                name='material',
                property='Mechanical:Elasticity:TestGeneralNonlinearElasticity')
            # add the nonconstant force density property to the material
            OOF.Property.Parametrize.Mechanical.ForceDensity.TestNonconstantForceDensity(
                testno=force_no)
            OOF.Material.Add_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonconstantForceDensity')
            OOF.Material.Assign(
                material='material', microstructure='microstructure', pixels=all)

            # set the boundary conditions for the given test no
            self.setBoundaryConditions( 'Dirichlet', Displacement, Force_Balance,
                                        self.elasticity_solns[soln_no]["DirichletBC"] )

            # compute the solution using Newton's method
            test_solver = Newton(
                relative_tolerance=1e-08,
                absolute_tolerance=1e-13,
                maximum_iterations=20)

            self.nonlinearElasticityEqnEngine( soln_no, test_solver )

            # remove the boundary conditions for the given test no
            self.removeBoundaryConditions()

            # remove the nonlin elasticity and nonconst force density properties
            OOF.Material.Remove_property(
                name='material',
                property='Mechanical:Elasticity:TestGeneralNonlinearElasticity')
            OOF.Material.Remove_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonconstantForceDensity')

        # delete the other properties, fields, equations needed for this test
        OOF.Subproblem.Equation.Deactivate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)
        OOF.Subproblem.Field.Undefine(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)


    @memorycheck.check("microstructure")
    def NonlinearElasticityNoDeriv(self):

        nonlin_elasticity_tests = [ {"test_no":1,"force_no":3,"soln_no":3} ]

        # define the elasticity eqn related quantities needed for this test
        OOF.Subproblem.Field.Define(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Subproblem.Field.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Mesh.Field.In_Plane(
            mesh='microstructure:skeleton:mesh',
            field=Displacement)
        OOF.Subproblem.Equation.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)

        # iterate through nonlinear elasticity test by alternating
        # between various test examples and nonlinear solvers
        for test in nonlin_elasticity_tests:

            test_no  = test["test_no"]
            force_no = test["force_no"]
            soln_no  = test["soln_no"]

            # add the nonlinear elasticity property to the material
            OOF.Property.Parametrize.Mechanical.Elasticity.TestGeneralNonlinearElasticityNoDeriv(
                testno=test_no)
            OOF.Material.Add_property(
                name='material',
                property='Mechanical:Elasticity:TestGeneralNonlinearElasticityNoDeriv')
            # add the nonconstant force density property to the material
            OOF.Property.Parametrize.Mechanical.ForceDensity.TestNonconstantForceDensity(
                testno=force_no)
            OOF.Material.Add_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonconstantForceDensity')
            OOF.Material.Assign(
                material='material', microstructure='microstructure', pixels=all)

            # set the boundary conditions for the given test no
            self.setBoundaryConditions( 'Dirichlet', Displacement, Force_Balance,
                                        self.elasticity_solns[soln_no]["DirichletBC"] )

            # compute the solution using Newton's method
            test_solver = Newton(
                relative_tolerance=1e-13,
                absolute_tolerance=1e-13,
                maximum_iterations=20)

            self.nonlinearElasticityEqnEngine( soln_no, test_solver )

            # remove the boundary conditions for the given test no
            self.removeBoundaryConditions()

            # remove the nonlin elasticity and nonconst force density properties
            OOF.Material.Remove_property(
                name='material',
                property='Mechanical:Elasticity:TestGeneralNonlinearElasticityNoDeriv')
            OOF.Material.Remove_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonconstantForceDensity')

        # delete the other properties, fields, equations needed for this test
        OOF.Subproblem.Equation.Deactivate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)
        OOF.Subproblem.Field.Undefine(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)


    @memorycheck.check("microstructure")
    def LargeStrain(self):

        large_strain_tests = [ {"force_no":4,"soln_no":5} ]

        # define the elasticity eqn related quantities needed for this test
        OOF.Subproblem.Field.Define(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Subproblem.Field.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)
        OOF.Mesh.Field.In_Plane(
            mesh='microstructure:skeleton:mesh',
            field=Displacement)
        OOF.Subproblem.Equation.Activate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)

        # iterate through large strain tests by alternating
        # between various test examples and nonlinear solvers
        for test in large_strain_tests:

            force_no = test["force_no"]
            soln_no  = test["soln_no"]

            # add the large strain property to the material
            OOF.Property.Parametrize.Mechanical.Elasticity.LargeStrain.Isotropic(
                cijkl=IsotropicRank4TensorLame(lmbda=-1,mu=1))
            OOF.Material.Add_property(
                name='material',
                property='Mechanical:Elasticity:LargeStrain:Isotropic')
            # add the nonconstant force density property to the material
            OOF.Property.Parametrize.Mechanical.ForceDensity.TestNonconstantForceDensity(
                testno=force_no)
            OOF.Material.Add_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonconstantForceDensity')
            OOF.Material.Assign(
                material='material', microstructure='microstructure', pixels=all)

            # set the boundary conditions for the given test no
            self.setBoundaryConditions(
                'Dirichlet', Displacement, Force_Balance,
                self.elasticity_solns[soln_no]["DirichletBC"] )

            # compute the solution using Newton's method
            test_solver = Newton(
                relative_tolerance=1e-08,
                absolute_tolerance=1e-13,
                maximum_iterations=20)

            self.nonlinearElasticityEqnEngine( soln_no, test_solver )

            # remove the boundary conditions for the given test no
            self.removeBoundaryConditions()

            # remove the nonlin elasticity and nonconst force density properties
            OOF.Material.Remove_property(
                name='material',
                property='Mechanical:Elasticity:LargeStrain:Isotropic')
            OOF.Material.Remove_property(
                name='material',
                property='Mechanical:ForceDensity:TestNonconstantForceDensity')

        # delete the other properties, fields, equations needed for this test
        OOF.Subproblem.Equation.Deactivate(
            subproblem='microstructure:skeleton:mesh:default',
            equation=Force_Balance)
        OOF.Subproblem.Field.Undefine(
            subproblem='microstructure:skeleton:mesh:default',
            field=Displacement)


    def nonlinearElasticityEqnEngine(self,soln_no,test_solver):

        soln_func   = self.elasticity_solns[soln_no]["Solution"]

        OOF.Subproblem.Set_Solver(
            subproblem='microstructure:skeleton:mesh:default',
            solver_mode=AdvancedSolverMode(
                time_stepper=StaticDriver(),
                nonlinear_solver=test_solver,
                symmetric_solver=ConjugateGradient(
                    preconditioner=ICPreconditioner(),
                    tolerance=1e-13,
                    max_iterations=1000),
                asymmetric_solver=BiConjugateGradient(
                    preconditioner=ICPreconditioner(),
                    tolerance=1e-13,
                    max_iterations=1000)))

        OOF.Mesh.Set_Field_Initializer(
            mesh='microstructure:skeleton:mesh',
            field=Displacement,
            initializer=ConstTwoVectorFieldInit(cx=0.0,cy=0.0))
        OOF.Mesh.Apply_Field_Initializers(
            mesh='microstructure:skeleton:mesh')

        OOF.Mesh.Solve(
            mesh='microstructure:skeleton:mesh',
            endtime=self.time)

        from ooflib.engine import mesh
        from ooflib.SWIG.engine import field

        mesh_obj  = mesh.meshes["microstructure:skeleton:mesh"].getObject()
        field_ptr = field.getField( "Displacement" )

        L2_error = computeVector2DErrorL2( soln_func, mesh_obj, field_ptr,
                                           self.numX, self.numY, time=self.time )
        print("L2 error: ", L2_error)

        self.assertTrue( L2_error < 1.e-2 )


#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

test_set = [
    NonlinearPropertyTest("NonlinearHeatSource"),
    NonlinearPropertyTest("NonlinearHeatSourceNoDeriv"),
    NonlinearPropertyTest("NonlinearHeatConductivity"),
    NonlinearPropertyTest("NonlinearHeatConductivityNoDeriv"),
    NonlinearPropertyTest("NonlinearForceDensity"),
    NonlinearPropertyTest("NonlinearForceDensityNoDeriv"),
    NonlinearPropertyTest("NonlinearElasticity"),
    NonlinearPropertyTest("NonlinearElasticityNoDeriv"),
    NonlinearPropertyTest("LargeStrain"),
    RotatingSquare("Static"),
    RotatingSquare("QuasiStatic"),
    ]
