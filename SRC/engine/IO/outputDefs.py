# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# This file generates concrete Outputs from the Outputs defined in
# outputClones.py.  The concrete Outputs are stored in a tree
# structure which can easily be converted to a menu.

from ooflib.SWIG.common import config
from ooflib.SWIG.common import switchboard
from ooflib.SWIG.engine import field
from ooflib.SWIG.engine import flux
from ooflib.SWIG.engine import outputval
from ooflib.SWIG.engine import planarity
from ooflib.common import debug
from ooflib.common import enum
from ooflib.common import primitives
from ooflib.common.IO import parameter
from ooflib.common.IO import xmlmenudump
from ooflib.engine import problem
from ooflib.engine.IO import output
from ooflib.engine.IO import outputClones
from ooflib.engine.IO import propertyoutputreg

######################################

# Displacement

# Although Displacement is a Field, it's convenient to treat it as a
# Point so that it can be added to a position.

Displacement = field.getField("Displacement")
disp0, disp1 = Displacement.components(planarity.IN_PLANE)

displacementFieldOutput = outputClones.FieldOutput.clone(
    name="displacement field",
    params=dict(field=Displacement))

# displacementOutput produces the Displacement Field as a set of Points.

def _disp2point(mesh, elements, coords, field):
    # Convert displacement field OutputVals to Points
    return [primitives.Point(f[disp0], f[disp1]) for f in field]

displacementOutput = output.Output(
    name="dispPoint",
    otype=primitives.Point,
    callback=_disp2point,
    inputs=[outputval.OutputValParameter('field')]
    )
displacementOutput.connect('field', displacementFieldOutput)

# Enhanced position is scalefactor*Displacement + original position

enhancedPosition = outputClones.PointSumOutput.clone(
    name='enhanced position',
    params=dict(b=1),
    tip='Exaggerated displacement field.',
    discussion=xmlmenudump.loadFile(
    'DISCUSSIONS/engine/output/enhancedPosOutput.xml'))
enhancedPosition.connect('point1', displacementOutput)
enhancedPosition.connect('point2', outputClones.posOutput)
enhancedPosition.aliasParam('a', 'factor', default=1.0,
                            tip='Displacement multiplier.')

# Actual position is enhanced position with a scalefactor of 1
actualPosition = enhancedPosition.clone(
    name='actual position',
    params=dict(factor=1.0),
    tip='Displaced position.',
    discussion=xmlmenudump.loadFile('DISCUSSIONS/engine/output/actualPosOutput.xml')
    )

originalPosition = outputClones.posOutput.clone(
    discussion=xmlmenudump.loadFile('DISCUSSIONS/engine/output/originalPosOutput.xml'))

output.definePositionOutput('original', originalPosition)
output.definePositionOutput('actual', actualPosition)
output.definePositionOutput('enhanced', enhancedPosition)

######################

output.defineAggregateOutput('Field:Value', outputClones.FieldOutput,
                             ordering=1.0)
output.defineAggregateOutput('Field:Derivative', outputClones.FieldDerivOutput,
                             ordering=1.1)
output.defineAggregateOutput('Field:Invariant',
                             outputClones.FieldInvariantOutput,
                             ordering=1.2)
output.defineAggregateOutput('Flux:Value', outputClones.FluxOutput,
                             ordering=2.0)
output.defineAggregateOutput('Flux:Invariant', outputClones.FluxInvariantOutput,
                             ordering=2.1)

output.defineScalarOutput('Field:Component', outputClones.FieldCompOutput, 
                          ordering=1.0)
output.defineScalarOutput('Field:Invariant', outputClones.FieldInvariantOutput,
                          ordering=1.1)
output.defineScalarOutput('Field:Derivative:Component',
                          outputClones.FieldDerivCompOutput, ordering=1.2)
output.defineScalarOutput('Field:Derivative:Invariant',
                          outputClones.FieldDerivInvariantOutput, ordering=1.3)
output.defineScalarOutput('Flux:Component', outputClones.FluxCompOutput, 
                          ordering=2.0)
output.defineScalarOutput('Flux:Invariant', outputClones.FluxInvariantOutput, 
                          ordering=2.1)

from ooflib.common import strfunction

xyfunc = outputClones.ScalarFunctionOutput.clone(
    tip='Compute an arbitrary scalar function of x and y.',
    discussion=xmlmenudump.loadFile(
        'DISCUSSIONS/engine/output/funcOutput.xml'))
output.defineScalarOutput('XYFunction', xyfunc, ordering=100)

xyvecfunc = outputClones.VectorFunctionOutput.clone(
    tip='Compute an arbitrary vector function of x and y.',
    discussion=xmlmenudump.loadFile(
        'DISCUSSIONS/engine/output/vecfuncOutput.xml'))
output.defineAggregateOutput('Vector XYFunction', xyvecfunc,
                             ordering=100)

###########

negElectricFieldOutput = outputClones.FieldDerivOutput.clone(
    params=dict(field=field.getField("Voltage"))
    )
electricFieldOutput = outputClones.NegateOutput.clone()
electricFieldOutput.connect("scalee", negElectricFieldOutput)
#output.defineAggregateOutput('E Field', electricFieldOutput)
eFieldCompOutput = outputClones.ComponentOutput.clone()
eFieldCompOutput.connect('field', electricFieldOutput)
#output.defineScalarOutput('E Field:component', eFieldCompOutput)

## TODO: Uncomment the above two defineOutput lines, but only after
## fixing the problems with the MeshParamWidgets.  The problem is that
## the widget for the component of the E field looks for a
## FieldWidget, and doesn't find one because the Field isn't a
## settable parameter.

######################

# PropertyOutputs, which are computed by the Property objects in a
# Material, only have to be identified by name, type, and (possibly)
# initializer here.  The computations are all done by the
# PropertyOutput<TYPE> classes and the various Property::output virtual
# functions.

class EnergyType(enum.EnumClass(
    ("Total", "All contributions to the energy"),
    ("Elastic", "Elastic energy-- one half stress times elastic strain"),
    ("Electric", "Electric energy-- one half total polarization times electric field")
    )):
    tip="Varieties of Energy."
    discussion="""<para>
    <classname>EnergyType</classname> indicates which energy is to be
    computed by the <xref linkend='Output-Scalar-Energy'/> Output.
    </para>"""

def _Energy_shortrepr(self):
    etype = self.findParam("etype").value
    return etype.string() + " Energy"

# ScalarPropertyOutputRegistration places the output in both the
# ScalarOutput and AggregateOutput trees.

propertyoutputreg.ScalarPropertyOutputRegistration(
    "Energy",
    parameters=[enum.EnumParameter("etype", EnergyType, default="Total",
                                  tip='The type of energy to compute.')],
    ordering=3,
    srepr = _Energy_shortrepr,
    tip='Compute an energy density.',
    discussion="""
      <para>Compute the energy density of the &fields; on the &mesh;.
      Different values of <varname>etype</varname> include different
      contributions to the energy.</para>"""
    )


#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

# Parameters used for PropertyParameterOutputs

class ReferenceFrame(enum.EnumClass(
        ("Lab", "The laboratory reference frame"),
        ("Crystal", "The crystal reference frame")
        )):
    tip="Evaluate quantities in the lab or crystal reference frame."
    discussion="""<para>The values of anisotropic quantities can be
    printed in either the laboratory or crystal reference frames.
    When printing an anisotropic modulus, for example, in the crystal
    frame the values will be just what you entered when you defined
    the &property;.  In the lab frame the values will be modified by
    the &material;'s <link
    linkend="RegisteredClass-Orientation"><classname>orientation</classname></link>.</para>"""

class VoigtPairListParameter(parameter.ListOfStringsParameter):
    def checker(self, x):
        if not isinstance(x, (list, tuple)):
            parameter.raiseTypeError(type(x), "list of Voigt index pairs")
        for s in x:
            if not (isinstance(s, str) and len(s)==2 and
                    s[0] in "123456" and s[1] in "123456"):
                parameter.raiseTypeError("list of %s" % type(s).__name__,
                                         "list of Voigt index pairs [1-6][1-6]")
    def valueDesc(self):
        return "A list of character strings of the form 'XY'" \
            " where X and Y are digits from 1 to 6."

class SymmIndexPairListParameter(parameter.ListOfStringsParameter):
    def checker(self, x):
        if not isinstance(x, (list, tuple)):
            parameter.raiseTypeError(type(x), "list of index pairs")
        for s in x:
            if not (isinstance(s, str) and len(s)==2 and
                    s[0] in "123" and s[1] in "123"):
                parameter.raiseTypeError("list of %s" % type(s).__name__,
                                         "list of index pairs [1-3][1-3]")
    def valueDesc(self):
        return "A list of character strings of the form 'XY'" \
            "where X and Y are digits from 1 to 3."

class Rank3TensorIndexParameter(parameter.ListOfStringsParameter):
    # Pair of integers, ([1-3], [1-6])
    # This class is poorly named because the name doesn't indicate the
    # symmetry of the tensor (ie, 2nd index is Voigt).
    def checker(self, x):
        if not isinstance(x, (list, tuple)):
            parameter.raiseTypeError(type(x), "list of index pairs")
        for s in x:
            if not (isinstance(s, str) and len(s)==2 and
                    s[0] in "123" and s[1] in "123456"):
                parameter.raiseTypeError("list of %s" % type(s).__name__,
                                         "list of index pairs [1-3][1-6]")
    def valueDesc(self):
        return "A list of character strings of the form 'XY'" \
            "where X is a digit from 1 to 3 and Y is a Voigt index from 1 to 6."

# PropertyParameterOutputs, which output the parameters defining
# material properties.  The ordering of the registrations should echo
# the ordering of the Properties in the Materials page.

## TODO: Can these registrations be created automatically by the
## Property registrations?  It wouldn't be easy, because there's not a
## one-to-one relationship between the Property registrations and the
## PropertyOutput registrations.

# Mechanical

propertyoutputreg.ModulusPropertyOutputRegistration(
    name="Material Constants:Mechanical:Elastic Modulus C",
    symbol="C",
    parameters=[
        VoigtPairListParameter(
            "components",
            tip="Evaluate the selected components of the modulus."),
        enum.EnumParameter(
            "frame", ReferenceFrame, default="Crystal",
            tip="Report the modulus in this reference frame.")
    ],
    tip="Print components of the elastic modulus.",
    discussion="""<para>Print the given components of the <link
    linkend='PropertyType-Elasticity'>elastic modulus</link>.</para>""",
    ordering=10)

propertyoutputreg.ModulusPropertyOutputRegistration(
    name="Material Constants:Mechanical:Stress-free Strain epsilon0",
    symbol="epsilon0",
    parameters=[
        SymmIndexPairListParameter(
            "components",
            tip="Evaluate the selected components of the stress-free strain."),
        enum.EnumParameter(
            "frame", ReferenceFrame, default="Crystal",
            tip="Report the stress-free strain in this reference frame.")
        ],
    tip="Print components of the stress-free strain.",
    discussion="""<para>Print the given components of the <link
    linkend='PropertyType-StressFreeStrain'>stress-free
    strain</link>.</para>""",
    ordering=11)

propertyoutputreg.TwoVectorParamPropertyOutputRegistration(
    name="Material Constants:Mechanical:Force Density F",
    symbol="F",
    tip="Print the force density",
    discussion="""<para>Print the x and y components of the <link
    linkend='PropertyType-ForceDensity'>force density</link>.</para>""",
    ordering=12)

propertyoutputreg.ScalarParamOutputRegistration(
    name="Material Constants:Mechanical:Mass Density",
    srepr=lambda s: "Mass Density",
    tip="Print the mass density",
    discussion="""<para>Print the value of the <link
    linkend='PropertyType-MassDensity'>mass density</link>.</para>""",
    ordering=13)

propertyoutputreg.ScalarParamOutputRegistration(
    name="Material Constants:Mechanical:Damping",
    srepr=lambda s: "Damping",
    tip="Print the damping coefficient.",
    discussion="""<para>Print the value of the <link
    linkend='PropertyType-Damping'>damping coefficient</link>.</para>""",
    ordering=15)

propertyoutputreg.ModulusPropertyOutputRegistration(
    name="Material Constants:Mechanical:Viscosity",
    symbol="g",
    parameters=[
        VoigtPairListParameter(
            "components",
            tip="Evaluate the selected components of the modulus"),
        enum.EnumParameter(
            "frame", ReferenceFrame, default="Crystal",
            tip="Report the viscosity in this reference frame.")
        ],
    tip="Print the viscosity.",
    discussion="""<para>Print the given components of the <link
    linkend='PropertyType-ViscoElasticity'>viscoelasticity
    tensor</link>.</para>""",
    ordering=14)

# Thermal

propertyoutputreg.ModulusPropertyOutputRegistration(
    name="Material Constants:Thermal:Conductivity K",
    symbol="K",
    parameters=[
        SymmIndexPairListParameter(
            "components",
            tip="Evaluate the selected components of the conductivity."),
        enum.EnumParameter(
            "frame", ReferenceFrame, default="Crystal",
            tip="Report the conductivity in this reference frame.")
        ],
    tip="Print the thermal conductivity.",
    discussion="""<para>Print the given components of the <link
    linkend='PropertyType-ThermalConductivity'>thermal conductivity
    tensor</link>.</para>""",
    ordering=20)

propertyoutputreg.ScalarParamOutputRegistration(
    name="Material Constants:Thermal:Heat Capacity",
    srepr=lambda s: "Heat Capacity",
    tip="Print the heat capacity.",
    discussion="""<para>Print the value of the <link
    linkend='PropertyType-HeatCapacity'>heat capacity</link>.</para>""",
    ordering=21)

propertyoutputreg.ScalarParamOutputRegistration(
    name="Material Constants:Thermal:Heat Source",
    srepr=lambda s: "Heat Source",
    tip="Print the heat source magnitude",
    discussion="""<para>Print the magnitude of the <link
    linkend='PropertyType-HeatSource'>heat source</link>.</para>""",
    ordering=22)

propertyoutputreg.ModulusPropertyOutputRegistration(
    name="Material Constants:Electric:Dielectric Permittivity epsilon",
    symbol="epsilon",
    parameters=[
        SymmIndexPairListParameter(
            "components",
            tip="Evaluate the selected components of the permittivity."),
        enum.EnumParameter(
            "frame", ReferenceFrame, default="Crystal",
            tip="Report the permittivity in this reference frame.")
        ],
    tip="Print the dielectric permittivity",
    discussion="""<para>Print the given components of the <link
    linkend='PropertyType-DielectricPermittivity'>dielectric
    permittivity</link>.</para>""",
    ordering=30)

propertyoutputreg.ScalarParamOutputRegistration(
    name="Material Constants:Electric:Space Charge",
    srepr=lambda s: "Space Charge",
    tip="Print the space charge.",
    discussion="""<para>Print the magnitude of the <link
    linkend='PropertyType-ChargeDensity'>space charge
    density</link>.</para>""",
    ordering=31)

# Couplings

propertyoutputreg.ModulusPropertyOutputRegistration(
    name="Material Constants:Couplings:Thermal Expansion alpha",
    symbol="alpha",
    parameters=[
        SymmIndexPairListParameter(
            "components",
            tip="Evaluate the selected components of the thermal expansion coefficient"),
        enum.EnumParameter(
            "frame", ReferenceFrame, default="Crystal",
            tip="Report the thermal expansion coefficient in this reference frame.")],
    tip="Print the thermal expansion coefficient",
    discussion="""<para>Print the given components of the <link
    linkend='PropertyType-ThermalExpansion'>thermal expansion
    tensor</link>.</para>""",
    ordering=50)

propertyoutputreg.ScalarParamOutputRegistration(
    name="Material Constants:Couplings:Thermal Expansion T0",
    srepr=lambda s: "T0",
    tip="Print the thermal expansion reference temperature.",
    discussion="""<para>Print the temperature at which the <link
    linkend='PropertyType-ThermalExpansion'>thermal expansion</link>
    vanishes.</para>""",
    ordering=50.5)

propertyoutputreg.ModulusPropertyOutputRegistration(
    name="Material Constants:Couplings:Piezoelectric Coefficient D",
    symbol="D",
    parameters=[
        Rank3TensorIndexParameter(
            "components",
            tip="Evaluate the selected components of the piezoelectric coefficient."),
        enum.EnumParameter(
            "frame", ReferenceFrame, default="Crystal",
            tip="Report the stress-free strain in this reference frame.")
        ],
    tip="Print the piezoelectric coefficient.",
    discussion="""<para>Print the given components of the <link
    linkend='PropertyType-PiezoElectricity'>piezoelectric
    tensor</link>.</para>""",
    ordering=51)

# TODO: PyroElectricity

# Orientation

propertyoutputreg.OrientationPropertyOutputRegistration(
    "Material Constants:Orientation",
    tip="Compute the orientation at each point.",
    discussion="""<para>Print the value of the <link
    linkend='PropertyType-Orientation'>orientation</link>.</para>""",
    ordering=1000,
)

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

# Time output.  This is intended to generate a "time" column when
# concatenating Outputs.

def _timeOutput(mesh, elements, coords):
    return [outputval.ScalarOutputVal(mesh.getCurrentTime())]

TimeOutput = output.Output(
    name = "time",
    otype = outputval.ScalarOutputVal,
    callback = _timeOutput,
    srepr = lambda s: "Time",
    instancefn = outputClones.scalar_instancefn,
    column_names = lambda s: ["Time"]
    )

output.defineAggregateOutput('Time', TimeOutput, ordering=10000)
output.defineScalarOutput('Time', TimeOutput, ordering=10000)
