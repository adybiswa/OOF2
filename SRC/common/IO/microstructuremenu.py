# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.common import config
from ooflib.SWIG.common import activearea
from ooflib.SWIG.common import switchboard
from ooflib.SWIG.common import ooferror
from ooflib.SWIG.common import pixelsetboundary
from ooflib.common import enum
from ooflib.common import debug
from ooflib.common import pixelselection
from ooflib.common import parallel_enable
from ooflib.common import primitives
from ooflib.common.IO import automatic
from ooflib.common.IO import datafile
from ooflib.common.IO import filenameparam
from ooflib.common.IO import mainmenu
from ooflib.common.IO import microstructureIO
from ooflib.common.IO import oofmenu
from ooflib.common.IO import parameter
from ooflib.common.IO import whoville
from ooflib.common.IO import xmlmenudump
from ooflib.image import imagecontext
import ooflib.common.microstructure

if parallel_enable.enabled():
    from ooflib.common.IO import microstructureIPC

micromenu = mainmenu.OOF.addItem(oofmenu.OOFMenuItem(
    'Microstructure',
    cli_only=True,
    help="Create and manipulate &micro; objects.",
    discussion="""<para>
    Commands for creating and manipulating Microstructures.
    </para>""",
    xrefs=["Section-Tasks-Microstructure"]
))

#######################

sizeparams = parameter.ParameterGroup(
    parameter.FloatParameter('width', 1., tip='Width in physical units.'),
    parameter.FloatParameter('height', 1.,tip='Height in physical units.'),
    parameter.IntParameter('width_in_pixels', 10, tip='Width in pixels.'),
    parameter.IntParameter('height_in_pixels', 10, tip='Height in pixels.'))


def newMicrostructure(menuitem, name,
                      width, height,
                      width_in_pixels, height_in_pixels):
    if width<=0 or height<=0 or width_in_pixels<=0 or height_in_pixels<=0:
        raise ooferror.PyErrUserError("Negative sizes are not allowed.")

    if parallel_enable.enabled():
        # For the rear-end guys
        microstructureIPC.msmenu.New_Parallel(name=name,
                                             width=width, height=height,
                                             width_in_pixels=width_in_pixels,
                                             height_in_pixels=height_in_pixels)

    # Serial mode & #0 in parallel mode
    ms = ooflib.common.microstructure.Microstructure(
        name, primitives.iPoint(width_in_pixels,
                                height_in_pixels),
        primitives.Point(width, height)
        )

def msNameResolver(param, name):
    if param.automatic():
        nm = 'microstructure'
    else:
        nm = name
    return ooflib.common.microstructure.microStructures.uniqueName(nm)
    
micromenu.addItem(
    oofmenu.OOFMenuItem(
    'New',
    callback=newMicrostructure,
    params=parameter.ParameterGroup(
            whoville.AutoWhoNameParameter(
                'name', resolver=msNameResolver,
                value=automatic.automatic,
                tip="Name of the new Microstructure.")) 
    + sizeparams,
    help="Create a new Microstructure.",
    discussion=xmlmenudump.loadFile("DISCUSSIONS/common/menu/newmicro.xml")
    ))

############################

def renameMicrostructure(menuitem, microstructure, name):
    if parallel_enable.enabled():
        microstructureIPC.msmenu.Rename(microstructure=microstructure,name=name)
        return

    ms = ooflib.common.microstructure.microStructures[microstructure]
    ms.reserve()
    aa = ms.getObject().activearea
    aa.begin_writing()
    try:
        aa.rename(name)
    finally:
        aa.end_writing()
    ps = ms.getObject().pixelselection
    ps.begin_writing()
    try:
        ps.rename(name)
    finally:
        ps.end_writing()
    ms.begin_writing()
    try:
        ms.rename(name, exclude=ms.getObject().name())
    finally:
        ms.end_writing()
        ms.cancel_reservation()

micromenu.addItem(
    oofmenu.OOFMenuItem('Rename',
                        callback=renameMicrostructure,
                        params=[
    parameter.StringParameter('microstructure', '',
                              tip='Old name for the microstructure.'),
    whoville.WhoNameParameter('name', value='', 
                              tip='New name for the microstructure.')
    ],
                        help="Rename a Microstructure.",
                        discussion="""<para>
                        Change the name of an existing &micro;.
                        </para>"""
                        ))

############################

def copyMicrostructure(menuitem, microstructure, name):
    if parallel_enable.enabled():
        microstructureIPC.msmenu.Copy(microstructure=microstructure,name=name)
        return

    ms = ooflib.common.microstructure.microStructures[microstructure]
    ms.begin_reading()
    grouplist = []
    try:
        sourceMS = ms.getObject()
        newMS = sourceMS.nominalCopy(name)
        newMScontext = ooflib.common.microstructure.microStructures[name]
        # First, copy images and load them to copied MS.
        for imagectxt in sourceMS.getImageContexts():
            sourceimage = imagectxt.getObject()
            immidge = sourceimage.clone(sourceimage.name())
            imagecontext.imageContexts.add([newMS.name(),immidge.name()],
                                           immidge,
                                           parent=newMScontext)
        # Copy pixel groups
        for grpname in sourceMS.groupNames():
            sourcegrp = sourceMS.findGroup(grpname)
            # Ignore "newness", all groups will be new.
            (newgrp, newness) = newMS.getGroup(grpname)
            newgrp.add(sourcegrp.members())
            newgrp.set_meshable(sourcegrp.is_meshable())
            grouplist.append(newgrp)
    finally:
        ms.end_reading()
        
    for g in grouplist:
        switchboard.notify("new pixel group", g)

micromenu.addItem(oofmenu.OOFMenuItem(
    'Copy',
    callback=copyMicrostructure,
    params=[parameter.StringParameter('microstructure', '',
                                      tip='The source microstructure.'),    
            parameter.AutomaticNameParameter('name', 
                                             resolver=msNameResolver,
                                             value=automatic.automatic,
                                         tip="Name of the new Microstructure.")
            ],
    help="Copy a Microstructure.",
    discussion=
    """<para>A copied &micro; has the same physical size and pixel
    size as the original &micro;, and includes
    <emphasis>copies</emphasis> of all &images; and <link
    linkend='Section-Concepts-Microstructure-Pixel_Group'>pixel
    groups</link> contained in the original.  It does
    <emphasis>not</emphasis> include any &skels; or &meshes; from the
    original.</para>"""
    ))

#############################

def deleteMicrostructure(menuitem, microstructure):
    if parallel_enable.enabled():
        microstructureIPC.msmenu.Delete(microstructure=microstructure)
    else:
        mscontext = ooflib.common.microstructure.microStructures[microstructure]
        mscontext.lockAndDelete()

micromenu.addItem(oofmenu.OOFMenuItem(
    'Delete',
    callback=deleteMicrostructure,
    params=[
    parameter.StringParameter('microstructure', '',
                              tip='Name of the microstructure to be deleted.')],
    help="Destroy a Microstructure.",
    discussion=
    """<para>Delete a &micro; and all of its associated &images;,
    &skels;, and &meshes;.  Be really sure that you want to do this
    before you do it. </para>"""
    ))

## TODO: Add a Delete All command to remove all Microstructures (as
## requested by a user).

#########################

def saveMicrostructure(menuitem, filename, mode, format, microstructure):
    ms = ooflib.common.microstructure.microStructures[microstructure]
    dfile = datafile.writeDataFile(filename, mode.string(), format)
    microstructureIO.writeMicrostructure(dfile, ms)
    dfile.close()

mainmenu.OOF.File.Save.addItem(oofmenu.OOFMenuItem(
    'Microstructure',
    callback=saveMicrostructure,
    ordering=30,
    params=[
    filenameparam.WriteFileNameParameter('filename', tip="Name of the file."),
    filenameparam.WriteModeParameter('mode', tip="write or append?"),
    enum.EnumParameter('format', datafile.DataFileFormat, datafile.ASCII,
                       tip="File format."),
    whoville.WhoParameter('microstructure',
                          ooflib.common.microstructure.microStructures,
                          tip=parameter.emptyTipString)],
    help="Save a Microstructure in a file.",
    discussion="""
    <para>Store a &micro; in a file in one of several <link
    linkend='Section-Concepts-FileFormats'><varname>formats</varname></link>.
    The file can be reloaded by <xref
    linkend='MenuItem-OOF.File.Load.Script'/> or <xref
    linkend='MenuItem-OOF.File.Load.Data'/>, depending on the file
    format.</para>
    """,
    xrefs=["Section-Tasks-Microstructure"]
   ))

def _fixmenu(*args):
    import sys
    if ooflib.common.microstructure.microStructures.nActual() == 0:
        mainmenu.OOF.File.Save.Microstructure.disable()
    else:
        mainmenu.OOF.File.Save.Microstructure.enable()
_fixmenu()
    
switchboard.requestCallback(('new who', 'Microstructure'), _fixmenu)
switchboard.requestCallback(('remove who', 'Microstructure'), _fixmenu)


#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

# Set parameters used when computing element homogeneity.

saveTilingFactor = None
saveMinTileScale = None
saveFixedSubdivision = None

def setHomogParams(menuitem, factor, minimumTileSize, fixedSubdivision):
    if factor >= 1.0 or factor <= 0.0:
        raise ooferror.PyErrUserError("factor must be between 0 and 1.")

    global saveTilingFactor, saveMinTileScale, saveFixedSubdivision
    saveTilingFactor = pixelsetboundary.cvar.tilingfactor
    saveMinTileScale = pixelsetboundary.cvar.mintilescale
    saveFixedSubdivision = pixelsetboundary.cvar.fixed_subdivision
    
    pixelsetboundary.cvar.tilingfactor = factor
    pixelsetboundary.cvar.mintilescale = minimumTileSize
    if fixedSubdivision is automatic.automatic:
        pixelsetboundary.cvar.fixed_subdivision = 0
    else:
        pixelsetboundary.cvar.fixed_subdivision = fixedSubdivision

def resetHomogParams(menuitem):
    global saveTilingFactor, saveMinTileScale, saveFixedSubdivision
    pixelsetboundary.cvar.tilingfactor = saveTilingFactor
    pixelsetboundary.cvar.mintilescale = saveMinTileScale
    pixelsetboundary.cvar.fixed_subdivision = saveFixedSubdivision
    

micromenu.addItem(oofmenu.OOFMenuItem(
    'SetHomogeneityParameters',
    callback=setHomogParams,
    ordering=1000,
    secret=1,
    no_doc=1,
    params=[
        parameter.FloatRangeParameter(
            'factor', range=(0.0, 1.0, 0.01), default=0.5,
            tip='Refinement factor for hierarchical tiles'),
        parameter.IntParameter(
            'minimumTileSize', default=6,
            tip='Minimum tile size, in pixel units'),
        parameter.AutoIntParameter(
            'fixedSubdivision', default=automatic.automatic,
            tip='Fixed number of subdivisions')

    ],
    help="Set parameters for calculating element homogeneity.",
    discussion="""<para>This command is used in the regression tests
    to check aspects of the element homogeneity calculuation.</para>"""
))

micromenu.addItem(oofmenu.OOFMenuItem(
    'ResetHomogeneityParameters',
    callback=resetHomogParams,
    ordering=1001,
    secret=1,
    no_doc=1,
    help="Reset parameters for calculating element homogeneity.",
    discussion="""<para>This command is used in the regression tests
    to check aspects of the element homogeneity calculuation.</para>"""
))
    

# Force recategorization of pixels.  Used when testing.

def recategorize(menuitem, microstructure):
    ms = ooflib.common.microstructure.microStructures[microstructure]
    ms.getObject().recategorize()

micromenu.addItem(oofmenu.OOFMenuItem(
    'Recategorize',
    callback=recategorize,
    secret=1,
    no_doc=1,
    params=[whoville.WhoParameter('microstructure',
                                  ooflib.common.microstructure.microStructures,
                                  tip=parameter.emptyTipString)],
    help="Force pixel recategorization.",
    discussion="""<para>This command is used in the regression tests
    to check aspects of the element homogeneity calculuation.</para>"""
    ))


## This is for debugging the tiling computation in pixelsetboundary.C.
## There's no GUI access to this function.  

# def printHomogeneityStats(menuitem):
#     pixelsetboundary.printHomogStats()

# micromenu.addItem(oofmenu.OOFMenuItem(
#     'PrintHomogeneityStats',
#     callback=printHomogeneityStats,
#     ordering=1001,
#     help='Print useful information for testing and debugging'))
