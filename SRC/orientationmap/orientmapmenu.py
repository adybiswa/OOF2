# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

from ooflib.SWIG.common import latticesystem
from ooflib.SWIG.common import ooferror
from ooflib.SWIG.common import switchboard
from ooflib.SWIG.engine import angle2color
from ooflib.SWIG.orientationmap import orientmapdata
from ooflib.common import debug
from ooflib.common import enum
from ooflib.common import labeltree
from ooflib.common.IO import automatic
from ooflib.common.IO import filenameparam
from ooflib.common.IO import mainmenu
from ooflib.common.IO import microstructuremenu
from ooflib.common.IO import oofmenu
from ooflib.common.IO import reporter
from ooflib.common.IO import parameter
from ooflib.common.IO import whoville
from ooflib.common.IO import xmlmenudump
from ooflib.engine.IO import orientationmatrix
from ooflib.image.IO import imagemenu
import ooflib.common.microstructure
import os

## TODO: Add a way of rotating all orientations by a given amount,
## since the zero orientation may be defined differently than we
## expect it to be.  It should be possible to do this on a pixelgroup
## by pixelgroup basis, since the offsets may be different for
## different crystal symmetries.

## TODO: Add an AutoGroup command that creates pixel groups of
## contiguous pixels with similar orientations.

## TODO: Add an OrientationBurn pixel selection method that selects
## contiguous similarly oriented pixels.  Other pixel color selection
## operations could also be extended to orientations.

orientmapmenu = mainmenu.OOF.addItem(oofmenu.OOFMenuItem(
    'OrientationMap',
    cli_only=False,
    help='Commands for working with Orientation Maps.',
    discussion=xmlmenudump.loadFile(
            'DISCUSSIONS/orientationmap/menu/orientmapmenu.xml')
    ))

## Parameters for listing only those Microstructures that have or
## don't have Orientation Map data.  The two classes are identical
## here, but have different widgets.

class MicrostructureWithOrientMapParameter(whoville.WhoParameter):
    def __init__(self, name, value=None, default=None, tip=None):
        whoville.WhoParameter.__init__(self, name,
                                       whoville.getClass('Microstructure'),
                                       value=value, default=default, tip=tip)

class MicrostructureWithoutOrientMapParameter(whoville.WhoParameter):
    def __init__(self, name, value=None, default=None, tip=None):
        whoville.WhoParameter.__init__(self, name,
                                       whoville.getClass('Microstructure'),
                                       value=value, default=default, tip=tip)

####################

def _loadOrientationMap(menuitem, filename, reader, microstructure):
    # format is an Enum object.  microstructure is a name.
    mscontext = ooflib.common.microstructure.microStructures[microstructure]
    mscontext.reserve()
    mscontext.begin_writing()
    try:
        # Check to see if the microstructure already has an
        # orientation map.  It can't have more than one.
        if orientmapdata.getOrientationMap(mscontext.getObject()) is not None:
            raise ooferror.PyErrUserError(
                "A Microstructure can contain only one orientation map.")

        data = reader.read(filename)        # creates OrientMap object

        if (mscontext.getObject().sizeInPixels() != data.sizeInPixels() or
            mscontext.getObject().size() != data.size()):
            raise ooferror.PyErrUserError(
                "Cannot load orientation map into an existing Microstructure of a different size. ms=%sx%s (%dx%d), map=%sx%s (%dx%d)" % (
                mscontext.getObject().size()[0],
                mscontext.getObject().size()[1],
                mscontext.getObject().sizeInPixels()[0],
                mscontext.getObject().sizeInPixels()[1],
                data.size()[0], data.size()[1],
                data.sizeInPixels()[0], data.sizeInPixels()[1]))
        # Registering the OrientMapData object under its
        # microstructure's name allows it to be found by the
        # OrientationMapProp Property in C++.
        orientmapdata.registerOrientMap(microstructure, data)
        data.setMicrostructure(mscontext.getObject())
        # Storing it as a Python Microstructure plug-in allows the
        # data to be found by Python (and keeps a live reference to
        # it, so we don't have to transfer ownership to C++).  All of
        # this registration doesn't actually duplicate the data.
        orientmapplugin = mscontext.getObject().getPlugIn("OrientationMap")
        orientmapplugin.set_data(data, filename)
        orientmapplugin.timestamp.increment()
    finally:
        mscontext.end_writing()
        mscontext.cancel_reservation()
    reader.postProcess(mscontext)
    orientmapdata.orientationmapNotify(mscontext.getObject())


loadparams = [
    filenameparam.ReadFileNameParameter(
        'filename',
        tip="Name of the file containing the orientation map data."),
    parameter.RegisteredParameter('reader', orientmapdata.OrientMapReader,
                                  tip="Description of the data file format"),
    MicrostructureWithoutOrientMapParameter(
        'microstructure', tip='Add the orientation map to this Microstructure.')
    ]

mainmenu.OOF.File.Load.addItem(oofmenu.OOFMenuItem(
    'OrientationMap',
    callback=_loadOrientationMap,
    ordering=10,
    params=loadparams,
    help = "Load an orientation map into a Microstructure",
    discussion =xmlmenudump.loadFile('DISCUSSIONS/orientationmap/menu/load.xml')))

# OOF.OrientationMap.Load is the same as OOF.File.Load.OrientationMap
orientmapmenu.addItem(
    oofmenu.OOFMenuItem(
        'Load',
        callback=_loadOrientationMap,
        ordering=0,
        params=loadparams,
        help = "Load an orientation map into a Microstructure",
        discussion=xmlmenudump.loadFile(
            'DISCUSSIONS/orientationmap/menu/load.xml')
    ))

##############

def _deleteOrientationMap(menuitem, microstructure):
    ms = ooflib.common.microstructure.microStructures[microstructure]
    ms.reserve()
    ms.begin_writing()
    try:
        orientmapdata.removeOrientMap(microstructure) # remove C++ pointer
        orientmapdata.clearOrientationMap(ms.getObject()) # remove plug-in ref
    finally:
        ms.end_writing()
        ms.cancel_reservation()
    orientmapdata.orientationmapNotify(ms.getObject())

orientmapmenu.addItem(oofmenu.OOFMenuItem(
    'Delete',
    callback=_deleteOrientationMap,
    ordering=1,
    params=[MicrostructureWithOrientMapParameter(
        'microstructure',
        tip='Remove the orientation map from this Microstructure.')],
    help="Remove the orientation map from a Microstructure.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/orientationmap/menu/remove.xml')))

##############

def _createMSFromOrientationMapFile(menuitem, filename, reader, microstructure):
    data = reader.read(filename)        # creates OrientMap object
    if data is None:
        return
    ms = ooflib.common.microstructure.Microstructure(microstructure,
                                                   data.sizeInPixels(),
                                                   data.size())
    mscontext = ooflib.common.microstructure.microStructures[microstructure]
    mscontext.reserve()
    mscontext.begin_writing()
    try:
        orientmapdata.registerOrientMap(microstructure, data)
        data.setMicrostructure(mscontext.getObject())
        orientmapplugin = ms.getPlugIn('OrientationMap')
        orientmapplugin.set_data(data, filename)
        orientmapplugin.timestamp.increment()
    finally:
        mscontext.end_writing()
        mscontext.cancel_reservation()
    reader.postProcess(mscontext)
    orientmapdata.orientationmapNotify(ms)

def msOrientMapFileNameResolver(param, name):
    if param.automatic():
        basename = os.path.basename(param.group['filename'].value)
    else:
        basename = name
    # Remove colons from the basename.
    return ooflib.common.microstructure.microStructures.uniqueName(
        basename.replace(':','.'))

microstructuremenu.micromenu.addItem(
    oofmenu.OOFMenuItem(
        'Create_From_OrientationMap_File',
        callback=_createMSFromOrientationMapFile,
        params=parameter.ParameterGroup(
            filenameparam.ReadFileNameParameter(
                'filename',
                tip='Name of the Orientation Map file.'),
            parameter.RegisteredParameter(
                'reader', orientmapdata.OrientMapReader,
                tip="The method for reading the file."),
            parameter.AutomaticNameParameter(
                'microstructure',
                msOrientMapFileNameResolver,
                automatic.automatic,
                tip='Name of the new Microstructure.')),
        help="Load an Orientation Map file and create a Microstructure from it.",
        discussion=xmlmenudump.loadFile(
            'DISCUSSIONS/orientationmap/menu/create.xml'),
        xrefs=["Section-Tasks-Microstructure",
               "MenuItem-OOF.File.Load.OrientationMap"]
    ))


################

def _imageNameResolver(param, startname):
    msname = labeltree.makePath(param.group['microstructure'].value)[0]
    if param.automatic():
        # The automatic name for an image created from orientation
        # data is the filename of the data, minus any directory path
        # or suffix that it might have.
        msobj = ooflib.common.microstructure.microStructures[msname].getObject()
        basename = os.path.splitext(os.path.split(
            orientmapdata.getOrientationMapFile(msobj))[1])[0]
    else:
        basename = startname
    return whoville.getClass('Image').uniqueName([msname, basename])

def _imageFromOrientationMap(menuitem, microstructure, imagename, colorscheme):
    ms = ooflib.common.microstructure.microStructures[microstructure]
    orientdata = orientmapdata.getOrientationMap(ms.getObject())
    immidge = orientdata.createImage(imagename, colorscheme) # OOFImage object
    immidge.setSize(ms.getObject().size())
    immidge.setMicrostructure(ms.getObject())
    imagemenu.loadImageIntoMS(immidge, microstructure)
    switchboard.notify("redraw")

orientmapmenu.addItem(oofmenu.OOFMenuItem(
    'Convert_to_Image',
    callback=_imageFromOrientationMap,
    ordering=3,
    params= parameter.ParameterGroup(
    MicrostructureWithOrientMapParameter('microstructure',
                      tip='Convert the Orientation Map in this Microstructure'),
    whoville.AutoWhoNameParameter('imagename',
                                  resolver=_imageNameResolver,
                                  value=automatic.automatic,
                                  tip="Name to give to the new Image."),
    parameter.RegisteredParameter('colorscheme',
                                  angle2color.Angle2Color,
                                  tip="How to convert angles to colors.")),
    help="Convert an Orientation Map into an Image, so that pixel selection tools will work on it.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/orientationmap/menu/image.xml')
    ))
    

################

def _misorientation(menutitem, orientation1, orientation2, lattice_symmetry):
    # Orientation.misorientation(), defined in
    # engine/IO/orientationmatrix.py, returns an angle in degrees. It
    # calls COrientation.misorientation(), which returns an angle in
    # radians.
    misor = orientation1.misorientation(orientation2,
                                        lattice_symmetry.schoenflies())
    reporter.report("misorientation=", misor)

mainmenu.helpmenu.addItem(oofmenu.OOFMenuItem(
    'Misorientation_Calculator',
    callback=_misorientation,
    ordering=10,
    params=[
        parameter.RegisteredParameter('orientation1',
                                      orientationmatrix.Orientation,
                                      tip="An orientation"),
        parameter.RegisteredParameter('orientation2',
                                      orientationmatrix.Orientation,
                                      tip="Another orientation"),
        latticesystem.LatticeSymmetryParameter(
            'lattice_symmetry',
            value=latticesystem.SpaceGroup(1),
            tip="Lattice symmetry")
        ],
    help="Print the misorientation (in degrees) between two orientations.",
    discussion="""<para>Print the <link
    linkend="Section-Concepts-Material-Orientation">misorientation</link>
    between two 3D orientations in the given lattice system.</para>""",
    xrefs=["Section-Concepts-Material-Orientation",
           "Section-Graphics-PixelInfo-Misorient"]
))
    
    

################

# Sensitize the OrientationMap menus when Microstructures are added or deleted.

def _sensitize(path):
    msclass = whoville.getClass('Microstructure')
    # If this module was somehow loaded first, the Microstructure
    # class might not be defined.
    if msclass and msclass.nActual() > 0:
        orientmapmenu.enable()
        mainmenu.OOF.File.Load.OrientationMap.enable()
    else:
        orientmapmenu.disable()
        mainmenu.OOF.File.Load.OrientationMap.disable()

_sensitize(None)

switchboard.requestCallback(('new who', 'Microstructure'),
                            _sensitize)
switchboard.requestCallback(('remove who', 'Microstructure'),
                            _sensitize)
