# -*- python -*-


# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.common import config
from ooflib.SWIG.common import switchboard
from ooflib.SWIG.common import pixelattribute
from ooflib.SWIG.common import activearea
from ooflib.common import debug
from ooflib.common import primitives
from ooflib.common.IO import mainmenu
from ooflib.common.IO import oofmenu
from ooflib.common.IO import parameter
from ooflib.common.IO import whoville
from ooflib.common.IO import xmlmenudump
import ooflib.common.microstructure

OOF = mainmenu.OOF
micromenu = OOF.LoadData.addItem(oofmenu.OOFMenuItem(
    'Microstructure',
    help='Commands describing Microstructures in data files.',
    discussion=xmlmenudump.loadFile(
    'DISCUSSIONS/common/menu/microstructureIO.xml')
    ))

#########

import sys
def _newMicrostructure(menuitem, name, isize, size):
    sys.stdout.flush()
    try:
        oldms = ooflib.common.microstructure.microStructures[name]
    except KeyError:
        pass
    else:
        oldms.lockAndDelete()
    ms = ooflib.common.microstructure.Microstructure(name, isize, size)

micromenu.addItem(oofmenu.OOFMenuItem(
    'New',
    callback=_newMicrostructure,
    params=[parameter.StringParameter('name', tip="Name of Microstructure."),
            primitives.iPointParameter('isize', tip="Pixel resolution of Microstructure."),
            primitives.PointParameter('size', tip="Physical size of Microstructure.")],
    help="Create a new Microstructure.  Used internally in data files.",
    
    discussion="""<para>Create a new <quote>empty</quote> &micro;,
    containing no data other than its size.  This command is used only
    in <link linkend='MenuItem-OOF.LoadData.Microstructure'>data
    files</link>.</para>"""
    ))

##########    

def _newpixelgroup(menuitem, microstructure, group, meshable):
    mscontext = ooflib.common.microstructure.microStructures[microstructure]
    ms = mscontext.getObject()
    mscontext.begin_writing()
    try:
        (grp, newness) = ms.getGroup(group)
        grp.set_meshable(meshable)
    finally:
        mscontext.end_writing()
        
    if newness:
        switchboard.notify("new pixel group", grp)


micromenu.addItem(oofmenu.OOFMenuItem(
    'PixelGroup',
    callback=_newpixelgroup,
    params=[whoville.WhoParameter('microstructure',
                                  ooflib.common.microstructure.microStructures,
                                  tip=parameter.emptyTipString),
            parameter.StringParameter('group', tip="Name of the pixel group."),
            parameter.BooleanParameter('meshable',
                                       tip="1 (true) for meshable or 0 (false) for non-meshable.")
            ],
    help="Create a pixel group. Used internally in data files.",

    discussion="""<para>Create <link
    linkend='Section-Concepts-Microstructure-Pixel_Group'>pixel
    groups</link> in a &micro; <link
    linkend='MenuItem-OOF.LoadData.Microstructure'>data
    file</link>.  This command is only used in data files.</para>"""
    ))

##########

categorymenu = micromenu.addItem(
    oofmenu.OOFMenuItem(
        'DefineCategory',
        help="Define pixel categories in a data file.",
        discussion="""<para>Commands that define pixel attributes
        (Material, Group membership, etc) for categories of pixels
        in a &micro; data file.</para>
        """,
    ))


class CategoryMap:
    def __init__(self, array):
        # array is a list of lists of Ints, which are pixel categories
        self.pxls = {}                  # list of pixels, keyed by category
        for j, sublist in enumerate(array):
            if config.dimension() == 2:
                for i,category in enumerate(sublist):
                    where = primitives.iPoint(i, j)
                    try:
                        self.pxls[category].append(where)
                    except KeyError:
                        self.pxls[category] = [where]

            elif config.dimension() == 3:
                for i, subsublist in enumerate(sublist):
                    for k, category in enumerate(subsublist):
                        where = primitives.iPoint(k, i, j)
                        try:
                            self.pxls[category].append(where)
                        except KeyError:
                            self.pxls[category] = [where]
                        
    def getPixels(self, category):
        try:
            return self.pxls[category]
        except KeyError:
            return []
            
categoryMap = {} # CategoryMaps keyed by microstructure name

def getCategoryPixels(microstructure, category):
    map = categoryMap[microstructure]
    return map.getPixels(category)
    

def _readCategories(menuitem, microstructure, categories):
    categoryMap[microstructure] = CategoryMap(categories)

if config.dimension() == 2:
    categoryparams = [whoville.WhoParameter('microstructure',
                                  ooflib.common.microstructure.microStructures,
                                  tip=parameter.emptyTipString),
            parameter.ListOfListOfIntsParameter('categories',
                                               tip="Category map for pixels.")
            ]
    
elif config.dimension() == 3:
    categoryparams = [whoville.WhoParameter('microstructure',
                                  ooflib.common.microstructure.microStructures,
                                  tip=parameter.emptyTipString),
            parameter.ListOfListOfListOfIntsParameter('categories',
                                               tip="Category map for pixels.")
            ]
    
micromenu.addItem(oofmenu.OOFMenuItem(
    'Categories',
    callback=_readCategories,
    params=categoryparams,
    help="Assign pixels to categories. Used internally in data files.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/categories.xml')
    ))

#########

def _endCategories(menuitem, microstructure):
##    debug.fmsg()
    del categoryMap[microstructure]

micromenu.addItem(oofmenu.OOFMenuItem(
    'EndCategories',
    callback=_endCategories,
    params=[whoville.WhoParameter('microstructure',
                                  ooflib.common.microstructure.microStructures,
                                  tip=parameter.emptyTipString)],
    help="Clean up after pixel category definition.  Used internally in data files.",

    discussion="""<para>Clean up the data structures used to define
    <link linkend='Definition:Category'>pixel categories</link> in
    <link linkend='MenuItem-OOF.LoadData.Microstructure'>data
    files</link> usage.  This command is not used outside of data
    files.</para>"""

    ))

##########

# When other modules add data to Microstructures, they have to also
# make sure that the data is stored in output files.  They can
# register a function to do that by calling
# registerMicrostructureIOPlugIn.  The argument to
# registerMicrostructureIOPlugIn will be called when the MS is
# written, with the MS context and datafile as arguments.
  
_ioplugins = []
def registerMicrostructureIOPlugIn(func):
    _ioplugins.append(func)

#Interface branch
#The functions in this list get called
#at the end of writeMicrostructure
_ioplugins_last = []
def registerMicrostructureIOPlugIn_last(func):
    _ioplugins_last.append(func)

##########

def writeMicrostructure(datafile, mscontext):
    ms = mscontext.getObject()
    
    mscontext.begin_reading()
    try:
        datafile.startCmd(OOF.LoadData.Microstructure.New)
        datafile.argument('name', ms.name())
        datafile.argument('size', ms.size())
        datafile.argument('isize', ms.sizeInPixels())
        datafile.endCmd()

        for ioplugin in _ioplugins:
            ioplugin(datafile, mscontext)
##        for image in ms.getImageContexts():
##            image.writeImage(datafile)

        # Store pixel attributes by storing category definitions.
        for grpname in ms.groupNames():
            grp = ms.findGroup(grpname)
            datafile.startCmd(OOF.LoadData.Microstructure.PixelGroup)
            datafile.argument('microstructure', ms.name())
            datafile.argument('group', grpname)
            datafile.argument('meshable', grp.is_meshable())
            datafile.endCmd()
        # Create the actual active areas themselves.
        for aaname in ms.activeAreaNames():
            datafile.startCmd(OOF.LoadData.Microstructure.NewActiveArea)
            datafile.argument('microstructure', ms.name())
            datafile.argument('name', aaname)
            datafile.endCmd()
        categories = ms.getCategoryMapRO()
        # Save categories
        datafile.startCmd(OOF.LoadData.Microstructure.Categories)
        datafile.argument('microstructure', ms.name())
        datafile.argument('categories', categories)
        datafile.endCmd()
        # Find representative pixels for each category
        reppxls = {}
        i = 0
        if config.dimension() == 2:
            for row in categories:
                j = 0
                for ctgry in row:
                    reppxls[ctgry] = primitives.iPoint(j,i)
                    j += 1
                i += 1
        elif config.dimension() == 3:
            for slab in categories:
                j = 0
                for row in slab:
                    k = 0
                    for ctgry in row:
                        reppxls[ctgry] = primitives.iPoint(k,j,i)
                        k += 1
                    j += 1
                i += 1
        # Save definitions of pixel categories
        for i in range(pixelattribute.nAttributes()):
            reg = pixelattribute.getRegistration(i)
            reg.writeGlobalData(datafile, ms)
            for category in range(len(reppxls)):
                reppxl = reppxls[category]
                datafile.startCmd(getattr(
                    OOF.LoadData.Microstructure.DefineCategory, reg.name()))
                datafile.argument('microstructure', ms.name())
                datafile.argument('category', category)
                if reg.writeData(datafile, ms, reppxl):
                    datafile.endCmd()
                else:
                    datafile.discardCmd()
        datafile.startCmd(OOF.LoadData.Microstructure.EndCategories)
        datafile.argument('microstructure', ms.name())
        datafile.endCmd()

        #Interface branch
        #Note that materials that are not assigned to pixels do not get saved
        #when the microstructure is saved, if we rely on the above mechanism
        #(reg.writeGlobalData is not called for these materials).
        for ioplugin in _ioplugins_last:
            ioplugin(datafile, mscontext)

    finally:
        mscontext.end_reading()
