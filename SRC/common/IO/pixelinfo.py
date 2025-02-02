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
from ooflib.SWIG.common import timestamp
from ooflib.common import debug
from ooflib.common import toolbox
from ooflib.common import primitives
from ooflib.common.IO import oofmenu
from ooflib.common.IO import parameter
from ooflib.common.IO import reporter

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

# Plug-in classes can add new menu items.  To be useful they probably
# should correspond to a PixelInfoGUIPlugIn that invokes the menu
# items.  They should be derived from PixelInfoPlugin, and re-define
# makeMenu.

# Subclasses of PixelInfoPlugIn to instantiate in each toolbox instance
plugInClasses = []       

# The meta class for PixelInfoPlugIn takes care of listing each
# subclass in plugInClasses, and notifies existing graphics windows of
# the plug in, in case the windows already exist when the plugin is
# loaded.

class PixelInfoPlugInMetaClass(type):
    def __init__(cls, name, bases, dict):
        super(PixelInfoPlugInMetaClass, cls).__init__(name, bases, dict)
        plugInClasses.append(cls)
        switchboard.notify("new pixel info plugin", cls)

class PixelInfoPlugIn(object, metaclass=PixelInfoPlugInMetaClass):
    def __init__(self, toolbox):
        self.toolbox = toolbox
    def makeMenu(self, menu):
        # The argument is the toolbox's menu, to which new commands
        # should be added.
        pass
    def draw(self, displaymethod, canvaslayer, pixel, microstructure):
        # Define this for any plug-in that wants to draw something on
        # the canvas.  PixelInfoDisplay.draw() calls each plug-in's
        # draw() method.  The displaymethod argument is the
        # PixelInfoDisplay.  The pixel argument is the current pixel
        # of the PixelInfoToolbox, which may or may not be the pixel
        # that the plug-in wants to draw.
        pass

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

class PixelInfoToolbox(toolbox.Toolbox):
    def __init__(self, gfxwindow):
        self.point = None               # location of last query
        self.timestamp = timestamp.TimeStamp()
        toolbox.Toolbox.__init__(self, "Pixel_Info", gfxwindow)

        self.plugIns = [plugInClass(self) for plugInClass in plugInClasses]

        self.sbcallbacks = [
            switchboard.requestCallback("new pixelinfo plugin",
                                        self.newPlugIn)
            ]

    def makeMenu(self, menu):
        self.menu = menu
        menu.xrefs.append("Section-Graphics-PixelInfo")
        positionparams=[parameter.IntParameter('x', 0, tip="The x coordinate."),
                        parameter.IntParameter('y', 0, tip="The y coordinate.")]
        helpstring="Query the pixel that is closest to the given point(x,y)."
        menu.addItem(oofmenu.OOFMenuItem(
            'Query',
            callback=self.queryPixel,
            params=positionparams,
            help=helpstring,
            discussion="""<para>
            Display information about the pixel that is closest to the
            given point.  In GUI mode, the information appears in the
            <link linkend='Section-Graphics-PixelInfo'>Pixel
            Info</link> toolbox in the graphics window.  This command
            has no effect when the GUI is not running.
            </para>"""
            ))
        menu.addItem(oofmenu.OOFMenuItem(
            'Clear',
            callback=self.clearPixel,
            params=[],
            help="Reset the pixel info toolbox.",
            discussion="""<para>
            Clear any displayed information from previous mouse clicks.
            In GUI mode, this clears the <link
            linkend='Section-Graphics-PixelInfo'>Pixel Info</link>
            toolbox in the graphics window.  This command has no
            effect if the GUI is not running.
            </para>"""
            ))

        for plugin in self.plugIns:
            plugin.makeMenu(menu)

    def close(self):
        switchboard.removeCallbacks(self.sbcallbacks)
        toolbox.Toolbox.close(self)

    def newPlugIn(self, pluginClass):
        self.plugIns.append(pluginClass(self))

    def findPlugIn(self, pluginClass):
        for plugin in self.plugIns:
            if isinstance(plugin, pluginClass):
                return plugin

    def queryPixel(self, menuitem, x, y): # menu callback
        self.timestamp.increment()
        self.point = primitives.iPoint(x, y)
        switchboard.notify(self)        # caught by GUI toolbox
        switchboard.notify('redraw')

    def clearPixel(self, menuitem): # Menu callback.
        self.timestamp.increment()
        self.point = None
        switchboard.notify(self)
        switchboard.notify('redraw')
        
    def currentPixel(self):
        return self.point

    def findMicrostructure(self):
        ## This used to check for Skeletons and Meshes, too, and use
        ## their getMicrostructure function.  That led to some
        ## confusing situations, when a Skeleton was displayed over an
        ## Image from a different Microstructure, for example.  So now
        ## it doesn't return anything when no Microstructure or Image
        ## is displayed.
        who = self.gfxwindow().topwho('Microstructure', 'Image')
        if who is not None:
            return who.getMicrostructure()
        return None

    def getTimeStamp(self):
        return self.timestamp

    tip="Get information about a pixel."
    discussion="""<para>
    Get information about a pixel, based on mouse input.
    </para>"""
    
toolbox.registerToolboxClass(PixelInfoToolbox, ordering=1.0)
