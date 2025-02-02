# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.common import color
from ooflib.common import debug
from ooflib.common import registeredclass
from ooflib.common.IO import display
from ooflib.common.IO import ghostgfxwindow
from ooflib.common.IO import mainmenu
from ooflib.common.IO import oofmenu
from ooflib.common.IO import parameter
from ooflib.common.IO import xmlmenudump

import oofcanvas

# This object should be created via the registration, and not
# directly via the initializer, becasue the registration creation
# method gives it a timestamp.

class SkeletonNodeSelectionDisplay(display.DisplayMethod):
    def __init__(self, color, size):
        self.color = color
        self.size = size
        display.DisplayMethod.__init__(self)
    def draw(self, gfxwindow):
        skel = self.who.resolve(gfxwindow)
        if skel is not None:
            clr = color.canvasColor(self.color)
            retr = skel.nodeselection.retrieve().copy()
            size = len(retr)
            for node in retr:
                pt = node.position()

                dot = oofcanvas.CanvasDot.create(pt, 1.2*self.size)
                dot.setFillColor(oofcanvas.white.opacity(self.color.getAlpha()))
                self.canvaslayer.addItem(dot)

                dot = oofcanvas.CanvasDot.create(pt, self.size)
                dot.setFillColor(clr)
                self.canvaslayer.addItem(dot)
                
    def getTimeStamp(self, gfxwindow):
        return max(self.timestamp,
                   self.who.resolve(gfxwindow).nodeselection.timestamp)



defaultNodeSelColor = color.RGBAColor(0.07, 0.09, 0.96, 1.0)
defaultNodeSelSize = 5

def _setNodeSelParams(menuitem, color, size):
    global defaultNodeSelColor
    global defaultNodeSelSize
    defaultNodeSelColor = color
    defaultNodeSelSize = size

nodeselparams = [
    color.TranslucentColorParameter('color', defaultNodeSelColor,
                                    tip="Color for the selected nodes."),
    parameter.IntRangeParameter('size', (0,20), defaultNodeSelSize,
                                tip="Node size.")]

mainmenu.gfxdefaultsmenu.Skeletons.addItem(oofmenu.OOFMenuItem(
    'Selected_Nodes',
    callback=_setNodeSelParams,
    ordering=2,
    params=nodeselparams,
    help="Set default parameters for displaying selected skeleton nodes.",
    discussion="""<para>

    Set default parameters for the
    <xref linkend="RegisteredClass-SkeletonNodeSelectionDisplay"/>,
    which displays the currently selected &skel; &nodes; in the graphics
    window.  See
    <xref linkend="RegisteredClass-SkeletonNodeSelectionDisplay"/>
    for a discussion of the parameters. This command may be put in the
    &oof2rc; file to set defaults for all &oof2; sessions.
    
    </para>""",
    xrefs=["Section-Tasks-SkeletonSelection"]
))

nodeSelectDisplay = registeredclass.Registration(
    'Selected Nodes',
    display.DisplayMethod,
    SkeletonNodeSelectionDisplay,
    params=nodeselparams,
    ordering=2.2,
    layerordering= display.PointLike(3),
    whoclasses=('Skeleton',),
    tip="Display the currently selected nodes.",
    discussion=xmlmenudump.loadFile(
        'DISCUSSIONS/engine/reg/nodeselectdisplay.xml'),
    xrefs=["Section-Tasks-SkeletonSelection"]
    )

def defaultNodeSelectDisplay():
    return nodeSelectDisplay(color=defaultNodeSelColor, size=defaultNodeSelSize)

ghostgfxwindow.PredefinedLayer('Skeleton', '<topmost>',
                               defaultNodeSelectDisplay)
