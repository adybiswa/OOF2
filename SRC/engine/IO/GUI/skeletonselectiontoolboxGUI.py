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
from ooflib.SWIG.common.IO.GUI.OOFCANVAS import oofcanvasgui
from ooflib.common import debug
from ooflib.common.IO.GUI import genericselectGUI
from ooflib.common.IO.GUI import gtklogger
from ooflib.common.IO.GUI import regclassfactory
from ooflib.common.IO.GUI import toolboxGUI
from ooflib.engine import skeletonselectionmethod
from ooflib.engine import skeletonselmodebase

from gi.repository import Gtk

# The SkeletonSelectionToolbox GUI is a ToolboxGUI that contains other
# ToolboxGUI's.  The inner GUI's are instances of
# SkeletonSelectionToolboxModeGUI.  Inner toolboxes are selected by a
# set of radio buttons at the top of the outer toolbox.  The inner
# toolboxes and the buttons are created automatically from the
# SkeletonSelectionMode classes.  Each of the inner gui toolboxes
# corresponds to a non-gui toolbox class.  From the gfxwindow's point
# of view, though, there's only one gui toolbox (the outer one), so
# only one of the non-gui toolboxes has a makeGUI routine attached to
# it.

class SkeletonSelectionToolboxModeGUI(genericselectGUI.GenericSelectToolboxGUI):
    def __init__(self, mode, tb):
        self.mode = mode
        genericselectGUI.GenericSelectToolboxGUI.__init__(self, mode.name,
                                                          tb, mode.methodclass)
        # Switchboard callbacks that should be performed even when the
        # toolbox isn't active go here.  Callbacks that are performed
        # only when the toolbox IS active are installed in activate().
        self.sbcallbacks.append(
            switchboard.requestCallbackMain(self.mode.newselectionsignal,
                                            self.newSelection)
            )

    def methodFactory(self, **kwargs):
        return regclassfactory.RegisteredClassFactory(
            self.method.registry, title="Method:", name="Method", **kwargs)
                                                          
    def activate(self):
        genericselectGUI.GenericSelectToolboxGUI.activate(self)
        self.activecallbacks = [
            switchboard.requestCallbackMain((self.gfxwindow(),
                                             'layers changed'),
                                            self.layerChangeCB) ,
            switchboard.requestCallbackMain(self.mode.changedselectionsignal,
                                            self.changedSelection)
            ]
    def deactivate(self):
        genericselectGUI.GenericSelectToolboxGUI.deactivate(self)
        map(switchboard.removeCallback, self.activecallbacks)
        self.activecallbacks = []

    def getSource(self):
        return self.gfxwindow().topwho('Skeleton')

    def finish_up(self, ptlist, shift, ctrl, selmeth):
        self.setCoordDisplay(selmeth, ptlist)
        self.selectionMethodFactory.set_defaults()
        menuitem = getattr(self.toolbox.menu, selmeth.name())
        menuitem.callWithDefaults(skeleton=self.getSourceName(),
                                  points=ptlist, shift=shift, ctrl=ctrl)
    def undoCB(self, button):
        self.toolbox.menu.Undo(skeleton=self.getSourceName())

    def redoCB(self, button):
        self.toolbox.menu.Redo(skeleton=self.getSourceName())

    def clearCB(self, button):
        self.toolbox.menu.Clear(skeleton=self.getSourceName())

    def invertCB(self, button):
        self.toolbox.menu.Invert(skeleton=self.getSourceName())

    def hide(self):
        self.gtk.hide()

    def show(self):
        self.gtk.show_all()

        
class SkeletonSelectionToolboxGUI(toolboxGUI.GfxToolbox):
    def __init__(self, toolbox):
        # The 'toolbox' argument here is the non-gui toolbox
        # corresponding to one of the inner toolboxes.  It doesn't
        # matter which one.
        toolboxGUI.GfxToolbox.__init__(self, "Skeleton Selection", toolbox)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.gtk.add(vbox)
        bbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        gtklogger.setWidgetName(bbox, "Select")
        vbox.pack_start(bbox, expand=False, fill=False, padding=0)
        bbox.pack_start(Gtk.Label("Select: "),
                        expand=False, fill=False, padding=0)

        # self.tbbox = Gtk.Frame()       # holds SkeletonSelectionToolboxModes
        self.tbbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        vbox.pack_start(self.tbbox, expand=True, fill=True, padding=0)
        
        group = None
        self.tbdict = {}
        for mode in skeletonselmodebase.SkeletonSelectionMode.modes:
            if group:
                button = Gtk.RadioButton(mode.name, group=group)
            else:
                button = Gtk.RadioButton(mode.name)
                group = button
            bbox.pack_start(button, expand=False, fill=False, padding=0)
            gtklogger.setWidgetName(button, mode.name)
            gtklogger.connect(button, 'clicked', self.switchModeCB, mode.name)

            ## Get the actual toolbox for each mode.
            tb = self.gfxwindow().getToolboxByName(mode.toolboxName())
            tbgui = SkeletonSelectionToolboxModeGUI(mode, tb)
            self.tbdict[mode.name] = tbgui

        self.activecallbacks = []
        self.currentMode = None

    def switchModeCB(self, button, modename):
        if button.get_active() and self.currentMode != modename:
            self.setMode(modename)
        
    def setMode(self, modename):
        debug.mainthreadTest()
        if self.currentMode:
            mode = self.tbdict[self.currentMode]
            mode.deactivate()
            self.tbbox.remove(self.tbbox.get_children()[0])
        self.currentMode = modename
        mode = self.tbdict[modename]
        self.tbbox.add(mode.gtk)
        mode.show()
        mode.activate()

    def activate(self):
        if self.currentMode is None:
            self.setMode(
                skeletonselmodebase.SkeletonSelectionMode.modes[0].name)
        self.tbdict[self.currentMode].activate()
        toolboxGUI.GfxToolbox.activate(self)
    def deactivate(self):
        self.tbdict[self.currentMode].deactivate()
        toolboxGUI.GfxToolbox.deactivate(self)


######################################

## Although there are many non-gui SkeletonSelectionToolboxes, they
## all share a GUI panel, so only one of them has a makeGUI function.

def _makeGUI(self):
    return SkeletonSelectionToolboxGUI(self)

skeletonselmodebase.firstMode().tbclass.makeGUI = _makeGUI

#####################################

## Assignment of rubberband types to SkeletonSelectionRegistration
## instances.  Most assignments are to *instances*, and as such are
## not member functions.  The default assignment (no rubberband) is to
## the class, and so the function needs a 'self' argument.

def _NoRubberBand(self, reg):
    return None

skeletonselectionmethod.SkeletonSelectionRegistration.getRubberBand = \
    _NoRubberBand

def _RectangleSelectorRB(reg):
    return oofcanvasgui.RectangleRubberBand()

def _CircleSelectorRB(reg):
    return oofcanvasgui.CircleRubberBand()

def _EllipseSelectorRB(reg):
    return oofcanvasgui.EllipseRubberBand()

skeletonselectionmethod.rectangleNodeSelector.getRubberBand = \
    _RectangleSelectorRB
skeletonselectionmethod.circleNodeSelector.getRubberBand = \
    _CircleSelectorRB
skeletonselectionmethod.ellipseNodeSelector.getRubberBand = \
    _EllipseSelectorRB
skeletonselectionmethod.rectangleSegmentSelector.getRubberBand = \
    _RectangleSelectorRB
skeletonselectionmethod.circleSegmentSelector.getRubberBand = \
    _CircleSelectorRB
skeletonselectionmethod.ellipseSegmentSelector.getRubberBand = \
    _EllipseSelectorRB
skeletonselectionmethod.rectangleElementSelector.getRubberBand = \
    _RectangleSelectorRB
skeletonselectionmethod.circleElementSelector.getRubberBand = \
    _CircleSelectorRB
skeletonselectionmethod.ellipseElementSelector.getRubberBand = \
    _EllipseSelectorRB
