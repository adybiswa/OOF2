# -*- python -*-


# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.common import switchboard
from ooflib.common.IO import placeholder
from ooflib.common.IO import whoville
from ooflib.common import debug
from ooflib.common.IO.GUI import chooser
from ooflib.common.IO.GUI import gtklogger
from ooflib.common.IO.GUI import parameterwidgets
from ooflib.common.IO.GUI import whowidget
from ooflib.engine import mesh
from ooflib.engine.IO import meshcsparams
from ooflib.engine.IO import meshmenu

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

# Widget for MeshCrossSectionSet objects.  Just puts up a list of the
# things, and lets the user pick any number of them, or use the
# "selected" one, which has meaning in the context of a mesh object.
# Return value is a (possibly empty) list of strings, or
# placeholder.selection.

# This widget is currently used in the LayerEditor, so it may be
# assumed that a mesh widget that gives a nontrivial result will
# always be found.
class MeshCrossSectionSetParamWidget(parameterwidgets.ParameterWidget):
    def __init__(self, param, scope, name=None, **kwargs):
        debug.mainthreadTest()
        # Find the enclosing mesh widget.
        self.meshwidget = scope.findWidget(
            lambda x: isinstance(x, whowidget.WhoWidget)
            and x.whoclass is mesh.meshes)

        self.gtk = Gtk.Frame()
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2,
                       **kwargs)
        self.gtk.add(vbox)

        meshname = self.meshwidget.get_value()
        try:
            self.meshobj = mesh.meshes[meshname]
        except KeyError:
            # no mesh!
            self.meshobj = None
            vbox.add(Gtk.Label(label='No mesh!'))
        else:
            # In the proxy case, the behavior is to set the line
            # width, etc. for all the cross sections in whatever mesh
            # is referred to.  It doesn't make sense to single out
            # individual cross-sections in this case, since they'll
            # change when the proxy resolution changes.  The set_value
            # and get_value functions must also detect the proxy case
            # and behave a little differently, since the chooser
            # doesn't get built in that case.
            if isinstance(self.meshobj, whoville.WhoProxy):
                vbox.add(Gtk.Label(label='Selected'))
            else:
                self.chooser = chooser.MultiListWidget(
                    self.meshobj.allCrossSectionNames(), name="List",
                    hexpand=True, halign=Gtk.Align.FILL)
                self.selected = Gtk.CheckButton(label="Selected")
                gtklogger.setWidgetName(self.selected, "Selected")
                gtklogger.connect(self.selected, "clicked", self.selectedCB)
                vbox.pack_start(self.selected,
                                expand=False, fill=True, padding=0)
                vbox.pack_start(self.chooser.gtk,
                                expand=False, fill=True, padding=0)
      
        parameterwidgets.ParameterWidget.__init__(self, self.gtk, scope,
                                                  name=name,
                                                  expandable=True)
        self.set_value(param.value)

        self.gtk.show_all()

    def selectedCB(self, gtkobj):
        debug.mainthreadTest()
        if self.selected.get_active():
            self.set_value(placeholder.selection)
        else:
            self.chooser.clear()
            self.chooser.gtk.set_sensitive(True)
            lcsn=self.meshobj.selectedCSName()
            if lcsn:
                self.chooser.set_selection([lcsn])
        self.widgetChanged(True, interactive=True)

  
    def set_value(self, value):
        debug.mainthreadTest()
        if self.meshobj is not None:
            if isinstance(self.meshobj, whoville.WhoProxy):
                # There is no chooser in the proxy case.
                self.widgetChanged(True,interactive=False)
            else:
                self.chooser.update(self.meshobj.allCrossSectionNames())
                self.chooser.clear()
                if value==placeholder.selection:
                    lcsn=self.meshobj.selectedCSName()
                    if lcsn:
                        self.chooser.set_selection([lcsn])
                    self.chooser.gtk.set_sensitive(False)
                    self.selected.set_active(True)
                else:  # Value is list of names.
                    self.chooser.gtk.set_sensitive(True)
                    self.selected.set_active(False)
                    self.chooser.set_selection(value)
                self.widgetChanged(True, interactive=False)
        else:
            self.widgetChanged(False, interactive=False)
          
    def get_value(self):
        if self.meshobj is not None:
            if isinstance(self.meshobj, whoville.WhoProxy):
                return placeholder.selection
            else:
                if self.selected.get_active():
                    return placeholder.selection
                else:
                    return self.chooser.get_value()

    def cleanUp(self):
        self.chooser = None
        parameterwidgets.ParameterWidget.cleanUp(self)

def _make_MCSSPWidget(self, scope, **kwargs):
    return MeshCrossSectionSetParamWidget(self, scope, name=self.name, **kwargs)

meshcsparams.MeshCrossSectionSetParameter.makeWidget = _make_MCSSPWidget

###################

# Widget for picking a single cross section from the set of cross
# sections already defined on a mesh.  Also includes buttons for
# defining and editing cross sections.

class MeshCrossSectionParamWidget(parameterwidgets.ParameterWidget):
    def __init__(self, param, scope, name=None, **kwargs):
        debug.mainthreadTest()
        # Find the enclosing mesh widget.
        self.meshwidget = scope.findWidget(
            lambda x: isinstance(x, whowidget.WhoWidget)
            and x.whoclass is mesh.meshes)

        meshname = self.meshwidget.get_value()
        if meshname:
            self.meshobj = mesh.meshes[meshname]
            if issubclass(self.meshobj.__class__, whoville.WhoProxy):
                gfxwindow = scope.findData("gfxwindow")
                self.meshobj = mesh.resolve(gfxwindow)
        else:
            self.meshobj = None

        frame = Gtk.Frame(**kwargs)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2,
                       margin=2)
        frame.add(vbox)

        self.chooser = chooser.ChooserWidget([], callback=self.chooserCB,
                                             name="List")
        vbox.pack_start(self.chooser.gtk, expand=False, fill=False, padding=0)

        bbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        vbox.pack_start(bbox, expand=False, fill=False, padding=0)

        self.newbutton = Gtk.Button(label='New')
        gtklogger.setWidgetName(self.newbutton, "New")
        gtklogger.connect(self.newbutton, 'clicked', self.newCB)
        bbox.pack_start(self.newbutton, expand=True, fill=True, padding=0)

        self.copybutton = Gtk.Button(label='Copy')
        gtklogger.setWidgetName(self.copybutton, "Copy")
        gtklogger.connect(self.copybutton, 'clicked', self.copyCB)
        bbox.pack_start(self.copybutton, expand=True, fill=True, padding=0)

        self.editbutton = Gtk.Button(label='Edit')
        gtklogger.setWidgetName(self.editbutton, "Edit")
        gtklogger.connect(self.editbutton, 'clicked', self.editCB)
        bbox.pack_start(self.editbutton, expand=True, fill=True, padding=0)

        self.renamebutton = Gtk.Button(label='Rename')
        gtklogger.setWidgetName(self.renamebutton, "Rename")
        gtklogger.connect(self.renamebutton, 'clicked', self.renameCB)
        bbox.pack_start(self.renamebutton, expand=True, fill=True, padding=0)

        self.deletebutton = Gtk.Button(label='Remove')
        gtklogger.setWidgetName(self.deletebutton, "Remove")
        gtklogger.connect(self.deletebutton, 'clicked', self.deleteCB)
        bbox.pack_start(self.deletebutton, expand=True, fill=True, padding=0)

        parameterwidgets.ParameterWidget.__init__(self, frame, scope, name=name)

        self.sbcbs = [
            switchboard.requestCallbackMain(self.meshwidget,
                                            self.mesh_update),
            switchboard.requestCallbackMain("cross sections changed",
                                            self.cs_update),
            switchboard.requestCallbackMain(("cross section renamed",
                                             self.meshobj),
                                            self.renamedCS)
            ]



        self.update(interactive=False)

        # The parameter could have a non-None value which is not in
        # the list of allowed choices when this widget is created.  In
        # that case, do nothing.  If the parameter has a non-null
        # value that *is* in the list, set the widget to that value.
        if param.value in self.chooser.choices():
            self.set_value(param.value)

        self.gtk.show_all()

    def mesh_update(self, interactive):
        self.update(interactive)
    def cs_update(self):
        self.update(interactive=False)
        
    def update(self, interactive):
        meshname = self.currentMeshName()
        if meshname:
            self.meshobj = mesh.meshes[meshname]
            self.chooser.update(self.meshobj.allCrossSectionNames())
            self.chooser.set_state(self.meshobj.selectedCSName())
            self.sensitize()
        else:
            self.meshobj = None
            self.chooser.update([])
            self.sensitize()
        # This update can affect the validity, of course.
        self.widgetChanged(bool(self.chooser.get_value()), interactive)
    def sensitize(self):
        debug.mainthreadTest()
        meshok = self.meshobj is not None
        if meshok:
            ok = self.chooser.get_value() is not None
        else:
            ok = False
        self.newbutton.set_sensitive(meshok)
        self.copybutton.set_sensitive(ok)
        self.editbutton.set_sensitive(ok)
        self.renamebutton.set_sensitive(ok)
        self.deletebutton.set_sensitive(ok)

    def currentMeshName(self):
        return self.meshwidget.get_value()

    def chooserCB(self, *args):
        self.widgetChanged(self.chooser.nChoices()>0, interactive=True)

    def newCB(self, *args):             # gtk callback
        menuitem = meshmenu.csmenu.New
        if parameterwidgets.getParameters(menuitem.get_arg("name"),
                                          menuitem.get_arg("cross_section"),
                                          title="New cross section",
                                          scope=self.scope,
                                          parentwindow=self.gtk.get_toplevel()):
            menuitem.callWithDefaults(mesh=self.meshwidget.get_value())

    def deleteCB(self, *args):          # gtk callback
        menuitem = meshmenu.csmenu.Remove
        menuitem.callWithDefaults(mesh=self.meshwidget.get_value(),
                                  name=self.chooser.get_value())

    def renameCB(self, *args):          # gtk callback
        menuitem = meshmenu.csmenu.Rename
        oldname = self.chooser.get_value()
        namearg = menuitem.get_arg('name')
        namearg.value = oldname
        if parameterwidgets.getParameters(
                namearg,
                title='Rename cross section "%s"' % oldname,
                parentwindow=self.gtk.get_toplevel()):
            menuitem.callWithDefaults(mesh=self.meshwidget.get_value(),
                                      cross_section=oldname)
    def renamedCS(self, oldname, newname): # sb ("cross section renamed", mesh)
        self.chooser.set_state(newname)

    def editCB(self, *args):            # gtk callback
        menuitem = meshmenu.csmenu.Edit
        csname = self.chooser.get_value()
        csparam = menuitem.get_arg('cross_section')
        csparam.value = self.meshobj.getCrossSection(csname)
        if parameterwidgets.getParameters(
                menuitem.get_arg('cross_section'),
                title="Edit cross section " + csname,
                parentwindow=self.gtk.get_toplevel()):
            menuitem.callWithDefaults(mesh=self.meshobj.path(), name=csname)
    def copyCB(self, *args):
        menuitem = meshmenu.csmenu.Copy
        csname = self.chooser.get_value()
        if parameterwidgets.getParameters(
                menuitem.get_arg('mesh'),
                menuitem.get_arg('name'),
                title="Copy cross section " + csname,
                parentwindow=self.gtk.get_toplevel()):
            menuitem.callWithDefaults(current=self.meshobj.path(),
                                      cross_section=csname)
            
    def set_value(self, value):
        self.chooser.set_state(value)
        self.widgetChanged(bool(value), interactive=False)

    def get_value(self):
        return self.chooser.get_value()

    def cleanUp(self):
        self.chooser.destroy()
        for s in self.sbcbs:
            switchboard.removeCallback(s)
        parameterwidgets.ParameterWidget.cleanUp(self)
        

    
def _make_MCSPWidget(self, scope, **kwargs):
    return MeshCrossSectionParamWidget(self, scope, name=self.name, **kwargs)

meshcsparams.MeshCrossSectionParameter.makeWidget = _make_MCSPWidget
