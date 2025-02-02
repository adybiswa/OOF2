# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.common import switchboard
from ooflib.SWIG.engine import masterelement
from ooflib.SWIG.engine import meshdatacache
from ooflib.common import debug
from ooflib.common import labeltree
from ooflib.common import microstructure
from ooflib.common.IO import mainmenu
from ooflib.common.IO import parameter
from ooflib.common.IO import reporter
from ooflib.common.IO.GUI import chooser
from ooflib.common.IO.GUI import gtklogger
from ooflib.common.IO.GUI import gtkutils
from ooflib.common.IO.GUI import historian
from ooflib.common.IO.GUI import oofGUI
from ooflib.common.IO.GUI import parameterwidgets
from ooflib.common.IO.GUI import regclassfactory
from ooflib.common.IO.GUI import whowidget
from ooflib.engine import meshmod
from ooflib.engine import meshstatus
from ooflib.engine import skeletoncontext
import ooflib.engine.mesh

from ooflib.common.runtimeflags import digits 

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

meshmenu = mainmenu.OOF.Mesh

## TODO: Check that the time is updated correctly in the Mesh info
## pane after solving or using the time slider in the gfx window.  The
## time in the mesh page should always agree with the current time on
## the solver page, and be independent of the time on the gfx window.

class MeshPage(oofGUI.MainPage):
    def __init__(self):
        self.built = False
        oofGUI.MainPage.__init__(
            self, name="FE Mesh", ordering=200,
            tip="Create a Finite Element Mesh from a Skeleton.")
        mainbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.gtk.add(mainbox)

        centerbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3,
                            halign=Gtk.Align.CENTER, margin_top=2)
        mainbox.pack_start(centerbox, expand=False, fill=False, padding=0)
        self.meshwidget = whowidget.WhoWidget(ooflib.engine.mesh.meshes,
                                              scope=self)
        label = Gtk.Label(label="Microstructure=", halign=Gtk.Align.END)
        centerbox.pack_start(label, expand=False, fill=False, padding=0)
        centerbox.pack_start(self.meshwidget.gtk[0],
                             expand=False, fill=False, padding=0)

        label = Gtk.Label(label="Skeleton=",
                          halign=Gtk.Align.END, margin_start=5)
        centerbox.pack_start(label, expand=False, fill=False, padding=0)
        centerbox.pack_start(self.meshwidget.gtk[1],
                             expand=False, fill=False, padding=0)

        label = Gtk.Label(label="Mesh=", halign=Gtk.Align.END, margin_start=5)
        centerbox.pack_start(label, expand=False, fill=False, padding=0)
        centerbox.pack_start(self.meshwidget.gtk[2],
                             expand=False, fill=False, padding=0)

        # Centered box of buttons
        bbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,
                       halign=Gtk.Align.CENTER, homogeneous=False, spacing=3)
        mainbox.pack_start(bbox, expand=False, fill=False, padding=0)
        
        self.newbutton = gtkutils.StockButton('document-new-symbolic', "New...")
        gtklogger.setWidgetName(self.newbutton, 'New')
        gtklogger.connect(self.newbutton, 'clicked', self.newCB)
        self.newbutton.set_tooltip_text(
            "Create a new mesh from the current skeleton.")
        bbox.pack_start(self.newbutton, expand=False, fill=True, padding=0)
        
        self.renamebutton = gtkutils.StockButton('document-edit-symbolic',
                                                 "Rename...")
        gtklogger.setWidgetName(self.renamebutton, 'Rename')
        gtklogger.connect(self.renamebutton, 'clicked', self.renameCB)
        self.renamebutton.set_tooltip_text("Rename the current mesh.")   
        bbox.pack_start(self.renamebutton, expand=False, fill=True, padding=0)
        
        self.copybutton = gtkutils.StockButton('edit-copy-symbolic', "Copy...")
        gtklogger.setWidgetName(self.copybutton, 'Copy')
        gtklogger.connect(self.copybutton, 'clicked', self.copyCB)
        self.copybutton.set_tooltip_text("Copy the current mesh.")
        bbox.pack_start(self.copybutton, expand=False, fill=True, padding=0)
        
        self.deletebutton = gtkutils.StockButton('edit-delete-symbolic',
                                                 "Delete")
        gtklogger.setWidgetName(self.deletebutton, 'Delete')
        gtklogger.connect(self.deletebutton, 'clicked', self.deleteCB)
        self.deletebutton.set_tooltip_text("Delete the current mesh.")
        bbox.pack_start(self.deletebutton, expand=False, fill=True, padding=0)
        
        self.savebutton = gtkutils.StockButton('document-save-symbolic',
                                               "Save...")
        gtklogger.setWidgetName(self.savebutton, 'Save')
        gtklogger.connect(self.savebutton, 'clicked', self.saveCB)
        self.savebutton.set_tooltip_text("Save the current mesh to a file.")
        bbox.pack_start(self.savebutton, expand=False, fill=True, padding=0)

        mainpane = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL,
                              wide_handle=True)
        gtklogger.setWidgetName(mainpane, 'Pane')
        mainbox.pack_start(mainpane, expand=True, fill=True, padding=0)
        gtklogger.connect_passive(mainpane, 'notify::position')
        leftpane = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL,
                             wide_handle=True)
        gtklogger.setWidgetName(leftpane, 'leftpane')
        mainpane.pack1(leftpane, resize=True, shrink=False)
        gtklogger.connect_passive(leftpane, 'notify::position')

        infoframe = Gtk.Frame(
            label='Mesh Information',
            shadow_type=Gtk.ShadowType.IN,
            margin_top=2, margin_bottom=gtkutils.handle_padding,
            margin_start=2, margin_end=gtkutils.handle_padding)
        leftpane.pack1(infoframe, resize=True, shrink=False)
        scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN, margin=2)
        gtklogger.logScrollBars(scroll, "MeshInfo")
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        infoframe.add(scroll)
        self.infoarea = Gtk.TextView(name="fixedfont", cursor_visible=False,
                                     editable=False,
                                     left_margin=5, right_margin=5,
                                     top_margin=5, bottom_margin=5)
        scroll.add(self.infoarea)

        # Subproblem creation, deletion, etc.
        subprobframe = Gtk.Frame(
            label='Subproblems',
            shadow_type=Gtk.ShadowType.IN,
            margin_top=gtkutils.handle_padding, margin_bottom=2,
            margin_start=2, margin_end=gtkutils.handle_padding)

        gtklogger.setWidgetName(subprobframe, 'Subproblems')
        leftpane.pack2(subprobframe, resize=True, shrink=False)
        subpbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, margin=2,
                          spacing=2)
        subprobframe.add(subpbox)
        self.subpchooser = chooser.ScrolledChooserListWidget(
            callback=self.subpchooserCB,
            dbcallback=self.subprobEditCB,
            name="subprobChooser")
        subpbox.pack_start(self.subpchooser.gtk,
                           expand=True, fill=True, padding=0)

        # Grid containing buttons for operating on subproblems.
        subpbuttons = Gtk.Grid(column_homogeneous=True,
                               row_homogeneous=True,
                               row_spacing=2, column_spacing=2)
        subpbox.pack_start(subpbuttons, expand=False, fill=False, padding=0)

        self.subprobNew = gtkutils.StockButton('document-new-symbolic',
                                               "New...", hexpand=True)
        gtklogger.setWidgetName(self.subprobNew, "New")
        gtklogger.connect(self.subprobNew, "clicked", self.subprobNewCB)
        self.subprobNew.set_tooltip_text("Create a new subproblem.")
        subpbuttons.attach(self.subprobNew, 0,0, 1,1)

        self.subprobRename = Gtk.Button(label="Rename...", hexpand=True)
        gtklogger.setWidgetName(self.subprobRename, "Rename")
        gtklogger.connect(self.subprobRename, "clicked", self.subprobRenameCB)
        self.subprobRename.set_tooltip_text("Rename the selected subproblem")
        subpbuttons.attach(self.subprobRename, 1,0, 1,1)

        self.subprobEdit = gtkutils.StockButton('document-edit-symbolic',
                                                "Edit...", hexpand=True)
        gtklogger.setWidgetName(self.subprobEdit, "Edit")
        gtklogger.connect(self.subprobEdit, 'clicked', self.subprobEditCB)
        self.subprobEdit.set_tooltip_text("Edit the selected subproblem.")
        subpbuttons.attach(self.subprobEdit, 2,0, 1,1)

        self.subprobCopy = gtkutils.StockButton('edit-copy-symbolic', "Copy...",
                                                hexpand=True)
        gtklogger.setWidgetName(self.subprobCopy, "Copy")
        gtklogger.connect(self.subprobCopy, "clicked", self.subprobCopyCB)
        self.subprobCopy.set_tooltip_text("Copy the selected subproblem.")
        subpbuttons.attach(self.subprobCopy, 0,1, 1,1)

        self.subprobInfo = Gtk.Button(label="Info", hexpand=True)
        gtklogger.setWidgetName(self.subprobInfo, "Info")
        gtklogger.connect(self.subprobInfo, 'clicked', self.subprobInfoCB)
        self.subprobInfo.set_tooltip_text(
            "Print information about the selected subproblem")
        subpbuttons.attach(self.subprobInfo, 1,1, 1,1)
        
        self.subprobDelete = gtkutils.StockButton('edit-delete-symbolic',
                                                  "Delete", hexpand=True)
        gtklogger.setWidgetName(self.subprobDelete, "Delete")
        gtklogger.connect(self.subprobDelete, "clicked", self.subprobDeleteCB)
        self.subprobDelete.set_tooltip_text("Delete the selected subproblem.")
        subpbuttons.attach(self.subprobDelete, 2,1, 1,1)
        
        # Right hand side for element operations
        
        elementopsframe = Gtk.Frame(
            label="Mesh Operations", shadow_type=Gtk.ShadowType.IN,
            margin_start=gtkutils.handle_padding, margin_end=2,
            margin_top=2, margin_bottom=2)
        gtklogger.setWidgetName(elementopsframe, 'ElementOps')
        mainpane.pack2(elementopsframe, resize=False, shrink=False)
        elementopsbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        elementopsframe.add(elementopsbox)
        self.elementops = regclassfactory.RegisteredClassFactory(
            meshmod.MeshModification.registry,
            title="Method:",
            callback=self.elementopsCB,
            shadow_type=Gtk.ShadowType.NONE,
            scope=self, name="Method")
        elementopsbox.pack_start(self.elementops.gtk,
                                 expand=True, fill=True, padding=0)

        self.historian = historian.Historian(self.elementops.set,
                                             self.sensitizeHistory,
                                             setCBkwargs={'interactive':True})
        # Prev, OK, Next
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2,
                       margin_start=2, margin_end=2, margin_bottom=2)
        elementopsbox.pack_start(hbox, expand=False, fill=False, padding=0)
        self.prevbutton = gtkutils.prevButton()
        gtklogger.connect(self.prevbutton, 'clicked', self.prevCB)
        self.prevbutton.set_tooltip_text(
            "Recall the previous mesh element operation.")
        hbox.pack_start(self.prevbutton, expand=False, fill=False, padding=0)

        self.okbutton = gtkutils.StockButton('gtk-ok', 'OK')
        gtklogger.setWidgetName(self.okbutton, 'OK')
        gtklogger.connect(self.okbutton, 'clicked', self.okCB)
        self.okbutton.set_tooltip_text(
            'Perform the mesh operation defined above.')
        hbox.pack_start(self.okbutton, expand=True, fill=True, padding=0)

        self.nextbutton = gtkutils.nextButton()
        gtklogger.connect(self.nextbutton, 'clicked', self.nextCB)
        self.nextbutton.set_tooltip_text(
            'Recall the next mesh element operation.')
        hbox.pack_start(self.nextbutton, expand=False, fill=False, padding=0)

        self.built = True

        switchboard.requestCallbackMain("Mesh modified",
                                        self.recordModifier)
        switchboard.requestCallbackMain("mesh changed", self.meshchangeCB)
        switchboard.requestCallbackMain(("new who", "Microstructure"),
                                        self.newMSorSkeleton)
        switchboard.requestCallbackMain(("new who", "Skeleton"),
                                        self.newMSorSkeleton)
        switchboard.requestCallbackMain(("new who", "Mesh"),
                                        self.newMesh)
        switchboard.requestCallbackMain(("new who", "SubProblem"),
                                        self.newSubProblem)
        switchboard.requestCallbackMain(("rename who", "SubProblem"),
                                        self.renamedSubProblem)
        switchboard.requestCallbackMain(("remove who", "SubProblem"),
                                         self.removeSubProblem)
        switchboard.requestCallbackMain(self.meshwidget, self.meshwidgetCB)
        switchboard.requestCallbackMain("equation activated",
                                        self.equationCB)
        switchboard.requestCallbackMain("mesh status changed",
                                        self.statusChanged)
#         switchboard.requestCallbackMain("mesh boundaries changed",
#                                         self.newMeshBoundaries)

        switchboard.requestCallbackMain(('validity', self.elementops),
                                        self.validityChangeCB)

    def installed(self):
        self.sensitize()
        self.sensitizeHistory()
        self.update()

##     This doesn't seem to be necessary...
#     def newMeshBoundaries(self, mesh):
#         if mesh==self.currentMesh():
#             self.update()
#             self.sensitize()
            
    #######################

    def currentSkeletonFullName(self):
        return self.meshwidget.get_value(depth=2)
    def currentSkeletonContext(self):
        try:
            return skeletoncontext.skeletonContexts[
                self.currentSkeletonFullName()]
        except KeyError:
            return None

    def currentFullMeshName(self):
        return self.meshwidget.get_value()
    def currentMeshName(self):
        path = labeltree.makePath(self.currentFullMeshName())
        if path:
            return path[2]
    def currentMeshContext(self):
        try:
            return ooflib.engine.mesh.meshes[self.currentFullMeshName()]
        except KeyError:
            return None
    def currentMesh(self):
        ctxt = self.currentMeshContext()
        if ctxt:
            return ctxt.getObject()

    def currentSubProblemName(self):
        return self.subpchooser.get_value()
    def currentSubProblemContext(self):
        meshctxt = self.currentMeshContext()
        if meshctxt is not None:
            try:
                return meshctxt.get_subproblem(self.currentSubProblemName())
            except KeyError:
                return None
    def currentFullSubProblemName(self):
        ctxt = self.currentSubProblemContext()
        if ctxt:
            return ctxt.path()
            
    def sensitize(self):
        debug.mainthreadTest()
        skelok = self.currentSkeletonContext() is not None
        meshctxt = self.currentMeshContext()
        meshok = meshctxt is not None
        meshsync = meshok and not isinstance(meshctxt.status,
                                             meshstatus.OutOfSync)
        self.newbutton.set_sensitive(skelok)
        self.deletebutton.set_sensitive(meshok)
        self.renamebutton.set_sensitive(meshok)
        self.copybutton.set_sensitive(meshok and meshsync)
        self.savebutton.set_sensitive(meshok and meshsync)
        self.okbutton.set_sensitive(meshok and self.elementops.isValid())
        ## Anything that changes the list of subproblems will call
        ## sensitizeSubProblems via the Chooser callback, so it's not
        ## necessary to call it here.
        # self.sensitizeSubProblems()
        gtklogger.checkpoint("mesh page sensitized")

    def sensitizeHistory(self):
        debug.mainthreadTest()
        self.nextbutton.set_sensitive(self.historian.nextSensitive())
        self.prevbutton.set_sensitive(self.historian.prevSensitive())
        
    def update(self):
        self.set_state(self.currentFullMeshName())

    def recordModifier(self, path, modifier):  # callback for "Mesh modified"
        if modifier:
            self.historian.record(modifier)
        # might as well do the update for Mesh Information pane.
        self.set_state(path)

    def meshchangeCB(self, meshctxt): # switchboard "mesh changed"
        self.update_info()

    def update_info(self):
        themesh = self.currentMeshContext()
        textlines = []
        if themesh is not None:
            skel = themesh.getSkeleton()
            textlines.append(f"Status: {themesh.status.tag}")
            if themesh.outOfSync():
                textlines.append("*** Mesh must be rebuilt! ***")
            textlines.append(f"No. of Nodes:\t{themesh.nnodes()}")
            #Interface branch
            nel = themesh.nelements() + themesh.nedgements()
            textlines.append(f"No. of Elements:\t{nel}")
            masterelementenums = masterelement.getMasterElementEnums()
            corners = sorted(list(masterelementenums.keys())) # list of element geometries
            counts = [0]*(max(corners)+1)
            for elem in skel.element_iterator():
                counts[elem.nnodes()] += 1
            counts[2]=themesh.nedgements() #Interface branch
            for ncorners in corners:
                en = themesh.getMasterElementType(ncorners)
                ec = counts[ncorners]
                textlines.append(f"{ncorners} cornered element:\t {en} ({ec})")
            textlines.append(f"Time:\t{themesh.getCurrentTime():.{digits()}g}")
            textlines.append(
                "Data Cache Type: %s" %
                meshdatacache.getMeshDataCacheType(themesh.datacache))
            n = themesh.datacache.size()
            textlines.append(f"Data Cache Size: {n} time step{'s'*(n!=1)}") 
        else:                           # no current mesh
            textlines.append("No mesh!")
        buffer = self.infoarea.get_buffer()
        buffer.set_text('\n'.join(textlines))

    def statusChanged(self, meshctxt): # switchboard "mesh status changed"
        if meshctxt is self.currentMeshContext():
            self.update_info()
            self.sensitize()
            
    def set_state(self, meshpath):  # widget update & information update
        debug.mainthreadTest()
        path = labeltree.makePath(meshpath)
        self.meshwidget.set_value(path)
        self.update_info()
        self.set_subproblem_state()
        self.sensitize()
        self.sensitizeHistory()

    def newMSorSkeleton(self, path):
        # switchboard ("new who", "Microstructure") or ("new who", "Skeleton")
        if not self.currentMesh():
            self.meshwidget.set_value(path)

    def newMesh(self, meshname):        # switchboard ("new who", "Mesh")
        self.set_state(meshname)
        self.sensitize()
    
    def meshwidgetCB(self, interactive): # switchboard widget callback
        self.update()

    def equationCB(self, *args):  # switchboard "equation activated"
        switchboard.notify(self.meshwidget, interactive=1)

    def newCB(self, *args):             # gtk button callback
        menuitem = mainmenu.OOF.Mesh.New
        params = [x for x in menuitem.params if x.name !='skeleton']
        if parameterwidgets.getParameters(title='Create a new mesh',
                                          parentwindow=self.gtk.get_toplevel(),
                                          scope=self, *params):
            menuitem.callWithDefaults(skeleton=self.currentSkeletonFullName())

    def deleteCB(self, *args):          # gtk button callback
        if reporter.query(
                f"Really delete {self.currentFullMeshName()}?",
                "No", default="Yes",
                parentwindow=self.gtk.get_toplevel()) == "Yes":
            meshmenu.Delete(mesh=self.currentFullMeshName())

##    def removeMesh(self, path):    # switchboard ("remove who", "Mesh")
##        self.update()
##        self.sensitize()

    def copyCB(self, *args):            # gtk button callback
        menuitem = mainmenu.OOF.Mesh.Copy
        nameparam = menuitem.get_arg("name")
        fieldparam = menuitem.get_arg("copy_field")
        eqnparam = menuitem.get_arg("copy_equation")
        bcparam = menuitem.get_arg("copy_bc")
        if parameterwidgets.getParameters(nameparam, fieldparam, eqnparam,
                                          bcparam,
                                          parentwindow=self.gtk.get_toplevel(),
                                          title='Copy a mesh'):
            menuitem.callWithDefaults(mesh=self.currentFullMeshName())
        
    def renameCB(self, *args):          # gtk button callback
        menuitem = meshmenu.Rename
        namearg = menuitem.get_arg('name')
        curmeshpath = self.currentFullMeshName()
        namearg.value = labeltree.makePath(curmeshpath)[-1]
        if parameterwidgets.getParameters(
                namearg,
                parentwindow=self.gtk.get_toplevel(),
                title=f'Rename mesh "{namearg.value}"'):
            menuitem.callWithDefaults(mesh=curmeshpath)

    def saveCB(self, *args):
        menuitem = mainmenu.OOF.File.Save.Mesh
        meshname = self.meshwidget.get_value()
        params = [x for x in menuitem.params if x.name!="mesh"]
        if parameterwidgets.getParameters(ident='SaveMeshFromPage',
                                          parentwindow=self.gtk.get_toplevel(),
                                          title=f'Save Mesh "{meshname}"?',
                                          *params):
            menuitem.callWithDefaults(mesh=meshname)

    def prevCB(self, gtkobj):
        self.historian.prevCB()

    def nextCB(self, gtkobj):
        self.historian.nextCB()

    def elementopsCB(self, reg):
        self.historian.stateChangeCB(reg)
        self.sensitize()

    def okCB(self, gtkobj):
        path = self.meshwidget.get_value()
        modifier = self.elementops.get_value()
        if path and modifier:
            mainmenu.OOF.Mesh.Modify(mesh=path, modifier=modifier)

    def validityChangeCB(self, validity):
        self.sensitize()

    ######################

    # subproblem callbacks and sensitization

    def set_subproblem_state(self, subprobname=None):
        meshctxt = self.currentMeshContext()
        if meshctxt is not None:
            self.subpchooser.update(meshctxt.subproblemNames())
            if subprobname is not None:
                self.subpchooser.set_selection(subprobname)
        else:
            self.subpchooser.update([])
        self.sensitizeSubProblems()

    def subprobNewCB(self, gtkobj):
        menuitem = mainmenu.OOF.Subproblem.New
        params = [x for x in menuitem.params if x.name != 'mesh']
        if parameterwidgets.getParameters(title='Create a new subproblem',
                                          parentwindow=self.gtk.get_toplevel(),
                                          scope=self, *params):
            menuitem.callWithDefaults(mesh=self.currentFullMeshName())
    def subprobCopyCB(self, gtkobj):
        menuitem = mainmenu.OOF.Subproblem.Copy
        # Initialize the 'mesh' parameter to the current mesh, but
        # allow the user to change it.  Usually the subproblem will be
        # copied to the current mesh, but not always.
        meshparam = menuitem.get_arg('mesh')
        meshparam.value = self.currentFullMeshName()
        params = [x for x in menuitem.params if x.name != 'subproblem']
        if parameterwidgets.getParameters(title='Copy a subproblem',
                                          parentwindow=self.gtk.get_toplevel(),
                                          scope=self, *params):
            menuitem.callWithDefaults(
                subproblem=self.currentFullSubProblemName())
    def subprobRenameCB(self, gtkobj):
        menuitem = mainmenu.OOF.Subproblem.Rename
        namearg = menuitem.get_arg('name')
        cursubprob = self.currentFullSubProblemName()
        namearg.value = labeltree.makePath(cursubprob)[-1]
        if parameterwidgets.getParameters(
                namearg,
                parentwindow=self.gtk.get_toplevel(),
                title=f'Rename subproblem "{namearg.value}"'):
            menuitem.callWithDefaults(subproblem=cursubprob)

    def subprobInfoCB(self, gtkobj):
        mainmenu.OOF.Subproblem.Info(
            subproblem=self.currentFullSubProblemName())

    def subprobDeleteCB(self, gtkobj):
        if reporter.query(
                f'Really delete "{self.currentFullSubProblemName()}"',
                "No", default="Yes",
                parentwindow=self.gtk.get_toplevel()) == "Yes":
            mainmenu.OOF.Subproblem.Delete(
                subproblem=self.currentFullSubProblemName())

    def subprobEditCB(self, gtkobj):
        subprobname = self.currentFullSubProblemName()
        if subprobname:
            menuitem = mainmenu.OOF.Subproblem.Edit
            subpctxt = ooflib.engine.subproblemcontext.subproblems[subprobname]
            subpparam = menuitem.get_arg('subproblem')
            subpparam.set(subpctxt.subptype)
            if parameterwidgets.getParameters(
                    subpparam,
                    parentwindow=self.gtk.get_toplevel(),
                    title="Edit Subproblem definition", scope=self):
                menuitem.callWithDefaults(name=subprobname)
        

    def subpchooserCB(self, subp, interactive):
        if self.built:
            self.sensitizeSubProblems()

    def sensitizeSubProblems(self):
        debug.mainthreadTest()
        subpok = self.subpchooser.get_value() is not None
        defaultsubp = (self.subpchooser.get_value() ==
                       ooflib.engine.mesh.defaultSubProblemName)
        meshctxt = self.currentMeshContext()
        meshok = meshctxt is not None and not meshctxt.outOfSync()
        self.subprobNew.set_sensitive(meshok)
        self.subprobRename.set_sensitive(subpok and not defaultsubp)
        self.subprobCopy.set_sensitive(subpok)
        self.subprobDelete.set_sensitive(subpok and not defaultsubp)
        self.subprobEdit.set_sensitive(meshok and subpok and not defaultsubp)
        self.subprobInfo.set_sensitive(subpok)
        gtklogger.checkpoint("mesh page subproblems sensitized")

    def newSubProblem(self, subproblempath): # sb ("new who", "SubProblem")
        path = labeltree.makePath(subproblempath)
        if labeltree.makePath(self.currentFullMeshName()) == path[:-1]:
            self.set_subproblem_state(path[-1])
    def renamedSubProblem(self, oldpath, newname):
        # switchboard ("rename who", "SubProblem")
        path = labeltree.makePath(oldpath)
        if labeltree.makePath(self.currentFullMeshName()) == path[:-1]:
            self.set_subproblem_state(newname)
    def removeSubProblem(self, path):
        if labeltree.makePath(self.currentFullMeshName()) == path[:-1]:
            self.set_subproblem_state()
        

#############
        
mp = MeshPage()
