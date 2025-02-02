# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# Simple display methods for Skeletons and Meshes.  Simpler than
# contour plots, in any case.

import sys

from ooflib.SWIG.common import config
from ooflib.SWIG.common import lock
from ooflib.SWIG.common import smallmatrix
from ooflib.SWIG.common import switchboard
from ooflib.SWIG.common import threadstate
from ooflib.SWIG.engine import mastercoord
from ooflib.SWIG.engine import ooferror
from ooflib.common import color
from ooflib.common import debug
from ooflib.common import mainthread
from ooflib.common import parallel_enable
from ooflib.common import primitives
from ooflib.common import registeredclass
from ooflib.common import utils
from ooflib.common.IO import automatic
from ooflib.common.IO import colormap
from ooflib.common.IO import display
from ooflib.common.IO import ghostgfxwindow
from ooflib.common.IO import mainmenu
from ooflib.common.IO import oofmenu
from ooflib.common.IO import parameter
from ooflib.common.IO import reporter
from ooflib.common.IO import placeholder
from ooflib.common.IO import xmlmenudump
from ooflib.engine import mesh
from ooflib.engine import skeletoncontext
from ooflib.engine import skeletonmodifier
from ooflib.engine.IO import meshparameters
from ooflib.engine.IO import output
from ooflib.engine.IO import outputDefs
import ooflib.SWIG.engine.material
#Interface branch
from ooflib.common.IO import placeholder
from ooflib.engine.IO import meshparameters
from ooflib.engine.IO import materialparameter

import oofcanvas

if parallel_enable.enabled():
    from ooflib.SWIG.common import mpitools

FloatRangeParameter = parameter.FloatRangeParameter
IntRangeParameter = parameter.IntRangeParameter
AutoNumericParameter = parameter.AutoNumericParameter

##################

# Add a parameter to the graphics window's Settings menu, to control
# whether or not Mesh Elements with no Material are displayed.

def toggleEmptyDrawing(menuitem, hideEmpty):
    gfxwindow = menuitem.data
    gfxwindow.settings.hideEmptyElements = hideEmpty
    gfxwindow.draw()

def addToGfxSettings(gfxwindow):
    item = gfxwindow.menu.Settings.addItem(oofmenu.CheckOOFMenuItem(
        'Hide_Empty_Mesh_Elements',
        callback=toggleEmptyDrawing,
        value=gfxwindow.settings.hideEmptyElements,
        help="Toggle the display of elements with no Material.",
        discussion=xmlmenudump.loadFile(
            'DISCUSSIONS/engine/menu/hideempty.xml'),
        xrefs=["Chapter-Graphics"]))
    item.data = gfxwindow

switchboard.requestCallback('open graphics window', addToGfxSettings)
ghostgfxwindow.defineGfxSetting('hideEmptyElements', True)

##################

# Skeleton and Mesh display methods are very similar, except for how
# they get some of their data.  These two base classes encapsulate the
# *differences* between Skeletons and Meshes, as far as displaying is
# concerned.  They're mixed in with other base classes to create the
# actual DisplayMethods.

class SkeletonDisplayMethod(display.DisplayMethod):
    def __init__(self):
        display.DisplayMethod.__init__(self)
    def polygons(self, gfxwindow, skelcontext):
        skeleton = skelcontext.getObject()
        if parallel_enable.enabled():
            nodes = skeleton.all_skeletons["nodes"]
            elements = skeleton.all_skeletons["elements"]
            for i in range(mpitools.Size()):
                for el in elements[i]:
                    yield [primitives.Point(*nodes[i][ni]) for ni in el]
        else:
            for el in skeleton.element_iterator():
                yield el.perimeter()

# Dummy exception class, raised by
# _undisplaced_from_displaced_with_element if it overruns its
# iteration max.  This is expected to indicate failure due to range
# problems.  Caught and handled in the undisplaced_from_displaced
# function, which is the raising function's only caller.
class IterationMaxExceeded:
    pass
    
class MeshDisplayMethod(display.AnimationLayer, display.DisplayMethod):
    # A display method that displays data from a Mesh at positions
    # determined by the mesh.  self.where is a PositionOutput.
    def __init__(self, when):
        self.freezetime = None
        display.AnimationLayer.__init__(self, when)
        display.DisplayMethod.__init__(self)
    def incomputable(self, gfxwindow):
        if self.who is None:
            return True
        themesh = self.who.resolve(gfxwindow)
        return (display.DisplayMethod.incomputable(self, gfxwindow) or
                not themesh.boundedTime(self.when) or 
                self.where.incomputable(themesh))
    def clone(self, layerset=None):
        bozo = display.DisplayMethod.clone(self, layerset)
        bozo.freezetime = self.freezetime
        return bozo
    def freeze(self, gfxwindow):
        meshctxt = self.who.resolve(gfxwindow)
        if meshctxt:
            self.freezetime = meshctxt.getTime(self.when)
        display.DisplayMethod.freeze(self, gfxwindow)
    def unfreeze(self, gfxwindow):
        self.freezetime = None
        display.DisplayMethod.unfreeze(self, gfxwindow)
    def refreeze(self, layer):
        if isinstance(layer, MeshDisplayMethod):
            self.freezetime = layer.freezetime
        else:
            self.freezetime = None
        display.DisplayMethod.refreeze(self, layer)
    def getTime(self, meshctxt, gfxwindow):
        if self.freezetime is not None:
            return self.freezetime
        if self.when == placeholder.latest:
            return gfxwindow.displayTime
        return meshctxt.getTime(self.when)

    def animationTimes(self, gfxwindow):
        meshctxt = self.who.resolve(gfxwindow)
        return meshctxt.cachedTimes()
        
    def polygons(self, gfxwindow, meshctxt):
        themesh = meshctxt.getObject()
        meshctxt.restoreCachedData(self.getTime(meshctxt, gfxwindow))
        try:
            # PARALLEL_RCL: Make changes here to display parallel mesh
            # There is an issue with clicking on the skeleton or mesh
            # graphics: only the nodes or elements for the front-end
            # process get the cursor or mark

            if parallel_enable.enabled():
                # This snippet taken from SkeletonDisplayMethod
                nodes = themesh.all_meshskeletons["nodes"]
                elements = themesh.all_meshskeletons["elements"]
                polys = []
                for i in range(mpitools.Size()):
                    for el in elements[i]:
                        polys.append([primitives.Point(*nodes[i][ni])
                                      for ni in el])
                return polys
            else:
                if gfxwindow.settings.hideEmptyElements:
                    edges = [element.perimeter()
                             for element in themesh.elements()
                             if element.material() is not None]
                else:
                    # edges is a list of lists of Edges
                    edges = [element.perimeter()
                             for element in themesh.elements()]

                ## TODO PYTHON3 LATER: Can this all be done with generators
                ## instead of lists?  Maybe if all edges were
                ## evaluated at the same location, so that we wouldn't
                ## have to iterate over the edges to know how big to
                ## make the corners list?  Also see TODO in output.py.
                
                flatedges = utils.flatten(edges)
                # corners tells where on each edge to evaluate self.where
                corners = [[0.0]]*len(flatedges)
                # evaluate position output for all edges at once
                polys = self.where.evaluate(themesh, flatedges, corners)
                # give the corner list the same structure as the edge list: a
                # list of lists, where each sublist is the list of corners of
                # an element.
                polys = utils.unflatten(edges, polys)
                if len(polys) == 0:
                    mainthread.runBlock(
                        reporter.warn,
                        ("No mesh elements drawn! Are there no materials assigned?",))
                return polys
        finally:
            meshctxt.releaseCachedData()

    # Routines for converting between displaced and undisplaced
    # coordinates using this layer's PositionOutput object
    # ("self.where").  This is here primarily so that the mesh info
    # displays can be drawn in displaced coordinates -- since this
    # method refers to the topmost mesh display, only that display
    # (i.e. this object) knows the right PositionOutput to use.
    def displaced_from_undisplaced(self, gfxwindow, orig):
        meshctxt = self.who.resolve(gfxwindow)
        femesh = meshctxt.getObject()
        felem = meshctxt.enclosingElement(orig)
        return self._displaced_from_undisplaced_with_element(
            femesh, felem, orig)

    def _displaced_from_undisplaced_with_element(self, mesh, elem, orig):
        masterpos = elem.to_master(orig)
        res = self.where.evaluate(mesh, [elem], [[masterpos]])
        return res[0]

    
    def _undisplaced_from_displaced_with_element(self, mesh, elem, pos):
        # Search master space for an x such that f(x)=pos, and return x.

        delta = 0.001 # Small compared to master space.
        mtx = smallmatrix.SmallMatrix(2,2)
        rhs = smallmatrix.SmallMatrix(2,1)
        delta_x = mastercoord.MasterCoord(delta, 0.0)
        delta_y = mastercoord.MasterCoord(0.0, delta)
        
        res = elem.center()

        done = False
        # Magic numbers.  The function can actually fail, because for
        # higher-order elements, the displacement mapping is
        # parabolic, and the range of the parabolic mapping might not
        # include the point pos, if it's too far away from the
        # element.  So, if we've gone for more than maxiters
        # iterations, raise an exception.
        tolerance = 1.0e-10
        maxiters = 50

        count = 0
        while not done:
            count += 1
            if count > maxiters:
                raise IterationMaxExceeded()

            fwd = self.where.evaluate(mesh, [elem],[[res]])[0]
            fwddx = self.where.evaluate(mesh, [elem],[[res+delta_x]])[0]
            fwddy = self.where.evaluate(mesh, [elem],[[res+delta_y]])[0]
            
            dfdx = (fwddx-fwd)/delta
            dfdy = (fwddy-fwd)/delta

            diff = pos-fwd

            rhs.setitem(0,0,diff[0])
            rhs.setitem(1,0,diff[1])
            
            mtx.setitem(0,0,dfdx[0])
            mtx.setitem(0,1,dfdy[0])
            mtx.setitem(1,0,dfdx[1])
            mtx.setitem(1,1,dfdy[1])

            ## TODO OPT: For a 2x2 matrix, is it faster to write out
            ## the solution, rather than using a general purpose
            ## routine?
            r = mtx.solve(rhs)

            resid = (rhs[0,0]**2+rhs[1,0]**2)
            res = mastercoord.MasterCoord(res[0]+rhs[0,0], res[1]+rhs[1,0])

            if resid<tolerance:
                done = True

        return res

    # TODO OPT: This whole process leans fairly hard on the
    # single-point, single-element version of self.where.evaluate() --
    # performance improvements there would be welcome.
    def undisplaced_from_displaced(self, gfxwindow, pos):
        meshctxt = self.who.resolve(gfxwindow)
        femesh = meshctxt.getObject()

        hideEmpty = gfxwindow.settings.hideEmptyElements
        # Searches over the list of elements, sorted in order from
        # nearest to farthest (as measured by the center points) from
        # the click position.  This is much less expensive than
        # running the Newton-Raphson iteration extra useless times.
        ## TODO OPT: This is slow for large meshes, and in particular,
        ## is very slow to fail.  Find a cleverer way to select which
        ## elements to search.
        ellist = []
        for el in femesh.elements():
            if not (el.material() is None and hideEmpty):
                distance2 = (self.where.evaluate(
                    femesh, [el],[[el.center()]])[0] - pos)**2
                ellist.append( (distance2, el) )
        ellist.sort(key=lambda x: x[0])

        smallestdist = None
        smallestres = None
        smallestel = None
        for dist2, el in ellist:
            if not (el.material() is None and hideEmpty):
                try:
                    res = self._undisplaced_from_displaced_with_element(
                        femesh, el, pos)
                except IterationMaxExceeded:
                    # Newton-Raphson failed, probable range error, go
                    # on to the next element.
                    pass
                else:
                    # We got a result.  Find how far it is outside the
                    # master element
                    dist = el.masterelement().outOfBounds(res)

                    # If it's inside, then we're done
                    if dist <= 0.0:
                        return el.from_master(res)
                    # If it's outside, it may just be due to round off.
                    # We have to wait until we've looked at all the
                    # elements to make sure, though.
                    if smallestdist is None or dist < smallestdist:
                        smallestdist = dist
                        smallestres = res
                        smallestel = el
        # If we got here, then the point isn't inside any element.  If
        # it's close enough to the nearest element, use that element.
        # The distance is measured in master element space, so the
        # definition of "close enough" is scale invariant.
        if smallestdist < 0.2:
            if smallestel:
                return smallestel.from_master(smallestres)

        raise ooferror.PyErrBoundsError("No element found")
        
###########################

## Default values of display parameters for SkeletonEdgeDisplay and
## MeshEdgeDisplay, and menu items to set them.

defaultSkeletonWidth = 0.5
defaultMeshWidth = 0.5
widthRange = (0, 10, 0.1)
defaultSkeletonColor = color.black
defaultMeshColor = color.black
defaultMeshPosition = outputDefs.actualPosition

def _setDefaultSkeletonEdgeParams(menuitem, color, width):
    global defaultSkeletonColor
    defaultSkeletonColor = color
    global defaultSkeletonWidth
    defaultSkeletonWidth = width
    
mainmenu.gfxdefaultsmenu.Skeletons.addItem(oofmenu.OOFMenuItem(
    'Skeleton_Edges',
    callback=_setDefaultSkeletonEdgeParams,
    ordering = 0,
    params=[color.TranslucentColorParameter('color', defaultSkeletonColor,
                                            tip=parameter.emptyTipString),
            FloatRangeParameter('width', widthRange, defaultSkeletonWidth,
                                tip="Line thickness, in pixels.")
            ],
    help="Set the default parameters for Skeleton edge displays.",
    discussion="""<para>

    Set the default parameters for
    <link linkend="RegisteredClass-SkeletonEdgeDisplay"><classname>SkeletonEdgeDisplays</classname></link>.
    See <xref linkend="RegisteredClass-SkeletonEdgeDisplay"/> for the details.
    This command may be placed in the &oof2rc; file
    to set a default value for all &oof2; sessions.

    </para>"""))

def _setDefaultMeshEdgeParams(menuitem, color, width):
    global defaultMeshColor
    defaultMeshColor = color
    global defaultMeshWidth
    defaultMeshWidth = width

mainmenu.gfxdefaultsmenu.Meshes.addItem(oofmenu.OOFMenuItem(
    'Mesh_Edges',
    callback=_setDefaultMeshEdgeParams,
    ordering=0,
    params=[color.TranslucentColorParameter('color', defaultMeshColor,
                                            tip=parameter.emptyTipString),
            FloatRangeParameter('width', widthRange, defaultMeshWidth,
                                tip="Line thickness, in pixels.")],
    help="Set the default parameters for Mesh edge displays.",
    discussion="""<para>

    Set the default parameters for
    <link linkend="RegisteredClass-MeshEdgeDisplay"><classname>MeshEdgeDisplays</classname></link>.
    See <xref linkend="RegisteredClass-MeshEdgeDisplay"/> for the details.
    This command may be placed in the &oof2rc; file to set a default value
    for all &oof2; sessions.

    </para>"""))

def _setDefaultMeshPosition(menuitem, where):
    global defaultMeshPosition
    defaultMeshPosition = where

mainmenu.gfxdefaultsmenu.Meshes.addItem(oofmenu.OOFMenuItem(
    'Mesh_Position',
    callback=_setDefaultMeshPosition,
    ordering=1.1,
    params=[output.PositionOutputParameter(
        'where', value=defaultMeshPosition,
        tip='Plot at displaced or original position?')],
    help="Set the default displacement in Mesh displays.",
    discussion=xmlmenudump.loadFile(
        "DISCUSSIONS/engine/menu/defaultmeshpos.xml")
    ))

##################

# Common mesh display parameters.

meshdispparams = [
    placeholder.TimeParameter(
        'when',
        value=placeholder.latest,
        tip='Time at which to plot'),
    output.PositionOutputParameter(
        'where', value=defaultMeshPosition,
        tip="Plot at displaced or original position?")
]


####

class EdgeDisplay:
    def draw(self, gfxwindow):
        themesh = self.who.resolve(gfxwindow)
        polygons = self.polygons(gfxwindow, themesh)
        clr = color.canvasColor(self.color)
        for polygon in polygons:
            poly = oofcanvas.CanvasPolygon.create()
            poly.setLineWidthInPixels(self.width)
            poly.setLineColor(clr)
            poly.setLineJoin(oofcanvas.lineJoinBevel)
            poly.addPoints(polygon)
            self.canvaslayer.addItem(poly)
        
class MeshEdgeDisplay(EdgeDisplay, MeshDisplayMethod):
    # EdgeDisplay draws the edges of the Elements
    def __init__(self, when, where,
                 width=defaultMeshWidth, color=defaultMeshColor):
        if where is not None:
            self.where = where.clone()
        else:
            self.where = None           # ?
        self.width = width
        self.color = color
        MeshDisplayMethod.__init__(self, when)
    def draw(self, gfxwindow):
        EdgeDisplay.draw(self, gfxwindow)
    def getTimeStamp(self, gfxwindow):
        return max(
            MeshDisplayMethod.getTimeStamp(self, gfxwindow),
            gfxwindow.settings.getTimeStamp('hideEmptyElements'))

class SkeletonEdgeDisplay(EdgeDisplay, SkeletonDisplayMethod):
    def __init__(self, width=defaultSkeletonWidth, color=defaultSkeletonColor):
        self.width = width
        self.color = color
        SkeletonDisplayMethod.__init__(self)

registeredclass.Registration(
    'Element Edges',
    display.DisplayMethod,
    MeshEdgeDisplay,
    ordering=0.0,
    layerordering=display.Linear,
    params=meshdispparams + [
        color.TranslucentColorParameter('color', defaultMeshColor,
                                        tip="Color of the displayed edges."),
        FloatRangeParameter('width', widthRange, defaultMeshWidth,
                            tip="Line thickness, in pixels.")],
    whoclasses = ('Mesh',),
    tip="Draw the edges of Mesh Elements.",
    discussion=xmlmenudump.loadFile(
        'DISCUSSIONS/engine/reg/meshedgedisplay.xml')
)
    
registeredclass.Registration(
    'Element Edges',
    display.DisplayMethod,
    SkeletonEdgeDisplay,
    ordering=0.0,
    layerordering=display.Linear,
    params=[
        color.TranslucentColorParameter('color', defaultSkeletonColor,
                                        tip="Color of the displayed edges."),
        FloatRangeParameter('width', widthRange, defaultSkeletonWidth,
                            tip="Line thickness, in pixels.")],
    whoclasses = ('Skeleton',),
    tip="Draw the edges of Skeleton Elements.",
    discussion=xmlmenudump.loadFile(
        'DISCUSSIONS/engine/reg/skeletonedgedisplay.xml')
)

######################

## TODO: This should not just inherit the Skeleton's definition of
## "exterior", which is the edge of the rectangular Microstructure.
## It should draw any segment that has a Material on only one side,
## either because it's on the edge of the Microstructure or because
## one of its Elements is empty.

class PerimeterDisplay(MeshDisplayMethod):
    def __init__(self, when, where, width=0, color=color.black):
        self.where = where.clone()
        self.width = width
        self.color = color
        MeshDisplayMethod.__init__(self, when)
    def draw(self, gfxwindow):
        themesh = self.who.resolve(gfxwindow)
        femesh = themesh.getObject()
        themesh.restoreCachedData(self.getTime(themesh, gfxwindow))
        try:
            segs = oofcanvas.CanvasSegments.create()
            segs.setLineWidthInPixels(self.width)
            segs.setLineColor(color.canvasColor(self.color))
            for element in femesh.elements():
                el_edges = element.perimeter()
                for edge in el_edges:
                    if element.exterior(edge.startpt(), edge.endpt()):
                        pt0, pt1 = self.where.evaluate(femesh, [edge],
                                                       [[0.0, 1.0]])
                        segs.addSegment(pt0, pt1)
            self.canvaslayer.addItem(segs)
        finally:
            themesh.releaseCachedData()

registeredclass.Registration(
    'Perimeter',
    display.DisplayMethod,
    PerimeterDisplay,
    ordering=1.0,
    layerordering=display.SemiLinear,
    params=meshdispparams + [
        color.TranslucentColorParameter('color', color.black,
                                        tip=parameter.emptyTipString),
        FloatRangeParameter('width', widthRange, defaultMeshWidth,
                            tip="Line width.")
    ],
    whoclasses = ('Mesh',),
    tip="Outline the perimeter of the Mesh",
    discussion=xmlmenudump.loadFile(
        'DISCUSSIONS/engine/reg/perimeterdisplay.xml')
)

# #Interface branch
# class InterfaceElementDisplay(MeshDisplayMethod):
#     def __init__(self, when, where,
#                  boundary, #=placeholder.every.IDstring,
#                  material, #=materialparameter.InterfaceAnyMaterialParameter.extranames[0],
#                  width=0, color=color.black):
#         self.where = where.clone()
#         self.boundary=boundary
#         self.material=material
#         self.width = width
#         self.color = color
#         MeshDisplayMethod.__init__(self, when)
#     def draw(self, gfxwindow, device):
#         meshctxt = self.who.resolve(gfxwindow)
#         femesh = meshctxt.getObject()
#         device.comment("InterfaceElementDisplay")
#         device.set_lineColor(self.color)
#         device.set_lineWidth(self.width)
#         ANYstring=materialparameter.InterfaceAnyMaterialParameter.extranames[0]
#         NONEstring=materialparameter.InterfaceAnyMaterialParameter.extranames[1]
#         try:
#             meshctxt.restoreCachedData(self.getTime(meshctxt, gfxwindow))
#             if self.boundary==placeholder.every.IDstring:
#                 for edgement in femesh.interface_elements():
#                     if self.material!=ANYstring:
#                         if edgement.material():
#                             matname=edgement.material().name()
#                         else:
#                             matname=NONEstring
#                         if self.material!=matname:
#                             continue
#                     el_edges = edgement.perimeter()
#                     for edge in el_edges:
#                         pts = self.where.evaluate(femesh, [edge], [[0.0, 1.0]])
#                         device.draw_segment(primitives.Segment(pts[0], pts[1]))
#             else:
#                 for edgement in femesh.interface_elements():
#                     if self.material!=ANYstring:
#                         if edgement.material():
#                             matname=edgement.material().name()
#                         else:
#                             matname=NONEstring
#                         if self.material!=matname:
#                             continue
#                     if self.boundary not in edgement.namelist():
#                         continue
#                     el_edges = edgement.perimeter()
#                     for edge in el_edges:
#                         pts = self.where.evaluate(femesh, [edge], [[0.0, 1.0]])
#                         device.draw_segment(primitives.Segment(pts[0], pts[1]))
#         finally:
#             meshctxt.releaseCachedData()

# from ooflib.common import runtimeflags
# if runtimeflags.surface_mode:
#     registeredclass.Registration(
#         'InterfaceElement',
#         display.DisplayMethod,
#         InterfaceElementDisplay,
#         ordering=10,
#         layerordering=display.SemiLinear,
#         params=meshdispparams + [
#             meshparameters.MeshEdgeBdyParameterExtra(
#                 'boundary', placeholder.every.IDstring,
#                 tip='Only display edges on this boundary or interface.'),
#             materialparameter.InterfaceAnyMaterialParameter(
#                 'material',
#                 materialparameter.InterfaceAnyMaterialParameter.extranames[0],
#                 tip="Only display edges with this material assigned to them."),
#             ## TODO: Add settable defaults
#             color.ColorParameter('color', color.RGBColor(0.5, 0.3, 0.5),
#                                  tip=parameter.emptyTipString),
#             FloatRangeParameter('width', widthRange, defaultMeshWidth+2,
#                                 tip="Line width.")
#             ],
#         whoclasses = ('Mesh',),
#         tip="Highlight the edgements (1-D elements) on the Mesh."
#         )

######################

class MaterialDisplay:
    def draw(self, gfxwindow):
        themesh = self.who.resolve(gfxwindow)
        polygons = self.polygons(gfxwindow, themesh)
        # colorcache is a dictionary of colors keyed by Material.  It
        # prevents us from having to call material.fetchProperty for
        # each element.
        colorcache = {}
        for polygon, material in zip(polygons,
                                     self.materials(gfxwindow, themesh)):
            if material is not None:
                try:
                    # If material has been seen already, retrieve its color.
                    clr = colorcache[material]
                except KeyError:
                    # This material hasn't been seen yet.
                    try:
                        colorprop = material.fetchProperty('Color')
                        clr = color.canvasColor(colorprop.color())
                    except ooferror.PyErrNoSuchProperty:
                        clr = None
                    colorcache[material] = clr
                if clr is not None:
                    poly = oofcanvas.CanvasPolygon.create()
                    poly.setFillColor(clr)
                    poly.addPoints(polygon)
                    self.canvaslayer.addItem(poly)
 
    def getTimeStamp(self, gfxwindow):
        microstructure = self.who.resolve(gfxwindow).getMicrostructure()
        return max(display.DisplayMethod.getTimeStamp(self, gfxwindow),
                   ooflib.SWIG.engine.material.getMaterialTimeStamp(microstructure))
                    
class SkeletonMaterialDisplay(MaterialDisplay, SkeletonDisplayMethod):
    def __init__(self):
        SkeletonDisplayMethod.__init__(self)
    def materials(self, gfxwindow, skelctxt):
        skel = skelctxt.getObject()
        return (element.material(skelctxt)
                for element in skel.element_iterator())
    
class MeshMaterialDisplay(MaterialDisplay, MeshDisplayMethod):
    def __init__(self, when, where):
        self.where = where.clone()
        MeshDisplayMethod.__init__(self, when)
    def materials(self, gfxwindow, meshctxt):
        # Because MeshDisplayMethod.polygons only returns the borders
        # of elements with an assigned material, this should only
        # return the non-trivial materials.
        themesh = meshctxt.getObject()
        allmats = (element.material() for element in themesh.elements())
        if gfxwindow.settings.hideEmptyElements:
            return (mat for mat in allmats if mat)
        return allmats
    def getTimeStamp(self, gfxwindow):
        return max(MaterialDisplay.getTimeStamp(self, gfxwindow),
                   gfxwindow.settings.getTimeStamp('hideEmptyElements'))
    
    
registeredclass.Registration(
    'Material Color',
    display.DisplayMethod,
    SkeletonMaterialDisplay,
    layerordering=display.Planar(1),
    ordering=0.1,
    params=[],
    whoclasses=('Skeleton',),
    tip="Fill each Element with the color of its assigned Material.",
    discussion=xmlmenudump.loadFile(
        'DISCUSSIONS/engine/reg/skeletonmaterialdisplay.xml')
)

registeredclass.Registration(
    'Material Color',
    display.DisplayMethod,
    MeshMaterialDisplay,
    ordering=0.11,
    layerordering=display.Planar(2),
    params=meshdispparams,
    whoclasses=('Mesh',),
    tip="Fill each Element with the color of its assigned Material.",
    discussion=xmlmenudump.loadFile(
        'DISCUSSIONS/engine/reg/meshmaterialdisplay.xml')
)

###########################################

class SkeletonQualityDisplay(SkeletonDisplayMethod):
    contourmaplevels = 32
    def __init__(self, alpha, min, max, colormap):
        self.alpha = alpha
        self.colormap = colormap
        self.max = max
        self.min = min
        self.vmax = None
        self.vmin = None
        self.lock = lock.Lock()
        SkeletonDisplayMethod.__init__(self)
    def draw(self, gfxwindow):
        self.lock.acquire()
        try:
            skel = self.who.resolve(gfxwindow).getObject()
            # get polygons and element energy in one pass
            polyenergy = [(el.perimeter(), el.energyTotal(skel, self.alpha))
                        for el in skel.element_iterator()
                        if not el.illegal()]
            # find actual range of data
            self.vmax = self.vmin = polyenergy[0][1]
            for (p,e) in polyenergy[1:]:
                if e > self.vmax:
                    self.vmax = e
                if e < self.vmin:
                    self.vmin = e
            # Set plot limits to either the actual data extremes, or
            # to the passed in values.  Store the actual limits in
            # vmin and vmax.
            if self.max == automatic.automatic:
                emax = self.vmax
            else:
                emax = self.max
                self.vmax = max
            if self.min == automatic.automatic:
                emin = self.vmin
            else:
                emin = self.min
                self.vmin = min
            if emax == min:
                emax += 1.0
                self.vmax += 1.0
                emin -= 1.0
                self.vmin -= 1.0
            deltaE = emax - emin
            if deltaE == 0:
                deltaE = 1.0
            for polygon, energy in polyenergy:
                poly = oofcanvas.CanvasPolygon.create()
                poly.setFillColor(
                    color.canvasColor(self.colormap((energy-emin)/deltaE)))
                poly.addPoints(polygon)
                self.canvaslayer.addItem(poly)
        finally:
            self.lock.release()
    def getTimeStamp(self, gfxwindow):
        skelcontext = self.who.resolve(gfxwindow)
        return max(self.timestamp,
                   skelcontext.getTimeStamp(gfxwindow),
                   skelcontext.getMicrostructure().getTimeStamp())
    def contour_capable(self, gfxwindow):
        return not self.incomputable(gfxwindow)
    def get_contourmap_info(self):
        if self.vmax is not None:
            delta = (self.vmax - self.vmin)/(self.contourmaplevels-1.)
            return (self.vmin, self.vmax,
                    [self.vmin+x*delta for x in range(self.contourmaplevels)])
        return (0., 1., [0])
    def draw_contourmap(self, gfxwindow, cmaplayer):
        self.lock.acquire()
        try:
            if self.vmax is not None:
                aspect_ratio = gfxwindow.settings.aspectratio
                height = self.vmax - self.vmin
                width = height/aspect_ratio
                delta = height/(self.contourmaplevels-1.)
                for i in range(self.contourmaplevels):
                    low = i*delta
                    high = (i+1)*delta
                    rect = oofcanvas.CanvasRectangle.create((0.0, low),
                                                            (width, high))
                    if height > 0:
                        clr = color.canvasColor(self.colormap(low/height))
                    else:
                        clr = oofcanvas.black
                    rect.setFillColor(clr)
                    cmaplayer.addItem(rect)
        finally:
            self.lock.release()
    

registeredclass.Registration(
    'SkeletonQuality',
    display.DisplayMethod,
    SkeletonQualityDisplay,
    ordering=100,
    layerordering=display.Planar(1.1),
    whoclasses=('Skeleton',),
    params=[skeletonmodifier.alphaParameter,
            parameter.RegisteredParameter('colormap', colormap.ColorMap,
                                          colormap.ThermalMap(),
                                          tip="color scheme"),
            AutoNumericParameter('min', automatic.automatic,
                               tip="lowest energy to display, or 'automatic'"),
            AutoNumericParameter('max', automatic.automatic,
                              tip="highest energy to display, or 'automatic'")
            ],
    tip="Color each element according to its effective energy.",
    discussion=xmlmenudump.loadFile(
        'DISCUSSIONS/engine/reg/skelqualdisplay.xml')
    )

###########################################

def defaultSkeletonEdgeDisplay():
    return SkeletonEdgeDisplay(color=defaultSkeletonColor,
                               width=defaultSkeletonWidth)

ghostgfxwindow.DefaultLayer(skeletoncontext.skeletonContexts,
                            defaultSkeletonEdgeDisplay)

def defaultMeshEdgeDisplay():
    return MeshEdgeDisplay(when=placeholder.latest,
                           where=defaultMeshPosition,
                           color=defaultMeshColor,
                           width=defaultMeshWidth)

ghostgfxwindow.DefaultLayer(mesh.meshes, defaultMeshEdgeDisplay)
