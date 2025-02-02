# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

from ooflib.common import debug
from ooflib.common import primitives
from ooflib.common import registeredclass
from ooflib.common import utils
from ooflib.common.IO import parameter
from ooflib.common.IO import xmlmenudump
from ooflib.engine import skeletoncontext
from ooflib.engine.IO import skeletongroupparams

def _undo(menuitem, skeleton):
    skelcontext = skeletoncontext.skeletonContexts[skeleton]
    skelcontext.pinnednodes.undo()
    skelcontext.pinnednodes.signal()

def _redo(menuitem, skeleton):
    skelcontext = skeletoncontext.skeletonContexts[skeleton]
    skelcontext.pinnednodes.redo()
    skelcontext.pinnednodes.signal()

def _unpinall(menuitem, skeleton):
    skelcontext = skeletoncontext.skeletonContexts[skeleton]
    skelcontext.pinnednodes.start()
    skelcontext.pinnednodes.clear()
    skelcontext.pinnednodes.signal()

def _invert(menuitem, skeleton):
    skelcontext = skeletoncontext.skeletonContexts[skeleton]
    newpinned = skelcontext.getObject().notPinnedNodes()
    skelcontext.pinnednodes.start()
    skelcontext.pinnednodes.clear()
    skelcontext.pinnednodes.pin(newpinned)
    skelcontext.pinnednodes.signal()

def pinnodesmod(menuitem, skeleton, **params):
    registration = menuitem.data
    modifier = registration(**params)
    modifier(skeleton)

class PinNodesModifier(registeredclass.RegisteredClass):
    registry = []
    # Menu items are created from the subclasses, and the
    # documentation is in the menu items.  Why not use a
    # PinnedNodes.Modify menu item that takes a PinNodesModifier as an
    # argument?

class PinNodeSelection(PinNodesModifier):
    def __call__(self, skeleton):
        skelcontext = skeletoncontext.skeletonContexts[skeleton]
        pinnednodes = skelcontext.pinnednodes
        pinnednodes.start()
        pinnednodes.pin(skelcontext.nodeselection.retrieve())
        pinnednodes.signal()

registeredclass.Registration(
    'Pin Node Selection',
    PinNodesModifier,
    PinNodeSelection,
    ordering=0,
    tip="Pin selected nodes.",
    discussion="""<para>
    <link linkend='MenuItem-OOF.Skeleton.PinNodes'>Pin</link> the
    currently <link
    linkend='MenuItem-OOF.NodeSelection'>selected</link> &nodes;.
    </para>""",
    xrefs=["Section-Tasks-SkeletonSelection",
           "Section-Graphics-SkeletonSelection"]
)


class UnPinNodeSelection(PinNodesModifier):
    def __call__(self, skeleton):
        skelcontext = skeletoncontext.skeletonContexts[skeleton]
        pinnednodes = skelcontext.pinnednodes
        pinnednodes.start()
        pinnednodes.unpin(skelcontext.nodeselection.retrieve())
        pinnednodes.signal()

registeredclass.Registration(
    'UnPin Node Selection',
    PinNodesModifier,
    UnPinNodeSelection,
    ordering=1,
    tip="Unpin selected nodes.",
    discussion="""<para>
    <link linkend='MenuItem-OOF.Skeleton.PinNodes'>Unpin</link> the
    currently <link
    linkend='MenuItem-OOF.NodeSelection'>selected</link> &nodes;.
    </para>""",
    xrefs=["Section-Tasks-SkeletonSelection",
           "Section-Graphics-SkeletonSelection"]
)


class PinInternalBoundaryNodes(PinNodesModifier):
    def __call__(self, skeleton):
        skelcontext = skeletoncontext.skeletonContexts[skeleton]
        skel = skelcontext.getObject()
        pinnednodes = skelcontext.pinnednodes
        nodelist = []
        for node in skel.nodes:
            elements = node.neighborElements()
            cat = elements[0].dominantPixel(skel.MS)
            for element in elements[1:]:
                if cat != element.dominantPixel(skel.MS):
                    nodelist.append(node)
                    break
        pinnednodes.start()
        pinnednodes.pin(nodelist)
        pinnednodes.signal()

registeredclass.Registration(
    'Pin Internal Boundary Nodes',
    PinNodesModifier,
    PinInternalBoundaryNodes,
    ordering=2,
    tip="Pin all internal boundary nodes.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/engine/reg/pininternal.xml')
)

class PinSelectedSegments(PinNodesModifier):
    def __call__(self, skeleton):
        skelcontext = skeletoncontext.skeletonContexts[skeleton]
        pinnednodes = skelcontext.pinnednodes
        nodes = set()
        for segment in skelcontext.segmentselection.retrieve():
            nodes.add(segment.nodes()[0])
            nodes.add(segment.nodes()[1])
        pinnednodes.start()
        pinnednodes.pin(nodes)
        pinnednodes.signal()

registeredclass.Registration(
    'Pin Selected Segments',
    PinNodesModifier,
    PinSelectedSegments,
    ordering=4,
    tip="Pin nodes of selected segments.",
    discussion="""<para>
    <link linkend='MenuItem-OOF.Skeleton.PinNodes'>Pin</link> the
    &nodes; at the ends of the currently selected &sgmts; in the given
    &skel;.
    </para> """,
    xrefs=["Section-Tasks-SkeletonSelection",
           "Section-Graphics-SkeletonSelection"]
)


class PinSelectedElements(PinNodesModifier):
    def __init__(self, internal=0, boundary=1):
        self.internal = internal
        self.boundary = boundary

    def getAllNodes(self, context):
        nodes = set()
        for element in context.elementselection.retrieve():
            for nd in element.nodes:
                nodes.add(nd)
        return nodes

    def getBoundaryNodes(self, context):
        bound = set()
        for element in context.elementselection.retrieve():
            for i in range(element.nnodes()):
                n0 = element.nodes[i]
                n1 = element.nodes[(i+1)%element.nnodes()]
                seg = context.getObject().getSegment(n0, n1)
                # A segment is on the boundary of the selection if it
                # belongs to only one element.
                n = 0
                for el in seg.getElements():
                    if el.selected: n += 1
                if n == 1:
                    bound.add(n0)
                    bound.add(n1)
        return bound

    def getInternalNodes(self, context, allnodes):
        return allnodes - self.getBoundaryNodes(context)

    def __call__(self, skeleton):
        skelcontext = skeletoncontext.skeletonContexts[skeleton]
        pinnednodes = skelcontext.pinnednodes
        if self.internal and self.boundary:
            nodes = self.getAllNodes(skelcontext)
        elif not self.internal and self.boundary:
            nodes = self.getBoundaryNodes(skelcontext)
        elif self.internal and not self.boundary:
            allnodes = self.getAllNodes(skelcontext)
            nodes = self.getInternalNodes(skelcontext, allnodes)
        else:
            nodes = []
        pinnednodes.start()
        pinnednodes.pin(nodes)
        pinnednodes.signal()

registeredclass.Registration(
    'Pin Selected Elements',
    PinNodesModifier,
    PinSelectedElements,
    ordering=5,
    params=[parameter.BooleanParameter('internal', 0,
                                       tip='Select internal nodes.'),
            parameter.BooleanParameter('boundary', 1,
                                       tip='Select boundary nodes.')],
    tip="Pin nodes of selected elements.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/engine/reg/pinelements.xml'),
    xrefs=["Section-Tasks-SkeletonSelection",
           "Section-Graphics-SkeletonSelection"]
)
