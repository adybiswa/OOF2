# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 


# Smart widgets for the Director, with sufficient context-awareness
# to present only legal options, i.e. if the prospective boundary
# will be a loop, allow clockwise/counterclockwise, otherwise
# only allow left-to-right, top-to-bottom, etc.

from ooflib.SWIG.common import switchboard
from ooflib.common import debug
from ooflib.common import director
from ooflib.common.IO.GUI import parameterwidgets
from ooflib.common.IO.GUI import whowidget
from ooflib.engine import boundarybuilder
from ooflib.engine import skeletoncontext
from ooflib.engine import skeletonsegment
from ooflib.engine.IO.GUI import skeletongroupwidgets

#Interface branch
from ooflib.engine.IO.GUI import interfacewidget

# DirectorWidget, a smart EnumWidget that actually retrieves the
# requested segment set and sequences it, and then only presents
# allowed options for the Director.  

class DirectorWidget(parameterwidgets.EnumWidget):
    def __init__(self, enumclass, param, scope=None, name=None, **kwargs):
        parameterwidgets.EnumWidget.__init__(self, enumclass, param,
                                             scope, name, **kwargs)
        self.skelwidget = scope.findWidget(
            lambda x: isinstance(x, whowidget.WhoWidget)
            and x.whoclass is skeletoncontext.skeletonContexts)
        self.aggwidget = scope.findWidget(
            lambda x: \
            isinstance(x, skeletongroupwidgets.SkeletonAggregateWidget))

        self.segmenter = skeletongroupwidgets.segmenter[
            self.aggwidget.__class__]
            

        # We don't need to worry about the skeleton changing, because we
        # live in a modal dialog box.  So, might as well get the context
        # now.
        self.skelcontext = skeletoncontext.skeletonContexts[
            self.skelwidget.get_value()]
        # Do worry about when the aggregate changes.
        self.agg_callback = switchboard.requestCallbackMain(
            self.aggwidget, self.newAggregate)
        self.param = param
        self.update()


    def cleanUp(self):
        switchboard.removeCallback(self.agg_callback)
        parameterwidgets.EnumWidget.cleanUp(self)

    # Called at __init__ time, and also when the aggregate widget changes.
    # Runs loop-check and prevents the selection of disallowed directors.
    # Then runs the enclosed ChooserWidget's update function.
    def update(self):
        debug.mainthreadTest()
        loop = self.loop_check()
        parameterwidgets.EnumWidget.update(self)
        if loop==0:
            inactive_names=director.loopables
        elif loop==1:
            inactive_names=director.unloopables
        else: # Loop is -1, sequencing failed.
            inactive_names=director.loopables+director.unloopables

        #Interface branch
        inactive_names=inactive_names+["Non-sequenceable"]

        namelist = [ n for n in list(self.enumclass.names) if
                     n not in inactive_names ]
        if len(namelist)>0:
            self.widget.gtk.set_sensitive(1)
            self.widget.update(namelist, self.enumclass.helpdict)
            # Actually set the widget value, since the parameter may
            # have been previously set to a disallowed option.
            # "self.value" and "self.enumclass" are defined in the parent.
            self.value = self.enumclass(self.widget.get_value())
            self.widgetChanged(1, interactive=0)
        else:
            self.widget.gtk.set_sensitive(0)
            self.widget.update(["No edge sequence"])
            self.widgetChanged(0, interactive=0)

    # Just need to update when the aggregate changes.
    def newAggregate(self, *args, **kwargs):
        self.update()

    # Returns 1 for loop, 0 for line, -1 if the sequencer fails or if
    # the returned path is of length zero.
    # 
    # TODO OPT: The sequenced segments and nodes should be
    # cached somewhere, so the subsequently-called menu item doesn't
    # have to redo all this work.  While the duplication is of course
    # inherently wasteful, segment sequencing is actually pretty fast,
    # so this should be done after other performance enhancements have
    # made it noticeable.  The way this is done for boundary
    # modification is to have an "attempt" function in the skeleton
    # context and skeleton classes, and have the widget call that
    # function.  The advantage is that these data classes are a more
    # logical place to cache the partial result, although at the
    # moment, the modifiers do not cache data either.
    def loop_check(self):
        seg_set = self.segmenter(self.skelcontext, self.aggwidget.get_value())
        try:
            (segs, nodes, winding) = skeletonsegment.segSequence(seg_set)
        except skeletonsegment.SequenceError:
            return -1
        else:
            if len(nodes)>0:
                if nodes[0]==nodes[-1] and winding==[0,0]:
                    return 1
                for partner in nodes[0].getPartners():
                    if partner == nodes[-1] and winding==[0,0]:
                        return 1
                return 0
            return -1 # Zero-length list is also invalid.

def _DirectorParameter_makeWidget(self, scope=None, **kwargs):
    return DirectorWidget(self.enumclass, self, scope=scope, name=self.name,
                          **kwargs)

director.DirectorParameter.makeWidget = _DirectorParameter_makeWidget

#########################################################################
#Interface branch

class DirectorInterfacesWidget(parameterwidgets.EnumWidget):
    def __init__(self, enumclass, param, scope=None, name=None, **kwargs):
        parameterwidgets.EnumWidget.__init__(self, enumclass, param,
                                             scope, name, **kwargs)
        self.skelwidget = scope.findWidget(
            lambda x: isinstance(x, whowidget.WhoWidget)
            and x.whoclass is skeletoncontext.skeletonContexts)

##        self.aggwidget = scope.findWidget(
##            lambda x: \
##            isinstance(x, skeletongroupwidgets.SkeletonAggregateWidget))

##        self.segmenter = skeletongroupwidgets.segmenter[
##            self.aggwidget.__class__]

        self.interfacewidget = scope.findWidget(
            lambda x: \
            isinstance(x, interfacewidget.InterfacesWidget))
        
        # We don't need to worry about the skeleton changing, because we
        # live in a modal dialog box and the skeleton is set on
        # the task page.  So, might as well get the context
        # now.
        self.skelcontext = skeletoncontext.skeletonContexts[
            self.skelwidget.get_value()]
        # Do worry about when the aggregate changes.
##        self.agg_callback = switchboard.requestCallbackMain(
##            self.aggwidget, self.newAggregate)
        self.interface_callback = switchboard.requestCallbackMain(
            self.interfacewidget, self.newInterface)
        self.param = param
        self.update()


    def cleanUp(self):
        switchboard.removeCallback(self.interface_callback)
        parameterwidgets.EnumWidget.cleanUp(self)

    # Called at __init__ time, and also when the aggregate widget changes.
    # Runs loop-check and prevents the selection of disallowed directors.
    # Then runs the enclosed ChooserWidget's update function.
    def update(self):
        debug.mainthreadTest()
        loop = self.loop_check()
        parameterwidgets.EnumWidget.update(self)
        if loop==0:
            inactive_names=director.loopables
        elif loop==1:
            inactive_names=director.unloopables
        elif loop==-1: #sequencing failed.
            inactive_names=director.loopables+director.unloopables
        elif loop==-2: #no interface widget value
            inactive_names=director.loopables+director.unloopables+\
                            ["Non-sequenceable"]
        
        namelist = [ n for n in list(self.enumclass.names) if
                     n not in inactive_names ]
        if len(namelist)>0:
            self.widget.gtk.set_sensitive(1)
            self.widget.update(namelist, self.enumclass.helpdict)
            # Actually set the widget value, since the parameter may
            # have been previously set to a disallowed option.
            # "self.value" and "self.enumclass" are defined in the parent.
            self.value = self.enumclass(self.widget.get_value())
            self.widgetChanged(1, interactive=0)
        else:
            self.widget.gtk.set_sensitive(0)
            self.widget.update(["No edge sequence"])
            self.widgetChanged(0, interactive=0)
            

    # GTK callback for when a selection is made.
    def selection(self, gtkobj, name):
        parameterwidgets.EnumWidget.selection(self, gtkobj, name)

    # Just need to update when the aggregate changes.
    def newInterface(self, *args, **kwargs):
        self.update()

    # Returns 1 for loop, 0 for line, -1 if the sequencer fails or if
    # the returned path is of length zero.
    # 
    # TODO OPT: The sequenced segments and nodes should be
    # cached somewhere, so the subsequently-called menu item doesn't
    # have to redo all this work.  While the duplication is of course
    # inherently wasteful, segment sequencing is actually pretty fast,
    # so this should be done after other performance enhancements have
    # made it noticeable.  The way this is done for boundary
    # modification is to have an "attempt" function in the skeleton
    # context and skeleton classes, and have the widget call that
    # function.  The advantage is that these data classes are a more
    # logical place to cache the partial result, although at the
    # moment, the modifiers do not cache data either.
    def loop_check(self):
        #seg_set = self.segmenter(self.skelcontext, self.aggwidget.get_value())
        if self.interfacewidget.get_value() is None:
            return -2
        skelobj=self.skelcontext.getObject()
        (seg_set,direction_set)=skelobj.getInterfaceSegments(
            self.skelcontext, self.interfacewidget.get_value())
        try:
            (segs, nodes, winding) = skeletonsegment.segSequence(seg_set)
        except skeletonsegment.SequenceError:
            return -1
        else:
            if len(nodes)>0:
                if nodes[0]==nodes[-1] and winding==[0,0]:
                    return 1
                for partner in nodes[0].getPartners():
                    if partner == nodes[-1] and winding==[0,0]:
                        return 1
                return 0
            return -1 # Zero-length list is also invalid.

def _DirectorInterfacesParameter_makeWidget(self, scope=None, **kwargs):
    return DirectorInterfacesWidget(self.enumclass, self, scope=scope,
                                    name=self.name, **kwargs)

director.DirectorInterfacesParameter.makeWidget = \
    _DirectorInterfacesParameter_makeWidget
