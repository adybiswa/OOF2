# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.common import ooferror
from ooflib.common import debug
from ooflib.common.IO import whoville
from ooflib.common.IO.GUI import chooser
from ooflib.common.IO.GUI import parameterwidgets
from ooflib.common.IO.GUI import whowidget
from ooflib.engine import skeletoncontext
from ooflib.engine.IO import skeletonbdydisplay

# Widget for the skeleton boundary parameter.  Does clever things
# with scopes to resolve the correct graphics window, for finding the
# right skeleton, to get the boundary lists.

class SkeletonBoundaryListParamWidget(parameterwidgets.ParameterWidget,
                                      chooser.MultiListWidget):
    def __init__(self, param, scope, name=None, **kwargs):
        # Find the who widget with the skeleton.
        self.skelwidget = scope.findWidget(
            lambda x: isinstance(x, whowidget.WhoWidget)
            and x.whoclass is skeletoncontext.skeletonContexts)

        # Use it to get the context or proxy.
        skelname = self.skelwidget.get_value()
        skelctxt = skeletoncontext.skeletonContexts[skelname]
        if issubclass(skelctxt.__class__, whoville.WhoProxy):
            gfxwindow = scope.findData("gfxwindow")
            if gfxwindow is None:
                raise ooferror.PyErrUserError(
                    "Unable to find a Skeleton or its list of boundaries!")
            skelctxt = skelctxt.resolve(gfxwindow)

        bdynames = list(skelctxt.edgeboundaries.keys()) + \
                   list(skelctxt.pointboundaries.keys())
        chooser.MultiListWidget.__init__(self, objlist=bdynames,
                                         callback=self.chooserCB, **kwargs)
        parameterwidgets.ParameterWidget.__init__(self, self.gtk, scope, name,
                                                  expandable=True)
        self.set_selection(param.value)

        self.widgetChanged(1, interactive=0) # Always valid.

    def chooserCB(self, value=None, interactive=True):
        self.widgetChanged(1, interactive=interactive)

    
def _make_SBLPWidget(self, scope, **kwargs):
    return SkeletonBoundaryListParamWidget(self, scope, name=self.name,
                                           **kwargs)

skeletonbdydisplay.SkeletonBoundaryListParameter.makeWidget = _make_SBLPWidget
