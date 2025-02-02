# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.engine import cstrain
from ooflib.common.IO.GUI import regclassfactory
from ooflib.engine.IO.GUI import meshparamwidgets

class StrainTypeWidget(regclassfactory.RegisteredClassFactory,
                       meshparamwidgets.IndexableWidget,
                       meshparamwidgets.InvariandWidget):
    def __init__(self, value, scope, name, **kwargs):
        regclassfactory.RegisteredClassFactory.__init__(
            self, cstrain.StrainType.registry, obj=value,
            scope=scope, name=name, **kwargs)

def _makeStrainParameterWidget(self, scope=None, **kwargs):
    return StrainTypeWidget(self.value, scope=scope, name=self.name, **kwargs)

cstrain.StrainTypeParameter.makeWidget = _makeStrainParameterWidget

