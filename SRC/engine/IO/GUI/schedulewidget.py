# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.common import switchboard
from ooflib.common import debug
from ooflib.common.IO.GUI import regclassfactory
from ooflib.engine import outputschedule

class OutputScheduleWidget(regclassfactory.RegisteredClassFactory):
    def __init__(self, registry, obj=None, title=None, callback=None,
                 scope=None, name=None, **kwargs):

        self.scheduleTypeWidget = scope.findWidget(
            lambda w: (isinstance(w, regclassfactory.RegisteredClassFactory) and
                       w.registry is outputschedule.ScheduleType.registry))
                       
        regclassfactory.RegisteredClassFactory.__init__(
            self, registry, obj=obj, title=title, callback=callback,
            scope=scope, name=name, **kwargs)

        self.sbcb = switchboard.requestCallbackMain(self.scheduleTypeWidget,
                                                    self.typechanged)
    def typechanged(self, interactive):
        self.update(self.registry)

    def includeRegistration(self, registration):
        if self.scheduleTypeWidget.isValid():
            schedtype = self.scheduleTypeWidget.get_value()
            return registration.subclass.conditional == schedtype.conditional

    def cleanUp(self):
        switchboard.removeCallback(self.sbcb)
        regclassfactory.RegisteredClassFactory.cleanUp(self)


def _makeOutputScheduleWidget(self, scope, **kwargs):
    return OutputScheduleWidget(self.registry, self.value, scope=scope,
                                 name=self.name, **kwargs)
outputschedule.OutputScheduleParameter.makeWidget = _makeOutputScheduleWidget
