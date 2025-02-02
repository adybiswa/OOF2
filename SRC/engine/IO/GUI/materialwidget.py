# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.common import switchboard
from ooflib.SWIG.engine import material
from ooflib.common import debug
from ooflib.common.IO.GUI import chooser
from ooflib.common.IO.GUI import parameterwidgets
from ooflib.common.IO.GUI import whowidget
from ooflib.engine import materialmanager
from ooflib.engine import mesh
from ooflib.engine import subproblemcontext
from ooflib.engine.IO import materialmenu
from ooflib.engine.IO import materialparameter

class MaterialWidget(parameterwidgets.ParameterWidget):
    def __init__(self, param, scope=None, name=None, **kwargs):
        self.chooser = chooser.ChooserWidget([], callback=self.chooserCB,
                                             name=name, **kwargs)
        parameterwidgets.ParameterWidget.__init__(self, self.chooser.gtk, scope)
        self.sbcallbacks = [
            switchboard.requestCallbackMain("new_material", self.update),
            switchboard.requestCallbackMain("remove_material", self.update)
            ]
        self.update()
        if param.value is not None:
            self.set_value(param.value)
    def chooserCB(self, name):
        self.widgetChanged(validity=self.chooser.nChoices()>0, interactive=1)
    def cleanUp(self):
        switchboard.removeCallbacks(self.sbcallbacks)
        parameterwidgets.ParameterWidget.cleanUp(self)
    def update(self, *args):
        names = sorted(materialmanager.getMaterialNames())
        self.chooser.update(names)
        self.widgetChanged(len(names) > 0, interactive=0)
    def get_value(self):
        return self.chooser.get_value()
    def set_value(self, material):
        self.chooser.set_state(material)
        self.widgetChanged(validity=(material is not None), interactive=0)

def _MaterialParameter_makeWidget(self, scope=None, **kwargs):
    return MaterialWidget(self, scope, name=self.name, **kwargs)
materialparameter.MaterialParameter.makeWidget = _MaterialParameter_makeWidget

#Interface branch
class InterfaceMaterialWidget(MaterialWidget):
    def update(self, *args):
        names = sorted(materialmanager.getInterfaceMaterialNames())
        self.chooser.update(names)
        self.widgetChanged(len(names) > 0, interactive=0)

def _InterfaceMaterialParameter_makeWidget(self, scope=None, **kwargs):
    return InterfaceMaterialWidget(self, scope, name=self.name, **kwargs)

materialparameter.InterfaceMaterialParameter.makeWidget = \
    _InterfaceMaterialParameter_makeWidget

#Interface branch
class BulkMaterialWidgetExtra(MaterialWidget):
    def update(self, *args):
        names = sorted(materialmanager.getBulkMaterialNames())
        self.chooser.update(
            materialparameter.BulkMaterialParameterExtra.extranames + names)
        self.widgetChanged(len(names) > 0, interactive=0)

def _BulkMaterialParameterExtra_makeWidget(self, scope=None):
    return BulkMaterialWidgetExtra(self, scope, name=self.name)
materialparameter.BulkMaterialParameterExtra.makeWidget = \
    _BulkMaterialParameterExtra_makeWidget

#Interface branch
class BulkMaterialWidget(MaterialWidget):
    def update(self, *args):
        names = sorted(materialmanager.getBulkMaterialNames())
        self.chooser.update(names)
        self.widgetChanged(len(names) > 0, interactive=0)

def _BulkMaterialParameter_makeWidget(self, scope=None, **kwargs):
    return BulkMaterialWidget(self, scope, name=self.name, **kwargs)
materialparameter.BulkMaterialParameter.makeWidget = \
    _BulkMaterialParameter_makeWidget

class MeshMaterialWidget(MaterialWidget):
    def __init__(self, param, scope=None, name=None, **kwargs):
        self.meshwidget = scope.findWidget(
            lambda w: isinstance(w, whowidget.WhoWidget)
            and w.whoclass in (mesh.meshes, subproblemcontext.subproblems))
        MaterialWidget.__init__(self, param, scope, name, **kwargs)
        self.sbcallbacks.append(
            switchboard.requestCallbackMain(self.meshwidget, self.update))
    def update(self, *args):
        meshname = self.meshwidget.get_value(depth=3)
        meshctxt = mesh.meshes[meshname]
        matls = meshctxt.getObject().getAllMaterials()
        names = sorted([m.name() for m in matls])
        self.chooser.update(names)
        self.widgetChanged(len(names) > 0, interactive=0)

def _MeshMatParam_makeWidget(self, scope=None, **kwargs):
    return MeshMaterialWidget(self, scope, name=self.name, **kwargs)
materialparameter.MeshMaterialParameter.makeWidget = _MeshMatParam_makeWidget
        
########################

class AnyMaterialWidget(MaterialWidget):
    def update(self, *args):
        names = sorted(materialmanager.getMaterialNames())
        self.chooser.update(materialparameter.AnyMaterialParameter.extranames
                            + names)
        self.widgetChanged(validity=1, interactive=0)

def _AnyMaterialParameter_makeWidget(self, scope=None, **kwargs):
    return AnyMaterialWidget(self, scope, name=self.name, **kwargs)

materialparameter.AnyMaterialParameter.makeWidget = \
    _AnyMaterialParameter_makeWidget

#Interface branch
class InterfaceAnyMaterialWidget(MaterialWidget):
    def update(self, *args):
        names = sorted(materialmanager.getInterfaceMaterialNames())
        self.chooser.update(
            materialparameter.InterfaceAnyMaterialParameter.extranames + names)
        self.widgetChanged(validity=1, interactive=0)

def _InterfaceAnyMaterialParameter_makeWidget(self, scope=None, **kwargs):
    return InterfaceAnyMaterialWidget(self, scope, name=self.name, **kwargs)

materialparameter.InterfaceAnyMaterialParameter.makeWidget = \
    _InterfaceAnyMaterialParameter_makeWidget

########################
        
class MaterialsWidget(parameterwidgets.ParameterWidget):
    def __init__(self, param, scope=None, name=None, **kwargs):
        names = sorted(materialmanager.getMaterialNames())
        self.widget = chooser.ScrolledMultiListWidget(names,
                                                      callback=self.widgetCB,
                                                      name=name, **kwargs)
        parameterwidgets.ParameterWidget.__init__(self, self.widget.gtk, scope,
                                                  expandable=True)
        self.widget.set_selection(param.value)
        self.sbcallbacks = [
            switchboard.requestCallbackMain('new_material', self.newMaterial),
            switchboard.requestCallbackMain('remove_material',
                                            self.newMaterial)
            ]
        self.widgetChanged((param.value is not None), interactive=0) 
    def cleanUp(self):
        switchboard.removeCallbacks(self.sbcallbacks)
        parameterwidgets.ParameterWidget.cleanUp(self)
    def newMaterial(self, *args):
        names = sorted(materialmanager.getMaterialNames())
        self.widget.update(names)
    def get_value(self):
        return self.widget.get_value()
    def set_value(self, value):
        self.widget.set_selection(value)
    def widgetCB(self, list, interactive):
        self.widgetChanged(len(list) > 0, interactive=1)

def _MaterialsWidget_makeWidget(self, scope=None, **kwargs):
    return MaterialsWidget(self, scope, name=self.name, **kwargs)

materialparameter.ListOfMaterialsParameter.makeWidget = \
    _MaterialsWidget_makeWidget
