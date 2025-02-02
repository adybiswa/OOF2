# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.common import debug
# engine specific GUI initialization code
from ooflib.SWIG.common import config

    
import ooflib.engine.initialize

# Import widgets before GUI page definitions.
import ooflib.engine.IO.GUI.adaptivemeshrefineWidget
import ooflib.engine.IO.GUI.bdyinitparamwidget
import ooflib.engine.IO.GUI.bdymodparamwidget
import ooflib.engine.IO.GUI.boundarybuilderGUI
import ooflib.engine.IO.GUI.cijklparamwidgets
import ooflib.engine.IO.GUI.interfacewidget #Interface branch
import ooflib.engine.IO.GUI.invariantwidget
import ooflib.engine.IO.GUI.materialwidget
import ooflib.engine.IO.GUI.matrixmethodwidget
import ooflib.engine.IO.GUI.meshcswidget
import ooflib.engine.IO.GUI.meshdataGUI
import ooflib.engine.IO.GUI.meshparamwidgets
import ooflib.engine.IO.GUI.meshtimewidget
import ooflib.engine.IO.GUI.meshwidget
import ooflib.engine.IO.GUI.namedanalysiswidgets
import ooflib.engine.IO.GUI.newmeshWidget
import ooflib.engine.IO.GUI.outputdestinationwidget
import ooflib.engine.IO.GUI.outputschedulewidget
import ooflib.engine.IO.GUI.outputwidget
import ooflib.engine.IO.GUI.pbcparamwidget
import ooflib.engine.IO.GUI.profilefunctionwidget
import ooflib.engine.IO.GUI.rank3tensorwidgets
import ooflib.engine.IO.GUI.schedulewidget
import ooflib.engine.IO.GUI.skeletonbdyGUI
import ooflib.engine.IO.GUI.skeletongroupwidgets
import ooflib.engine.IO.GUI.strainwidget
import ooflib.engine.IO.GUI.tensorwidgets

# Toolbox definitions
import ooflib.engine.IO.GUI.meshcstoolboxGUI
import ooflib.engine.IO.GUI.meshinfoGUI
import ooflib.engine.IO.GUI.movenodeGUI
import ooflib.engine.IO.GUI.pinnodesGUI
import ooflib.engine.IO.GUI.pixelinfoGUI
import ooflib.engine.IO.GUI.skeletoninfoGUI
import ooflib.engine.IO.GUI.skeletonselectiontoolboxGUI

# GUI page definitions.

import ooflib.engine.IO.GUI.analyzePage
import ooflib.engine.IO.GUI.boundaryAnalysisPage
import ooflib.engine.IO.GUI.boundarycondPage
import ooflib.engine.IO.GUI.fieldPage
import ooflib.engine.IO.GUI.interfacePage #Interface branch
import ooflib.engine.IO.GUI.materialsPage
import ooflib.engine.IO.GUI.meshPage
import ooflib.engine.IO.GUI.pinnodesPage
import ooflib.engine.IO.GUI.scheduledOutputPage
import ooflib.engine.IO.GUI.skeletonBoundaryPage
import ooflib.engine.IO.GUI.skeletonPage
import ooflib.engine.IO.GUI.skeletonSelectionPage
import ooflib.engine.IO.GUI.solverPage
#import ooflib.engine.IO.GUI.outputPage
