# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

# Load a file that loads a file containing a syntax error and check
# that lines after the error aren't processed.

import tests
from ooflib.common import utils

findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
findWidget('OOF2').resize(782, 511)
checkpoint OOF.File.Load.Script
checkpoint OOF.File.LoadStartUp.Script
checkpoint toplevel widget mapped Error
findWidget('Error').resize(527, 282)
findWidget('OOF2').resize(782, 545)
assert tests.syntaxErrorMsg('File "TEST_DATA/syntaxerror.py", line 2\n    \'Twas brillig, and the slithy toves did gyre and gimble in the wabe.\n                                                                       ^\nSyntaxError: EOL while scanning string literal\n\nErrUserError: Script \'TEST_DATA/nestedsyntaxerr.py\' raised a SyntaxError exception')

findWidget('Error').resize(527, 282)
findWidget('Error:widget_GTK_RESPONSE_OK').clicked()
assert tests.noExecution()
assert utils.OOFeval("teststring") == "ok"

utils.OOFexec("teststring = None")

findMenu(findWidget('OOF2:MenuBar'), ['File', 'Load', 'Script']).activate()
checkpoint toplevel widget mapped Dialog-Script
findWidget('Dialog-Script').resize(453, 152)
findWidget('Dialog-Script:filename').set_text('TEST_DATA/nestedsyntaxerr.py')
findWidget('Dialog-Script:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.File.Load.Script
checkpoint toplevel widget mapped Error
findWidget('Error').resize(266, 162)
assert tests.syntaxErrorMsg('File "TEST_DATA/syntaxerror.py", line 2\n    \'Twas brillig, and the slithy toves did gyre and gimble in the wabe.\n                                                                       ^\nSyntaxError: EOL while scanning string literal\n\nErrUserError: Script \'TEST_DATA/nestedsyntaxerr.py\' raised a SyntaxError exception')

findWidget('Error:widget_GTK_RESPONSE_OK').clicked()
assert tests.noExecution()
assert utils.OOFeval("teststring") == "ok"

findMenu(findWidget('OOF2:MenuBar'), ['File', 'Quit']).activate()