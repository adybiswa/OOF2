# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

from ooflib.common import utils
import generics

findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
findWidget('OOF2').resize(782, 511)
checkpoint OOF.File.LoadStartUp.Script
checkpoint toplevel widget mapped Error
assert generics.errorMsg("NameError: global name 'y' is not defined\n\nErrUserError: Script 'TEST_DATA/pyerror.py' raised a NameError exception")

findWidget('Error').resize(266, 198)
findWidget('Error:widget_GTK_RESPONSE_OK').clicked()
assert utils.OOFeval('teststring') == 'ok'

utils.OOFexec("teststring = None")

findMenu(findWidget('OOF2:MenuBar'), ['File', 'Load', 'Script']).activate()
checkpoint toplevel widget mapped Dialog-Script
findWidget('Dialog-Script').resize(192, 92)
findWidget('OOF2').resize(782, 545)
findWidget('Dialog-Script:filename').set_text('')
findWidget('Dialog-Script:filename').set_text('T')
findWidget('Dialog-Script:filename').set_text('TE')
findWidget('Dialog-Script:filename').set_text('TES')
findWidget('Dialog-Script:filename').set_text('TEST')
findWidget('Dialog-Script:filename').set_text('TEST_')
findWidget('Dialog-Script:filename').set_text('TEST_D')
findWidget('Dialog-Script:filename').set_text('TEST_DA')
findWidget('Dialog-Script:filename').set_text('TEST_DAT')
findWidget('Dialog-Script:filename').set_text('TEST_DATA')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/p')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/py')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/pye')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/pyer')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/pyerr')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/pyerro')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/pyerror')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/pyerror.')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/pyerror.p')
findWidget('Dialog-Script:filename').set_text('TEST_DATA/pyerror.py')
findWidget('Dialog-Script:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.File.Load.Script
checkpoint toplevel widget mapped Error
assert generics.errorMsg("NameError: global name 'y' is not defined\n\nErrUserError: Script 'TEST_DATA/pyerror.py' raised a NameError exception")
findWidget('Error').resize(266, 198)
findWidget('Error:widget_GTK_RESPONSE_OK').clicked()
assert utils.OOFeval('teststring') == 'ok'
findMenu(findWidget('OOF2:MenuBar'), ['File', 'Quit']).activate()
