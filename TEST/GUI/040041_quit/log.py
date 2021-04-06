# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# Test quitting by closing the main OOF2 window and also saving a log file

findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
findWidget('OOF2').resize(782, 511)

findWidget('OOF2:Navigation:Next').clicked()
checkpoint page installed Microstructure
checkpoint meshable button set
checkpoint microstructure page sensitized
findWidget('OOF2:Microstructure Page:Pane').set_position(184)
findWidget('OOF2:Microstructure Page:New').clicked()
checkpoint toplevel widget mapped Dialog-Create Microstructure
findWidget('Dialog-Create Microstructure').resize(210, 236)
findWidget('OOF2').resize(782, 545)
findWidget('Dialog-Create Microstructure:widget_GTK_RESPONSE_OK').clicked()
checkpoint meshable button set
checkpoint microstructure page sensitized
findWidget('OOF2:Microstructure Page:Pane').set_position(189)
checkpoint active area status updated
checkpoint pixel page sensitized
checkpoint pixel page updated
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint Materials page updated
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint boundary page updated
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized Element
checkpoint Solver page sensitized
checkpoint OOF.Microstructure.New
event(Gdk.EventType.DELETE,window=findWidget('OOF2').get_window())
checkpoint toplevel widget mapped Questioner
findWidget('Questioner').resize(336, 86)
findWidget('Questioner:Save').clicked()
checkpoint toplevel widget mapped Dialog-Save Log File
findWidget('Dialog-Save Log File').resize(192, 122)
findWidget('Dialog-Save Log File:filename').set_text('s')
findWidget('Dialog-Save Log File:filename').set_text('se')
findWidget('Dialog-Save Log File:filename').set_text('ses')
findWidget('Dialog-Save Log File:filename').set_text('sess')
findWidget('Dialog-Save Log File:filename').set_text('sessi')
findWidget('Dialog-Save Log File:filename').set_text('sessio')
findWidget('Dialog-Save Log File:filename').set_text('session')
findWidget('Dialog-Save Log File:filename').set_text('session.')
findWidget('Dialog-Save Log File:filename').set_text('session.o')
findWidget('Dialog-Save Log File:filename').set_text('session.')
findWidget('Dialog-Save Log File:filename').set_text('session.l')
findWidget('Dialog-Save Log File:filename').set_text('session.lo')
findWidget('Dialog-Save Log File:filename').set_text('session.log')
findWidget('Dialog-Save Log File:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.File.Save.Python_Log
