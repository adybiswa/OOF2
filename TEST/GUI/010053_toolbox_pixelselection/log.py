# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# Check that modifier keys work correctly with the Repeat button in
# the Pixel Selection toolbox.

import tests
findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
findWidget('OOF2').resize(782, 545)
wevent(findWidget('OOF2:Navigation:PageMenu'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Microstructure']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Microstructure
findWidget('OOF2:Microstructure Page:Pane').set_position(235)
checkpoint meshable button set
checkpoint microstructure page sensitized
findWidget('OOF2:Microstructure Page:Pane').set_position(184)
findWidget('OOF2:Microstructure Page:NewFromFile').clicked()
checkpoint toplevel widget mapped Dialog-Load Image and create Microstructure
findWidget('Dialog-Load Image and create Microstructure').resize(237, 200)
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('e')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('ex')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('exa')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('exam')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examp')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('exampl')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('example')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/s')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/sm')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/sma')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/smal')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/small')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/small.')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/small.p')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/small.pp')
findWidget('Dialog-Load Image and create Microstructure:filename').set_text('examples/small.ppm')
findWidget('Dialog-Load Image and create Microstructure:widget_GTK_RESPONSE_OK').clicked()
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint active area status updated
checkpoint pixel page sensitized
findWidget('OOF2:Microstructure Page:Pane').set_position(189)
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
checkpoint microstructure page sensitized
checkpoint OOF.Microstructure.Create_From_ImageFile
findMenu(findWidget('OOF2:MenuBar'), ['Windows', 'Graphics', 'New']).activate()
checkpoint Move Node toolbox info updated
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
findWidget('OOF2 Graphics 1:Pane0').set_position(360)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(672)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(212)
checkpoint toplevel widget mapped OOF2 Graphics 1
findWidget('OOF2 Graphics 1').resize(800, 492)
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Windows.Graphics.New
findWidget('OOF2 Graphics 1').resize(800, 492)
findMenu(findWidget('OOF2 Graphics 1:MenuBar'), ['Layer', 'New']).activate()
checkpoint toplevel widget mapped Dialog-New
findWidget('Dialog-New').resize(395, 532)
wevent(findWidget('Dialog-New:category'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('Dialog-New:category').get_window())
checkpoint toplevel widget mapped chooserPopup-category
findMenu(findWidget('chooserPopup-category'), ['Image']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-category') # MenuItemLogger
findWidget('Dialog-New:widget_GTK_RESPONSE_OK').clicked()
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.0000000000000e+00)
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Layer.New
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Pixel Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(249)
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
# Select a rectangle
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Method:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Rectangle']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 4.0000000000000e+01)
findGfxWindow('Graphics_1').simulateMouse('down', 19.6125, 92.325, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 19.6125, 91.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 19.6125, 89.175, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 20.6625, 84.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 22.7625, 81.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 26.4375, 78.675, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 31.1625, 76.575, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 36.4125, 73.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 44.2875, 71.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 58.9875, 69.75, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 73.6875, 66.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 85.7625, 66.075, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 97.3125, 65.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 113.0625, 65.025, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 127.2375, 64.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 137.2125, 64.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 139.3125, 65.025, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 139.8375, 65.025, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 140.8875, 66.075, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 141.9375, 66.075, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 142.9875, 66.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 143.5125, 66.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 143.5125, 66.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 143.5125, 66.6, 1, False, False)
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint pixel page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Rectangle
assert tests.pixelSelectionSize(3375)

# Select a circle that intersects the rectangle
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Method:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Circle']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findGfxWindow('Graphics_1').simulateMouse('down', 76.8375, 81.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 77.3625, 81.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 77.8875, 80.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 79.4625, 78.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 81.5625, 75.525, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 83.1375, 71.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 84.7125, 69.225, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 85.7625, 66.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 86.2875, 65.025, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 86.8125, 63.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 88.3875, 61.875, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 88.9125, 59.775, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 89.4375, 59.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 89.9625, 57.675, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.0125, 56.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.0125, 55.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.5375, 55.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.5375, 54.525, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.5375, 54.525, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.5375, 54.525, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.0625, 54.525, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.0625, 54, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 92.0625, 54, 1, False, False)
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint pixel page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Circle
assert tests.pixelSelectionSize(3067)

# Undo the circle selection
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 2.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Undo').clicked()
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint pixel page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Undo
assert tests.pixelSelectionSize(3375)

# Repeat with no modifier keys
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat'), Gdk.EventType.BUTTON_RELEASE, button=1, state=256, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat').get_window())
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat').clicked()
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint pixel page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Circle
assert tests.pixelSelectionSize(3067)

# Undo
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Undo').clicked()
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint pixel page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Undo

# Repeat with the shift key
findWidget('OOF2 Graphics 1').resize(800, 492)
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat'), Gdk.EventType.BUTTON_RELEASE, button=1, state=257, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat').get_window())
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat').clicked()
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint pixel page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Circle
assert tests.pixelSelectionSize(4814)

# Undo
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Undo').clicked()
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint pixel page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Undo

# Repeat with the control key
findWidget('OOF2 Graphics 1').resize(800, 492)
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat'), Gdk.EventType.BUTTON_RELEASE, button=1, state=260, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat').get_window())
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat').clicked()
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint pixel page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Circle
assert tests.pixelSelectionSize(3186)

# Undo
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Undo').clicked()
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint pixel page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Undo

# Repeat with shift and control keys
findWidget('OOF2 Graphics 1').resize(800, 492)
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat'), Gdk.EventType.BUTTON_RELEASE, button=1, state=261, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat').get_window())
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Repeat').clicked()
checkpoint microstructure page sensitized
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint pixel page updated
checkpoint pixel page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Circle
assert tests.pixelSelectionSize(1628)

findWidget('OOF2').resize(782, 545)
findMenu(findWidget('OOF2:MenuBar'), ['File', 'Save', 'Python_Log']).activate()
checkpoint toplevel widget mapped Dialog-Python_Log
findWidget('Dialog-Python_Log').resize(192, 122)
findWidget('Dialog-Python_Log:filename').set_text('s')
findWidget('Dialog-Python_Log:filename').set_text('se')
findWidget('Dialog-Python_Log:filename').set_text('ses')
findWidget('Dialog-Python_Log:filename').set_text('sess')
findWidget('Dialog-Python_Log:filename').set_text('sessi')
findWidget('Dialog-Python_Log:filename').set_text('sessio')
findWidget('Dialog-Python_Log:filename').set_text('session')
findWidget('Dialog-Python_Log:filename').set_text('session.')
findWidget('Dialog-Python_Log:filename').set_text('session.l')
findWidget('Dialog-Python_Log:filename').set_text('session.lo')
findWidget('Dialog-Python_Log:filename').set_text('session.log')
findWidget('Dialog-Python_Log:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.File.Save.Python_Log
assert tests.filediff("session.log")

findMenu(findWidget('OOF2:MenuBar'), ['File', 'Quit']).activate()
checkpoint OOF.Graphics_1.File.Close