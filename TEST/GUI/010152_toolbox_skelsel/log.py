# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

# Segment mode test in the Skeleton Selection toolbox

import tests
tbox="OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection"
elbox=tbox+":Element"
ndbox=tbox+":Node"
sgbox=tbox+":Segment"

checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
checkpoint toplevel widget mapped OOF2 Activity Viewer
findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
findWidget('OOF2').resize(782, 511)

findMenu(findWidget('OOF2:MenuBar'), ['Windows', 'Graphics', 'New']).activate()
checkpoint Move Node toolbox info updated
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint toplevel widget mapped OOF2 Graphics 1
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Windows.Graphics.New
findWidget('OOF2 Graphics 1:Pane0').set_position(360)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(672)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(212)
findWidget('OOF2 Graphics 1').resize(800, 492)
findWidget('OOF2').resize(782, 545)
findWidget('OOF2 Graphics 1').resize(800, 492)
findMenu(findWidget('OOF2 Graphics 1:MenuBar'), ['Settings', 'New_Layer_Policy']).activate()
checkpoint toplevel widget mapped Dialog-New_Layer_Policy
findWidget('Dialog-New_Layer_Policy').resize(192, 86)
wevent(findWidget('Dialog-New_Layer_Policy:policy'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('Dialog-New_Layer_Policy:policy').get_window())
checkpoint toplevel widget mapped chooserPopup-policy
findMenu(findWidget('chooserPopup-policy'), ['Single']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-policy') # MenuItemLogger
findWidget('Dialog-New_Layer_Policy:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Graphics_1.Settings.New_Layer_Policy

# Open the Skeleton Selection toolbox
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Skeleton Selection']).activate() # MenuItemLogger
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(265)

# Switch to segment mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Select:Segment').clicked()
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'', 'ydown':'', 'xup':'', 'yup':''}, sgbox)
assert tests.sensitizationCheck({'Undo':False,'Redo':False,'Clear':False,'Invert':False},sgbox)
assert tests.sensitizationCheck({'Prev':False,'Repeat':False,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","No Skeleton!")

# Load a skeleton
findMenu(findWidget('OOF2:MenuBar'), ['File', 'Load', 'Data']).activate()
checkpoint toplevel widget mapped Dialog-Data
findWidget('Dialog-Data').resize(192, 92)
findWidget('Dialog-Data:filename').set_text('e')
findWidget('Dialog-Data:filename').set_text('ex')
findWidget('Dialog-Data:filename').set_text('exa')
findWidget('Dialog-Data:filename').set_text('exam')
findWidget('Dialog-Data:filename').set_text('examp')
findWidget('Dialog-Data:filename').set_text('exampl')
findWidget('Dialog-Data:filename').set_text('example')
findWidget('Dialog-Data:filename').set_text('examples')
findWidget('Dialog-Data:filename').set_text('examples/')
findWidget('Dialog-Data:filename').set_text('examples/t')
findWidget('Dialog-Data:filename').set_text('examples/tr')
findWidget('Dialog-Data:filename').set_text('examples/tri')
findWidget('Dialog-Data:filename').set_text('examples/tria')
findWidget('Dialog-Data:filename').set_text('examples/trian')
findWidget('Dialog-Data:filename').set_text('examples/triang')
findWidget('Dialog-Data:filename').set_text('examples/triangl')
findWidget('Dialog-Data:filename').set_text('examples/triangle')
findWidget('Dialog-Data:filename').set_text('examples/triangle.')
findWidget('Dialog-Data:filename').set_text('examples/triangle.s')
findWidget('Dialog-Data:filename').set_text('examples/triangle.sk')
findWidget('Dialog-Data:filename').set_text('examples/triangle.ske')
findWidget('Dialog-Data:filename').set_text('examples/triangle.skel')
findWidget('Dialog-Data:filename').set_text('examples/triangle.skele')
findWidget('Dialog-Data:filename').set_text('examples/triangle.skelet')
findWidget('Dialog-Data:filename').set_text('examples/triangle.skeleto')
findWidget('Dialog-Data:filename').set_text('examples/triangle.skeleton')
findWidget('Dialog-Data:widget_GTK_RESPONSE_OK').clicked()
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint pixel page sensitized
checkpoint active area status updated
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint Materials page updated
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint boundary page updated
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint Solver page sensitized
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint Graphics_1 Element sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint microstructure page sensitized
checkpoint microstructure page sensitized
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint Move Node toolbox writable changed
checkpoint Move Node toolbox info updated
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint Graphics_1 Element sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint boundary page updated
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint Solver page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint mesh bdy page updated
checkpoint boundary page updated
checkpoint mesh bdy page updated
checkpoint boundary page updated
checkpoint mesh bdy page updated
checkpoint boundary page updated
checkpoint mesh bdy page updated
checkpoint boundary page updated
checkpoint mesh bdy page updated
checkpoint boundary page updated
checkpoint mesh bdy page updated
checkpoint boundary page updated
checkpoint mesh bdy page updated
checkpoint boundary page updated
checkpoint mesh bdy page updated
checkpoint boundary page updated
checkpoint OOF.File.Load.Data
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'', 'ydown':'', 'xup':'', 'yup':''}, sgbox)
assert tests.sensitizationCheck({'Undo':False,'Redo':False,'Clear':False,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':False,'Repeat':False,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","0 (0%)")


# Select a segment
findGfxWindow('Graphics_1').simulateMouse('down', 22.525, 75.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 22.525, 75.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 22.525, 75.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 22.525, 75.55, 1, False, False)
checkpoint Graphics_1 Segment sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'--', 'ydown':'--', 'xup':'22.525', 'yup':'75.55'}, sgbox)
assert tests.sensitizationCheck({'Undo':True,'Redo':False,'Clear':True,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':False,'Repeat':True,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","1 (0.724638%)")

# Select a rectangle
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Method:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Rectangle']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 2.4000000000000e+01)
findGfxWindow('Graphics_1').simulateMouse('down', 8.175, 80.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 8.175, 80.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 8.175, 79.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 8.525, 75.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 9.575, 71.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 12.725, 67.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 17.975, 64.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 22.525, 61.9, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 35.475, 57.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 44.925, 54.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.025, 52.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 61.725, 50.7, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 67.675, 48.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 72.225, 47.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 76.075, 46.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 79.225, 46.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 46.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.975, 46.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 83.425, 46.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 84.475, 45.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 85.175, 46.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 86.225, 46.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 89.375, 45.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.475, 45.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.475, 45.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.475, 45.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.825, 45.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.875, 45.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.875, 45.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.875, 45.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 92.875, 45.45, 1, False, False)
checkpoint Graphics_1 Segment sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Rectangle
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'8.175', 'ydown':'80.8', 'xup':'92.875', 'yup':'45.45'}, sgbox)
assert tests.sensitizationCheck({'Undo':True,'Redo':False,'Clear':True,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Repeat':True,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","28 (20.2899%)")

# Select a circle
findWidget('OOF2 Graphics 1').resize(800, 492)
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Method:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Circle']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findGfxWindow('Graphics_1').simulateMouse('down', 49.475, 49.65, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 49.475, 49.65, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 49.825, 49.65, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.225, 50.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.675, 51.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.775, 52.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 58.575, 53.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 61.375, 55.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 64.525, 57, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 66.625, 58.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 68.025, 59.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 69.075, 60.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 70.125, 62.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 72.225, 64, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 73.975, 65.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 75.725, 66.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 76.425, 67.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 77.475, 68.2, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 77.825, 68.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 78.525, 69.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 78.875, 69.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 79.575, 69.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 79.925, 69.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 70.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 70.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 70.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 70.65, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 71, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 71, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 71.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 71.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.275, 71.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.625, 71.7, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.625, 72.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.625, 72.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.625, 72.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.625, 72.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.625, 72.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.975, 72.75, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.975, 72.75, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.975, 73.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 80.975, 73.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 81.325, 73.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 81.325, 73.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 81.325, 73.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 81.325, 73.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 81.675, 73.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 81.675, 73.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 81.675, 73.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 81.675, 73.45, 1, False, False)
checkpoint Graphics_1 Segment sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Circle
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'49.475', 'ydown':'49.65', 'xup':'81.675', 'yup':'73.45'}, sgbox)
assert tests.sensitizationCheck({'Undo':True,'Redo':False,'Clear':True,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Repeat':True,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","47 (34.058%)")

# Select an ellipse
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findWidget('chooserPopup-TBChooser').deactivate() # MenuLogger
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Method:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Ellipse']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findGfxWindow('Graphics_1').simulateMouse('down', 26.375, 69.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 26.375, 69.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 26.375, 68.9, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 26.025, 67.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 25.325, 65.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 24.275, 59.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 24.625, 54.9, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 25.675, 51.75, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 27.075, 50, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 28.125, 48.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 28.825, 47.2, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 30.225, 45.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 33.025, 44.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 38.275, 40.9, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 44.925, 38.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 50.875, 35.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 58.925, 33.2, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 63.125, 31.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 66.975, 28.65, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 71.875, 25.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 75.025, 23.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 78.525, 22, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 81.325, 20.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 83.075, 20.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 84.125, 21.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 85.175, 21.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 85.875, 21.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 86.925, 21.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 88.325, 20.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 89.725, 20.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 90.775, 19.9, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 91.825, 19.2, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.175, 19.2, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.175, 18.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.525, 18.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.525, 18.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 92.875, 18.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 93.575, 18.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 93.925, 18.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 94.975, 18.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 96.375, 18.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 98.125, 18.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 98.125, 18.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 98.125, 18.15, 1, False, False)
checkpoint Graphics_1 Segment sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Ellipse
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'26.375', 'ydown':'69.25', 'xup':'98.125', 'yup':'18.15'}, sgbox)
assert tests.sensitizationCheck({'Undo':True,'Redo':False,'Clear':True,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Repeat':True,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","22 (15.942%)")

# Clear
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Clear').clicked()
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Clear
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'26.375', 'ydown':'69.25', 'xup':'98.125', 'yup':'18.15'}, sgbox)
assert tests.sensitizationCheck({'Undo':True,'Redo':False,'Clear':False,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Repeat':True,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","0 (0%)")

# Undo
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 2.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Undo').clicked()
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Undo
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 0.0000000000000e+00)
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'26.375', 'ydown':'69.25', 'xup':'98.125', 'yup':'18.15'}, sgbox)
assert tests.sensitizationCheck({'Undo':True,'Redo':True,'Clear':True,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Repeat':True,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","22 (15.942%)")

# undo again
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Undo').clicked()
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Undo
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'26.375', 'ydown':'69.25', 'xup':'98.125', 'yup':'18.15'}, sgbox)
assert tests.sensitizationCheck({'Undo':True,'Redo':True,'Clear':True,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Repeat':True,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","47 (34.058%)")

# redo
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Redo').clicked()
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Redo
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'26.375', 'ydown':'69.25', 'xup':'98.125', 'yup':'18.15'}, sgbox)
assert tests.sensitizationCheck({'Undo':True,'Redo':True,'Clear':True,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Repeat':True,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","22 (15.942%)")

# invert
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 2.8000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Invert').clicked()
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Invert
assert not findWidget(tbox+":Select:Element").get_active()
assert not findWidget(tbox+":Select:Node").get_active()
assert findWidget(tbox+":Select:Segment").get_active()
assert tests.gtkMultiTextCompare({'xdown':'26.375', 'ydown':'69.25', 'xup':'98.125', 'yup':'18.15'}, sgbox)
assert tests.sensitizationCheck({'Undo':True,'Redo':False,'Clear':True,'Invert':True},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Repeat':True,'Next':False},sgbox)
assert tests.gtkTextCompare(sgbox+":size","116 (84.058%)")

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
findWidget('OOF2:Introduction Page:Scroll').get_vadjustment().set_value( 7.0000000000000e+00)
assert tests.filediff('session.log')

findMenu(findWidget('OOF2:MenuBar'), ['File', 'Quit']).activate()
checkpoint OOF.Graphics_1.File.Close
