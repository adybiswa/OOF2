# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

import tests
findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
findWidget('OOF2').resize(782, 511)
# Open a graphics window
findWidget('OOF2').resize(782, 511)
findMenu(findWidget('OOF2:MenuBar'), ['Windows', 'Graphics', 'New']).activate()
checkpoint Move Node toolbox info updated
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
findWidget('OOF2 Graphics 1:Pane0').set_position(360)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(672)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(212)
checkpoint toplevel widget mapped OOF2 Graphics 1
findWidget('OOF2 Graphics 1').resize(800, 492)
findWidget('OOF2').resize(782, 545)
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Windows.Graphics.New
findMenu(findWidget('OOF2 Graphics 1:MenuBar'), ['Settings', 'New_Layer_Policy']).activate()
checkpoint toplevel widget mapped Dialog-New_Layer_Policy
findWidget('Dialog-New_Layer_Policy').resize(192, 86)
event(Gdk.EventType.BUTTON_PRESS,x= 5.8000000000000e+01,y= 1.8000000000000e+01,button=1,state=0,window=findWidget('Dialog-New_Layer_Policy:policy').get_window())
checkpoint toplevel widget mapped chooserPopup-policy
findMenu(findWidget('chooserPopup-policy'), ['Single']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-policy') # MenuItemLogger
findWidget('Dialog-New_Layer_Policy:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Graphics_1.Settings.New_Layer_Policy
# Create a Microstructure and a Skeleton
event(Gdk.EventType.BUTTON_PRESS,x= 8.5000000000000e+01,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Microstructure']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Microstructure
findWidget('OOF2:Microstructure Page:Pane').set_position(235)
checkpoint meshable button set
checkpoint microstructure page sensitized
findWidget('OOF2:Microstructure Page:Pane').set_position(184)
findWidget('OOF2:Microstructure Page:New').clicked()
checkpoint toplevel widget mapped Dialog-Create Microstructure
findWidget('Dialog-Create Microstructure').resize(210, 236)
findWidget('Dialog-Create Microstructure:widget_GTK_RESPONSE_OK').clicked()
checkpoint meshable button set
findWidget('OOF2:Microstructure Page:Pane').set_position(189)
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint active area status updated
checkpoint pixel page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 5.4000000000000e+01,y= 1.8000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
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
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
findWidget('OOF2:Skeleton Page:Pane').set_position(417)
checkpoint skeleton page sensitized
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:New').clicked()
checkpoint toplevel widget mapped Dialog-New skeleton
findWidget('Dialog-New skeleton').resize(346, 254)
findWidget('Dialog-New skeleton:widget_GTK_RESPONSE_OK').clicked()
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized Element
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page sensitized
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
checkpoint contourmap info updated for Graphics_1
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint boundary page updated
checkpoint skeleton selection page grouplist Element
checkpoint skeleton page info updated
checkpoint skeleton selection page selection sensitized Element
checkpoint Solver page sensitized
checkpoint skeleton page info updated
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.New
# Go to the Skeleton Selection page and choose Segment mode
event(Gdk.EventType.BUTTON_PRESS,x= 7.4000000000000e+01,y= 7.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint skeleton selection page grouplist Element
checkpoint page installed Skeleton Selection
findWidget('OOF2:Skeleton Selection Page:Pane').set_position(474)
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
findWidget('OOF2:Skeleton Selection Page:Mode:Segment').clicked()
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
assert tests.sgmtSelectionCheck([])
assert tests.sensitization0()
assert tests.selectionSizeCheck(0)

# Select two elements in the graphics window
findGfxWindow('Graphics_1').simulateMouse('up', -0.16265965, 0.76646632, 1, False, False)
event(Gdk.EventType.BUTTON_PRESS,x= 6.8000000000000e+01,y= 8.0000000000000e+00,button=1,state=0,window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(265)
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
findGfxWindow('Graphics_1').simulateMouse('down', 0.43864262, 0.5981718, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.43513649, 0.5981718, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.43513649, 0.5981718, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 0.43513649, 0.5981718, 1, False, False)
checkpoint Graphics_1 Element sensitized
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint selection info updated Element
checkpoint Graphics_1 Element sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Element.Single_Element
findGfxWindow('Graphics_1').simulateMouse('down', 0.64901077, 0.5981718, 1, True, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.64901077, 0.5981718, 1, True, False)
findGfxWindow('Graphics_1').simulateMouse('up', 0.64901077, 0.5981718, 1, True, False)
checkpoint Graphics_1 Element sensitized
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Element.Single_Element
# Select boundary  segmetns from selected elements
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Select_from_Selected_Elements
assert tests.selectionSizeCheck(6)
assert tests.sensitization1()
assert tests.sgmtSelectionCheck([[11, 12], [11, 16], [12, 13], [13, 18], [16, 17], [17, 18]])
assert tests.groupCheck([])

# Create a group for the selected segments
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:New').clicked()
checkpoint toplevel widget mapped Dialog-Create a new Segment group
findWidget('Dialog-Create a new Segment group').resize(192, 92)
findWidget('Dialog-Create a new Segment group:name').delete_text(0, 11)
findWidget('Dialog-Create a new Segment group:name').insert_text('l', 11)
findWidget('Dialog-Create a new Segment group:name').insert_text('o', 1)
findWidget('Dialog-Create a new Segment group:name').insert_text('o', 2)
findWidget('Dialog-Create a new Segment group:name').insert_text('p', 3)
findWidget('Dialog-Create a new Segment group:name').insert_text(' ', 4)
findWidget('Dialog-Create a new Segment group:name').insert_text('g', 5)
findWidget('Dialog-Create a new Segment group:name').insert_text('r', 6)
findWidget('Dialog-Create a new Segment group:name').insert_text('o', 7)
findWidget('Dialog-Create a new Segment group:name').insert_text('u', 8)
findWidget('Dialog-Create a new Segment group:name').insert_text('p', 9)
findWidget('Dialog-Create a new Segment group:widget_GTK_RESPONSE_OK').clicked()
checkpoint skeleton selection page grouplist Segment
checkpoint OOF.SegmentGroup.New_Group
checkpoint skeleton selection page groups sensitized Segment
assert tests.sensitization2()
assert tests.groupCheck(['loop group (0 segments)'])

# Add the segments to the group
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:Add').clicked()
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint OOF.SegmentGroup.Add_to_Group
assert tests.groupCheck(['loop group (6 segments)'])

# Select the internal segments of the selected elements
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:Select from Selected Elements:internal').clicked()
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:Select from Selected Elements:boundary').clicked()
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:Select from Selected Elements:boundary').clicked()
# Select all the segments of the selected elements
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Select_from_Selected_Elements
assert tests.selectionSizeCheck(7)
assert tests.sgmtSelectionCheck([[11, 12], [11, 16], [12, 13], [12, 17], [13, 18], [16, 17], [17, 18]])
assert tests.sensitization3()

# Select just the internal segments
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:Select from Selected Elements:boundary').clicked()
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Select_from_Selected_Elements
assert tests.selectionSizeCheck(1)
assert tests.sgmtSelectionCheck([[12, 17]])
assert tests.sensitization3()

# Undo the selection
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:Undo').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Undo
# assert tests.selectionSizeCheck(7)
assert tests.sgmtSelectionCheck([[11, 12], [11, 16], [12, 13], [12, 17], [13, 18], [16, 17], [17, 18]])
assert tests.sensitization4()

# Clear the selection
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:Clear').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Clear
assert tests.selectionSizeCheck(0)
assert tests.sgmtSelectionCheck([])
assert tests.sensitization5()

# Clear the element selection
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Element:Clear').clicked()
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Element.Clear
# Get ready to select internal boundary segments
event(Gdk.EventType.BUTTON_PRESS,x= 7.1000000000000e+01,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Select Internal Boundary Segments']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Selection Page:Pane').set_position(466)
# Create a Material display in order to create internal boundaries
findMenu(findWidget('OOF2 Graphics 1:MenuBar'), ['Layer', 'New']).activate()
checkpoint toplevel widget mapped Dialog-New Graphics Layer
findWidget('Dialog-New Graphics Layer').resize(395, 532)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 2.7397260273973e-02)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 1.2328767123288e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 1.7808219178082e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 2.3287671232877e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 2.8767123287671e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 3.2876712328767e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 3.8356164383562e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 3.9726027397260e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 4.1095890410959e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 4.2465753424658e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 4.3835616438356e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 4.5205479452055e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 4.6575342465753e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 4.7945205479452e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 4.9315068493151e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 5.0684931506849e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 5.3424657534247e-01)
findWidget('Dialog-New Graphics Layer:how:Material:no_material:TranslucentGray:gray:slider').get_adjustment().set_value( 5.4794520547945e-01)
findWidget('Dialog-New Graphics Layer:widget_GTK_RESPONSE_OK').clicked()
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.4000000000000e+01)
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint Graphics_1 Element sensitized
checkpoint Graphics_1 Element sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Layer.New
# Select a region of pixels
event(Gdk.EventType.BUTTON_PRESS,x= 8.1000000000000e+01,y= 1.2000000000000e+01,button=1,state=0,window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Pixel Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(249)
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
event(Gdk.EventType.BUTTON_PRESS,x= 7.9000000000000e+01,y= 7.0000000000000e+00,button=1,state=0,window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Selection:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Rectangle']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 4.0000000000000e+01)
findGfxWindow('Graphics_1').simulateMouse('up', -0.10831455, 0.94878537, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('down', -0.027673428, 1.0118958, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', -0.020661157, 1.0118958, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', -0.0031304783, 1.0083897, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.074004508, 0.99436514, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.1616579, 0.96631605, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.22827448, 0.93476083, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.28086652, 0.89619334, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.33696469, 0.84009517, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.41059354, 0.7454295, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.48071625, 0.66128224, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.55083897, 0.59115953, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.60693714, 0.50701227, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.6525169, 0.44039569, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.67355372, 0.39481593, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.68407213, 0.37728525, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.6910844, 0.35975457, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.7016028, 0.34222389, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.71562735, 0.31768094, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.72965189, 0.29664413, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.74367643, 0.28261958, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.75068871, 0.27560731, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.76471325, 0.26158277, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.76821938, 0.2545705, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.76821938, 0.2545705, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.76821938, 0.25106436, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.24755823, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.24405209, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.24054596, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.23703982, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77873779, 0.23353368, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77873779, 0.23002755, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.22652141, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.22652141, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.22652141, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.22301528, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.22301528, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.22301528, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.22652141, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77873779, 0.22652141, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77873779, 0.22652141, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77873779, 0.22652141, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.77523166, 0.23002755, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 0.77523166, 0.23002755, 1, False, False)
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint pixel page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Rectangle
# Create a Material and add it to the selected pixels
event(Gdk.EventType.BUTTON_PRESS,x= 7.4000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Materials']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint Materials page updated
checkpoint page installed Materials
findWidget('OOF2:Materials Page:Pane').set_position(289)
event(Gdk.EventType.BUTTON_RELEASE,x= 1.8200000000000e+02,y= 1.1500000000000e+02,button=1,state=256,window=findWidget('OOF2:Materials Page:Pane:Property:PropertyScroll:PropertyTree').get_window())
findWidget('OOF2:Materials Page:Pane:Material:New').clicked()
checkpoint toplevel widget mapped Dialog-New material
findWidget('Dialog-New material').resize(196, 122)
findWidget('Dialog-New material:widget_GTK_RESPONSE_OK').clicked()
checkpoint Materials page updated
checkpoint OOF.Material.New
findWidget('OOF2:Materials Page:Pane:Material:Assign').clicked()
checkpoint toplevel widget mapped Dialog-Assign material material to pixels
findWidget('Dialog-Assign material material to pixels').resize(235, 122)
event(Gdk.EventType.BUTTON_PRESS,x= 4.7000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('Dialog-Assign material material to pixels:pixels').get_window())
checkpoint toplevel widget mapped chooserPopup-pixels
findWidget('chooserPopup-pixels').deactivate() # MenuLogger
findWidget('Dialog-Assign material material to pixels:widget_GTK_RESPONSE_OK').clicked()
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Material.Assign
# Go back to the Skeleton Selection page
event(Gdk.EventType.BUTTON_PRESS,x= 6.2000000000000e+01,y= 2.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton Selection
# Select internal boundary segments
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Select_Internal_Boundary_Segments
assert tests.sgmtSelectionCheck([[5, 6], [6, 7], [7, 8], [8, 13], [13, 18], [18, 23]])
assert tests.selectionSizeCheck(6)

# Select the segment group
event(Gdk.EventType.BUTTON_PRESS,x= 9.1000000000000e+01,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Select Group']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Selection Page:Pane').set_position(474)
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Select_Group
assert tests.sgmtSelectionCheck([[11, 12], [11, 16], [12, 13], [13, 18], [16, 17], [17, 18]])
assert tests.selectionSizeCheck(6)

# Undo
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:Undo').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Undo
assert tests.sgmtSelectionCheck([[5, 6], [6, 7], [7, 8], [8, 13], [13, 18], [18, 23]])
assert tests.selectionSizeCheck(6)

# Add the internal bdy segments to the group
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:Add').clicked()
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint OOF.SegmentGroup.Add_to_Group
assert tests.groupCheck(['loop group (11 segments)'])
assert tests.sgmtSelectionCheck([[5, 6], [6, 7], [7, 8], [8, 13], [13, 18], [18, 23]])

# Select the group again
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Select_Group
assert tests.sgmtSelectionCheck([[5, 6], [6, 7], [7, 8], [8, 13], [11, 12], [11, 16], [12, 13], [13, 18], [16, 17], [17, 18], [18, 23]])
assert tests.selectionSizeCheck(11)

# Select by homogeneity
event(Gdk.EventType.BUTTON_PRESS,x= 7.5000000000000e+01,y= 1.4000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Select by Homogeneity']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Select_by_Homogeneity
assert tests.sgmtSelectionCheck([[0, 5], [1, 6], [2, 7], [3, 8], [8, 9], [13, 14], [18, 19], [23, 24]])
assert tests.selectionSizeCheck(8)

# Select by homogeneity with a lower threshold
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:Select by Homogeneity:threshold:entry').set_text('0.')
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:Select by Homogeneity:threshold:entry').set_text('0.7')
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:Select by Homogeneity:threshold:entry').set_text('0.75')
widget_0=weakRef(findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:Select by Homogeneity:threshold:entry'))
if widget_0(): wevent(widget_0(), Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_0().get_window())
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Select_by_Homogeneity
assert tests.sgmtSelectionCheck([])
assert tests.selectionSizeCheck(0)

# Invert the selection
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:Invert').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Invert
assert tests.selectionSizeCheck(40)

# Unselect the group
event(Gdk.EventType.BUTTON_PRESS,x= 1.0900000000000e+02,y= 1.3000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Unselect Group']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Unselect_Group
assert tests.selectionSizeCheck(29)

# Clear the segment selecdtion from the toolbox
event(Gdk.EventType.BUTTON_PRESS,x= 8.8000000000000e+01,y= 1.0000000000000e+01,button=1,state=0,window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(265)
checkpoint selection info updated Element
checkpoint Graphics_1 Element sensitized
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Select:Segment').clicked()
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Clear').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Clear
assert tests.selectionSizeCheck(0)
assert tests.sgmtSelectionCheck([])

# Select a single segment in the graphics window
findGfxWindow('Graphics_1').simulateMouse('up', 0.10205359, 0.23703982, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('down', 0.10906587, 0.24755823, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 0.10906587, 0.24755823, 1, False, False)
checkpoint Graphics_1 Segment sensitized
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint selection info updated Segment
checkpoint Graphics_1 Segment sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment
assert tests.selectionSizeCheck(1)
assert tests.sgmtSelectionCheck([[5,6]])

# Select the group w/out unselecting
event(Gdk.EventType.BUTTON_PRESS,x= 9.1000000000000e+01,y= 1.3000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Add Group']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Add_Group
assert tests.sgmtSelectionCheck([[5, 6], [6, 7], [7, 8], [8, 13], [11, 12], [11, 16], [12, 13], [13, 18], [16, 17], [17, 18], [18, 23]])
assert tests.selectionSizeCheck(11)

# Do that again, with a single segment that's *not* in the group
findGfxWindow('Graphics_1').simulateMouse('up', -0.099812171, 0.18821688, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('down', 0.24729527, 0.14263711, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 0.24729527, 0.14263711, 1, False, False)
checkpoint Graphics_1 Segment sensitized
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint selection info updated Segment
checkpoint Graphics_1 Segment sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment
assert tests.sgmtSelectionCheck([[1,6]])
assert tests.selectionSizeCheck(1)

findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Add_Group
assert tests.sgmtSelectionCheck([[1, 6], [5, 6], [6, 7], [7, 8], [8, 13], [11, 12], [11, 16], [12, 13], [13, 18], [16, 17], [17, 18], [18, 23]])
assert tests.selectionSizeCheck(12)

# Clear, and compute the (trivial) intersecction with the group
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Clear').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Clear
event(Gdk.EventType.BUTTON_PRESS,x= 5.4000000000000e+01,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentAction:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Intersect Group']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Intersect_Group
# assert tests.sgmtSelectionCheck([])
assert tests.selectionSizeCheck(0)

# Select a segment in the group
findGfxWindow('Graphics_1').simulateMouse('up', 0.16314801, 0.24755823, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('down', 0.16314801, 0.24755823, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 0.16314801, 0.24755823, 1, False, False)
checkpoint Graphics_1 Segment sensitized
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment
assert tests.sgmtSelectionCheck([[5,6]])
assert tests.selectionSizeCheck(1)

findGfxWindow('Graphics_1').simulateMouse('up', 0.38052842, 0.2545705, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('down', 0.37351615, 0.24755823, 1, True, False)
findGfxWindow('Graphics_1').simulateMouse('up', 0.37351615, 0.24755823, 1, True, False)
checkpoint Graphics_1 Segment sensitized
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment
# Select a thid segment, not in the group
findGfxWindow('Graphics_1').simulateMouse('up', 0.4962309, 0.09679439, 1, True, False)
findGfxWindow('Graphics_1').simulateMouse('down', 0.4962309, 0.09679439, 1, True, False)
findGfxWindow('Graphics_1').simulateMouse('up', 0.4962309, 0.09679439, 1, True, False)
checkpoint Graphics_1 Segment sensitized
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment
# Find the intersection with the group again
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:SegmentHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Intersect_Group
assert tests.sgmtSelectionCheck([[5,6], [6,7]])
assert tests.selectionSizeCheck(2)

# Create a new different size Skeleton
event(Gdk.EventType.BUTTON_PRESS,x= 6.3000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
checkpoint skeleton page sensitized
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:New').clicked()
checkpoint toplevel widget mapped Dialog-New skeleton
findWidget('Dialog-New skeleton').resize(346, 254)
findWidget('Dialog-New skeleton:x_elements').set_text('')
findWidget('Dialog-New skeleton:x_elements').set_text('5')
findWidget('Dialog-New skeleton:y_elements').set_text('')
findWidget('Dialog-New skeleton:y_elements').set_text('5')
findWidget('Dialog-New skeleton:widget_GTK_RESPONSE_OK').clicked()
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.New
# Go back to the selecdtion page
event(Gdk.EventType.BUTTON_PRESS,x= 7.4000000000000e+01,y= 1.4000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton Selection
assert tests.chooserCheck('OOF2:Skeleton Selection Page:Skeleton', ['skeleton', 'skeleton<2>'])
assert tests.chooserStateCheck('OOF2:Skeleton Selection Page:Skeleton', 'skeleton')

# Switch to the new Skeleton
event(Gdk.EventType.BUTTON_PRESS,x= 3.5000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Skeleton').get_window())
checkpoint toplevel widget mapped chooserPopup-Skeleton
findMenu(findWidget('chooserPopup-Skeleton'), ['skeleton<2>']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Skeleton') # MenuItemLogger
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
assert tests.groupCheck([])
assert tests.selectionSizeCheck(0)
assert tests.sgmtSelectionCheck2([])
assert tests.sensitization6()

# Create a group in the new Skeleton
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:New').clicked()
checkpoint toplevel widget mapped Dialog-Create a new Segment group
findWidget('Dialog-Create a new Segment group').resize(192, 92)
findWidget('Dialog-Create a new Segment group:name').delete_text(0, 10)
findWidget('Dialog-Create a new Segment group:name').insert_text('g', 11)
findWidget('Dialog-Create a new Segment group:name').insert_text('r', 1)
findWidget('Dialog-Create a new Segment group:name').insert_text('p', 2)
findWidget('Dialog-Create a new Segment group:widget_GTK_RESPONSE_OK').clicked()
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint OOF.SegmentGroup.New_Group
assert tests.groupCheck(['grp (0 segments)'])
assert tests.sensitization7()

# Switch back to the first skeleton
event(Gdk.EventType.BUTTON_PRESS,x= 6.2000000000000e+01,y= 7.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Skeleton').get_window())
checkpoint toplevel widget mapped chooserPopup-Skeleton
findMenu(findWidget('chooserPopup-Skeleton'), ['skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Skeleton') # MenuItemLogger
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
assert tests.sgmtSelectionCheck([[5, 6], [6, 7]])
assert tests.groupCheck(['loop group (11 segments)'])
assert tests.sensitization3()

# And back to the second Skeleton
event(Gdk.EventType.BUTTON_PRESS,x= 8.2000000000000e+01,y= 1.6000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Skeleton').get_window())
checkpoint toplevel widget mapped chooserPopup-Skeleton
findMenu(findWidget('chooserPopup-Skeleton'), ['skeleton<2>']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Skeleton') # MenuItemLogger
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
assert tests.groupCheck(['grp (0 segments)'])
assert tests.sensitization7()

# Invert the selection in the second skeleton
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:Invert').clicked()
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint selection info updated Segment
checkpoint Graphics_1 Segment sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.SegmentSelection.Invert
assert tests.selectionSizeCheck(60)

# Go to the Skeleton Page
event(Gdk.EventType.BUTTON_PRESS,x= 8.8000000000000e+01,y= 8.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
checkpoint skeleton page sensitized
# Refine the second Skeleton
event(Gdk.EventType.BUTTON_PRESS,x= 1.2600000000000e+02,y= 1.2000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:targets:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['All Elements']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(496)
event(Gdk.EventType.BUTTON_PRESS,x= 6.3000000000000e+01,y= 1.2000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:degree:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Bisection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
# Refine the first Skeleton
event(Gdk.EventType.BUTTON_PRESS,x= 4.9000000000000e+01,y= 1.6000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Skeleton').get_window())
checkpoint toplevel widget mapped chooserPopup-Skeleton
findMenu(findWidget('chooserPopup-Skeleton'), ['skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Skeleton') # MenuItemLogger
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
# Back to the Skeleton Selection page
event(Gdk.EventType.BUTTON_PRESS,x= 5.1000000000000e+01,y= 1.6000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton Selection
assert tests.selectionSizeCheck(120)
assert tests.groupCheck(['grp (0 segments)'])

event(Gdk.EventType.BUTTON_PRESS,x= 8.0000000000000e+01,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Skeleton').get_window())
checkpoint toplevel widget mapped chooserPopup-Skeleton
findMenu(findWidget('chooserPopup-Skeleton'), ['skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Skeleton') # MenuItemLogger
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
assert tests.selectionSizeCheck(4)
assert tests.sgmtSelectionCheck([[30, 52], [31, 52], [31, 57], [32, 57]])
assert tests.groupCheck(['loop group (22 segments)'])

# Delete the first skeleton
event(Gdk.EventType.BUTTON_PRESS,x= 1.1000000000000e+02,y= 2.0000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Delete').clicked()
checkpoint toplevel widget mapped Questioner
findWidget('Questioner').resize(192, 86)
findWidget('Questioner:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 0.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.0000000000000e+00)
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
checkpoint boundary page updated
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton page info updated
checkpoint skeleton selection page selection sensitized Segment
checkpoint Solver page sensitized
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton page info updated
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page updated
checkpoint Field page sensitized
checkpoint skeleton page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Skeleton.Delete
event(Gdk.EventType.BUTTON_PRESS,x= 7.7000000000000e+01,y= 1.6000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton Selection
assert tests.chooserCheck('OOF2:Skeleton Selection Page:Skeleton', ['skeleton<2>'])
assert tests.sensitization8()

# Delete the Microstructure
event(Gdk.EventType.BUTTON_PRESS,x= 7.3000000000000e+01,y= 2.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Microstructure']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Microstructure
checkpoint meshable button set
checkpoint microstructure page sensitized
findWidget('OOF2:Microstructure Page:Delete').clicked()
checkpoint toplevel widget mapped Questioner
findWidget('Questioner').resize(196, 86)
findWidget('Questioner:Yes').clicked()
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 0.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.0000000000000e+00)
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint Graphics_1 Element sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint contourmap info updated for Graphics_1
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 0.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.0000000000000e+00)
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint selection info updated Element
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint Graphics_1 Element sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 0.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.0000000000000e+00)
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
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page updated
checkpoint Solver page sensitized
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint Field page sensitized
checkpoint Solver page sensitized
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 0.0000000000000e+00)
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint Graphics_1 Element sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint pixel page updated
checkpoint active area status updated
checkpoint pixel page sensitized
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint Materials page updated
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint boundary page updated
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page updated
checkpoint Solver page sensitized
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Microstructure.Delete
findWidget('OOF2:Microstructure Page:Pane').set_position(184)
event(Gdk.EventType.BUTTON_PRESS,x= 5.0000000000000e+01,y= 2.0000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton Selection
assert tests.sensitization9()
assert tests.selectionSizeCheck(None)

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
findWidget('Dialog-Python_Log:filename').set_text('session.lg')
findWidget('Dialog-Python_Log:filename').set_text('session.lpg')
findWidget('Dialog-Python_Log:filename').set_text('session.lg')
findWidget('Dialog-Python_Log:filename').set_text('session.log')
findWidget('Dialog-Python_Log').resize(194, 122)
findWidget('Dialog-Python_Log:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.File.Save.Python_Log
assert tests.filediff('session.log')

findMenu(findWidget('OOF2:MenuBar'), ['File', 'Quit']).activate()
checkpoint OOF.Graphics_1.File.Close