# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

import tests
tbox = "OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Mesh Info"

findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
findWidget('OOF2').resize(782, 511)

findMenu(findWidget('OOF2:MenuBar'), ['Settings', 'Graphics_Defaults', 'New_Layer_Policy']).activate()
checkpoint toplevel widget mapped Dialog-New_Layer_Policy
findWidget('Dialog-New_Layer_Policy').resize(192, 86)
findWidget('OOF2').resize(782, 545)
wevent(findWidget('Dialog-New_Layer_Policy:policy'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('Dialog-New_Layer_Policy:policy').get_window())
checkpoint toplevel widget mapped chooserPopup-policy
findMenu(findWidget('chooserPopup-policy'), ['Single']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-policy') # MenuItemLogger
findWidget('Dialog-New_Layer_Policy:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Settings.Graphics_Defaults.New_Layer_Policy
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

# Open the Mesh Info toolbox
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Mesh Info']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
checkpoint Graphics_1 Mesh Info sensitized
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(227)
assert not findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'','Y':''},tbox+":Click")
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':False,'Clear':False,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'','type':'','material':''},tbox+":ElementInfo")
assert tests.chooserListCheck(tbox+':ElementInfo:NodeList',[])

# Load a Mesh
findMenu(findWidget('OOF2:MenuBar'), ['File', 'Load', 'Data']).activate()
checkpoint toplevel widget mapped Dialog-Data
findWidget('Dialog-Data').resize(192, 92)
findWidget('Dialog-Data:filename').set_text('T')
findWidget('Dialog-Data:filename').set_text('TE')
findWidget('Dialog-Data:filename').set_text('TES')
findWidget('Dialog-Data:filename').set_text('TEST')
findWidget('Dialog-Data:filename').set_text('TEST_')
findWidget('Dialog-Data:filename').set_text('TEST_D')
findWidget('Dialog-Data:filename').set_text('TEST_DA')
findWidget('Dialog-Data:filename').set_text('TEST_DAT')
findWidget('Dialog-Data:filename').set_text('TEST_DATA')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/m')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/me')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/mes')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/mesh')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/meshi')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/meshin')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/meshinf')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/meshinfo')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/meshinfot')
findWidget('Dialog-Data:filename').set_text('TEST_DATA/meshinfotbox.mesh')
findWidget('Dialog-Data:widget_GTK_RESPONSE_OK').clicked()
checkpoint meshable button set
checkpoint microstructure page sensitized
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
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized Element
checkpoint Solver page sensitized
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint microstructure page sensitized
checkpoint meshable button set
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint Materials page updated
checkpoint Materials page updated
checkpoint Materials page updated
checkpoint Materials page updated
checkpoint Materials page updated
checkpoint Materials page updated
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized Element
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
checkpoint contourmap info updated for Graphics_1
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint boundary page updated
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page selection sensitized Element
checkpoint Solver page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
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
checkpoint Graphics_1 Mesh Info cleared position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint mesh bdy page updated
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Solver page sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Solver page sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Solver page sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint Graphics_1 Mesh Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.File.Load.Data
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'','Y':''},tbox+":Click")
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':False,'Clear':False,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'','type':'','material':''},tbox+":ElementInfo")
assert tests.chooserListCheck(tbox+':ElementInfo:NodeList',[])

# Zoom to fill, then go back to the Mesh Info toolbox
findWidget('OOF2 Graphics 1').resize(800, 492)
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Viewer']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(212)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 2.7000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Viewer:Zoom:Fill').clicked()
checkpoint OOF.Graphics_1.Settings.Zoom.Fill_Window
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Mesh Info']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
checkpoint Graphics_1 Mesh Info sensitized
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(227)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 0.0000000000000e+00)

# Query an Element
findGfxWindow('Graphics_1').simulateMouse('down', 17.024793, 95.110193, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 17.024793, 95.110193, 1, False, False)
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'15.321','Y':'86.7764'},tbox+":Click")
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':False,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'81','type':'Q4_4','material':'bounce'},tbox+":ElementInfo")
assert tests.chooserListCheck(tbox+':ElementInfo:NodeList',["FuncNode 89 at (10, 80)","FuncNode 90 at (20, 80)","FuncNode 101 at (20, 90)","FuncNode 100 at (10, 90)"],tolerance=1.e-6)

# Another Element query
findGfxWindow('Graphics_1').simulateMouse('scroll', 1, -0, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('scroll', 2, -0, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('scroll', 5, -0, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('scroll', 1, -0, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 26.280992, 95.495868, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('scroll', -0, -0, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('down', 26.280992, 95.495868, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 26.280992, 95.495868, 1, False, False)
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'24.9938','Y':'87.4629'},tbox+":Click")
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'82','type':'Q4_4','material':'bounce'},tbox+":ElementInfo")
assert tests.chooserListCheck(tbox+':ElementInfo:NodeList',["FuncNode 90 at (20, 80)","FuncNode 91 at (30, 80)","FuncNode 102 at (30, 90)","FuncNode 101 at (20, 90)"],tolerance=1.e-6)

# Switch to node mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Mesh Info:Click:Node').clicked()
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'24.9938','Y':'87.4629'},tbox+":Click")
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'101','type':'FuncNode','position':'(20, 90)'},tbox+":NodeInfo",tolerance=1.e-6)

# Query a node
findGfxWindow('Graphics_1').simulateMouse('down', 20.881543, 87.782369, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 20.881543, 87.782369, 1, False, False)
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'19.7376','Y':'80.5132'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'90','type':'FuncNode','position':'(20, 80)'},tbox+":NodeInfo",tolerance=1.e-6)

# Previous
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 6.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.6000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.4000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.5000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 4.6000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 5.9000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 7.1000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 7.8000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 8.9000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 9.9000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.0800000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.1500000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.1900000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.2400000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.2900000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.3600000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.4000000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.4200000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.4500000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.4900000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.5300000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.5600000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.5900000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.6000000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.6200000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Mesh Info:Prev').clicked()
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint contourmap info updated for Graphics_1
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'24.9938','Y':'87.4629'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.gtkMultiTextCompare({'index':'101','type':'FuncNode','position':'(20, 90)'},tbox+":NodeInfo",tolerance=1.e-6)

# Previous again.
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Mesh Info:Prev').clicked()
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint contourmap info updated for Graphics_1
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 5.6000000000000e+01)
# "Previous" took us to element mode.  Note that the switch to element
# mode there was implicit, not from the user.
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'24.9938','Y':'87.4629'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.gtkMultiTextCompare({'index':'82','type':'Q4_4','material':'bounce'},tbox+":ElementInfo")
assert tests.chooserListCheck(tbox+':ElementInfo:NodeList',["FuncNode 90 at (20, 80)","FuncNode 91 at (30, 80)","FuncNode 102 at (30, 90)","FuncNode 101 at (20, 90)"],tolerance=1.e-6)

# Clear
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Mesh Info:Clear').clicked()
checkpoint Graphics_1 Mesh Info cleared position
checkpoint Graphics_1 Mesh Info cleared
checkpoint contourmap info updated for Graphics_1
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.0000000000000e+00)
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'','Y':''},tbox+":Click")
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.gtkMultiTextCompare({'index':'','type':'','material':''},tbox+":ElementInfo")
assert tests.chooserListCheck(tbox+':ElementInfo:NodeList',[])

# Element query
findGfxWindow('Graphics_1').simulateMouse('down', 17.410468, 74.669421, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 17.410468, 74.669421, 1, False, False)
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'16.6842','Y':'68.2923'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'61','type':'Q4_4','material':'bounce'},tbox+":ElementInfo")
assert tests.chooserListCheck(tbox+':ElementInfo:NodeList',["FuncNode 67 at (10, 60)","FuncNode 68 at (20, 60)","FuncNode 79 at (20, 70)","FuncNode 78 at (10, 70)"],tolerance=1.e-6)

# Select a node in the node list
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Mesh Info:ElementInfo:NodeList').get_selection().select_path(Gtk.TreePath([1]))
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 5.0000000000000e+00)
checkpoint contourmap info updated for Graphics_1
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'16.6842','Y':'68.2923'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'61','type':'Q4_4','material':'bounce'},tbox+":ElementInfo")
assert tests.chooserListCheck(tbox+':ElementInfo:NodeList',["FuncNode 67 at (10, 60)","FuncNode 68 at (20, 60)","FuncNode 79 at (20, 70)","FuncNode 78 at (10, 70)"],tolerance=1.e-6)
assert tests.chooserListStateCheck(tbox+':ElementInfo:NodeList',["FuncNode 68 at (20, 60)"],tolerance=1.e-6)

# Double click on the node in the node list
tree=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Mesh Info:ElementInfo:NodeList')
column = tree.get_column(0)
tree.row_activated(Gtk.TreePath([1]), column)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 1.0000000000000e+00)
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'20','Y':'60'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'68','type':'FuncNode','position':'(20, 60)'},tbox+":NodeInfo",tolerance=1.e-6)

# Open a data viewer
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.0000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.9000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.8000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.7000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 4.6000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 5.4000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 6.2000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 7.2000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 8.2000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 9.0000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.0300000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.1700000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.3100000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.4500000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.5900000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.6200000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Mesh Info:NewDataViewer').clicked()
checkpoint Mesh_Data_1 position updated
checkpoint Mesh_Data_1 time updated
checkpoint Mesh_Data_1 mesh updated
checkpoint toplevel widget mapped Mesh Data 1
findWidget('Mesh Data 1').resize(278, 372)
checkpoint Mesh_Data_1 data updated
findWidget('Mesh Data 1').resize(278, 448)
assert tests.gtkMultiTextCompare({'x':'0.440861','y':'5.59887'},'Mesh Data 1:Data',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'20','y':'60'},'Mesh Data 1:ViewSource',tolerance=1.e-6)
assert not findWidget('Mesh Data 1:Freeze:Space').get_active()

# Query a node
findGfxWindow('Graphics_1').simulateMouse('down', 20.110193, 76.597796, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 20.110193, 76.597796, 1, False, False)
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint Mesh_Data_1 position updated
checkpoint Mesh_Data_1 data updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'19.4543','Y':'70.1964'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'79','type':'FuncNode','position':'(20, 70)'},tbox+":NodeInfo",tolerance=1.e-6)

# Query another node
findGfxWindow('Graphics_1').simulateMouse('down', 28.980716, 76.983471, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 28.980716, 76.983471, 1, False, False)
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint Mesh_Data_1 position updated
checkpoint Mesh_Data_1 data updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'28.4917','Y':'70.8268'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'80','type':'FuncNode','position':'(30, 70)'},tbox+":NodeInfo",tolerance=1.e-6)

assert tests.gtkMultiTextCompare({'x':'0.489024','y':'6.15665'},'Mesh Data 1:Data',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'28.4917','y':'70.8268'},'Mesh Data 1:ViewSource',tolerance=1.e-6)
assert not findWidget('Mesh Data 1:Freeze:Space').get_active()

# Freeze the data viewer
findWidget('Mesh Data 1:Freeze:Space').clicked()
assert tests.gtkMultiTextCompare({'x':'0.489024','y':'6.15665'},'Mesh Data 1:Data',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'28.4917','y':'70.8268'},'Mesh Data 1:ViewSource',tolerance=1.e-6)
assert findWidget('Mesh Data 1:Freeze:Space').get_active()
assert tests.is_sensitive('Mesh Data 1:Close')

# Open second data viewer
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Mesh Info:NewDataViewer').clicked()
checkpoint Mesh_Data_2 position updated
checkpoint Mesh_Data_2 time updated
checkpoint Mesh_Data_2 mesh updated
checkpoint toplevel widget mapped Mesh Data 2
findWidget('Mesh Data 2').resize(278, 372)
checkpoint Mesh_Data_2 data updated
findWidget('Mesh Data 2').resize(278, 448)
assert tests.gtkMultiTextCompare({'X':'28.4917','Y':'70.8268'},tbox+":Click",tolerance=1.e-6)

assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'28.4917','y':'70.8268'},'Mesh Data 1:ViewSource',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'x':'0.489024','y':'6.15665'},'Mesh Data 1:Data',tolerance=1.e-6)
assert findWidget('Mesh Data 1:Freeze:Space').get_active()
assert tests.is_sensitive('Mesh Data 1:Close')

assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'28.4917','y':'70.8268'},'Mesh Data 2:ViewSource',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'x':'0.489024','y':'6.15665'},'Mesh Data 2:Data',tolerance=1.e-6)
assert not findWidget('Mesh Data 2:Freeze:Space').get_active()
assert tests.is_sensitive('Mesh Data 2:Close')

# Query a node
findGfxWindow('Graphics_1').simulateMouse('down', 30.137741, 86.625344, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 30.137741, 86.625344, 1, False, False)
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint Mesh_Data_2 position updated
checkpoint Mesh_Data_2 data updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'29.3834','Y':'79.9069'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'91','type':'FuncNode','position':'(30, 80)'},tbox+":NodeInfo",tolerance=1.e-6)
# Mesh Data 1 is frozen and hasn't changed
assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'28.4917','y':'70.8268'},'Mesh Data 1:ViewSource',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'x':'0.489024','y':'6.15665'},'Mesh Data 1:Data',tolerance=1.e-6)
assert findWidget('Mesh Data 1:Freeze:Space').get_active()
assert tests.is_sensitive('Mesh Data 1:Close')
# Mesh Data 2 has updated
assert tests.gtkMultiTextCompare({'x':'0.754334','y':'6.71843'},'Mesh Data 2:Data',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'29.3834','y':'79.9069'},'Mesh Data 2:ViewSource',tolerance=1.e-6)
assert not findWidget('Mesh Data 2:Freeze:Space').get_active()
assert tests.is_sensitive('Mesh Data 1:Close')

# Query another node
findGfxWindow('Graphics_1').simulateMouse('up', 49.421488, 76.212121, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('down', 49.421488, 75.826446, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 49.421488, 75.826446, 1, False, False)
checkpoint Graphics_1 Mesh Info showed position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Mesh Info updated
checkpoint Mesh_Data_2 position updated
checkpoint Mesh_Data_2 data updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Mesh_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'49.4079','Y':'69.9505'},tbox+":Click",tolerance=1.e-6)
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'82','type':'FuncNode','position':'(50, 70)'},tbox+":NodeInfo",tolerance=1.e-6)
# Mesh Data 1 is frozen and hasn't changed
assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'28.4917','y':'70.8268'},'Mesh Data 1:ViewSource',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'x':'0.489024','y':'6.15665'},'Mesh Data 1:Data',tolerance=1.e-6)
assert findWidget('Mesh Data 1:Freeze:Space').get_active()
assert tests.is_sensitive('Mesh Data 1:Close')
# Mesh Data 2 has updated
assert tests.gtkMultiTextCompare({'x':'0.0135587','y':'5.87597'},'Mesh Data 2:Data',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'49.4079','y':'69.9505'},'Mesh Data 2:ViewSource',tolerance=1.e-6)
assert not findWidget('Mesh Data 2:Freeze:Space').get_active()

# Close the second mesh data viewer.
findWidget('Mesh Data 2:Close').clicked()
assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'28.4917','y':'70.8268'},'Mesh Data 1:ViewSource',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'x':'0.489024','y':'6.15665'},'Mesh Data 1:Data',tolerance=1.e-6)
assert findWidget('Mesh Data 1:Freeze:Space').get_active()
assert tests.is_sensitive('Mesh Data 1:Close')
checkpoint_count("Graphics_1 Mesh Info showed position")

# Unfreeze the first data viewer
findWidget('Mesh Data 1:Freeze:Space').clicked()
checkpoint Mesh_Data_1 position updated
checkpoint Mesh_Data_1 data updated
# Note that it retains its old data...
assert tests.gtkMultiTextCompare({'meshname':'microstructure:skeleton:mesh','x':'28.4917','y':'70.8268'},'Mesh Data 1:ViewSource',tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'x':'0.489024','y':'6.15665'},'Mesh Data 1:Data',tolerance=1.e-6)
assert not findWidget('Mesh Data 1:Freeze:Space').get_active()
assert tests.is_sensitive('Mesh Data 1:Close')

# Remove the mesh layer from the graphics window
wevent(findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList'), Gdk.EventType.BUTTON_PRESS, button=3, state=0, window=findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList').get_window())
checkpoint toplevel widget mapped PopUp-0
findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList').get_selection().select_path(Gtk.TreePath([10]))
checkpoint OOF.Graphics_1.Layer.Select
findMenu(findWidget('PopUp-0'), ['Delete']).activate() # MenuItemLogger
deactivatePopup('PopUp-0') # MenuItemLogger
checkpoint Graphics_1 Mesh Info cleared position
checkpoint Graphics_1 Mesh Info sensitized
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint Mesh_Data_1 time updated
checkpoint Mesh_Data_1 mesh updated
checkpoint Mesh_Data_1 data updated
checkpoint Mesh_Data_1 data updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Layer.Delete
# checkpoint_count("Graphics_1 Mesh Info showed position")
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+':Click:Node').get_active()
assert tests.gtkMultiTextCompare({'X':'','Y':''},tbox+":Click")
assert tests.sensitizationCheck({'NewDataViewer':True,'Prev':False,'Clear':False,'Next':False},tbox)
assert tests.gtkMultiTextCompare({'index':'','type':'','position':''},tbox+":NodeInfo")

# There should be no widgets inside the "Data" frame.  Length is 1
# because findAllWidgets reports the parent as one of the results.
assert len(findAllWidgets('Mesh Data 1:Data'))==1
assert tests.gtkMultiTextCompare({'meshname':'<No Mesh in window!>','x':'28.4917','y':'70.8268'},'Mesh Data 1:ViewSource',tolerance=1.e-6)
assert not findWidget('Mesh Data 1:Freeze:Space').get_active()
assert tests.is_sensitive('Mesh Data 1:Close')

# Close the first graphics window
findWidget('Mesh Data 1:Close').clicked()
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
assert tests.filediff("session.log", tolerance=1.e-6)

findMenu(findWidget('OOF2:MenuBar'), ['File', 'Quit']).activate()
checkpoint OOF.Graphics_1.File.Close