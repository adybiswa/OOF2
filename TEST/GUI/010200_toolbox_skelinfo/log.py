# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

import tests
tbox = "OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info"
elbox = tbox+":ElementInformation"
ndbox = tbox+":NodeInformation"
sgbox = tbox+":SegmentInformation"
cbox = tbox+":Click"

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
findMenu(findWidget('OOF2 Graphics 1:MenuBar'), ['Settings', 'New_Layer_Policy']).activate()
checkpoint toplevel widget mapped Dialog-New_Layer_Policy
findWidget('Dialog-New_Layer_Policy').resize(192, 86)
wevent(findWidget('Dialog-New_Layer_Policy:policy'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('Dialog-New_Layer_Policy:policy').get_window())
checkpoint toplevel widget mapped chooserPopup-policy
findMenu(findWidget('chooserPopup-policy'), ['Single']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-policy') # MenuItemLogger
findWidget('Dialog-New_Layer_Policy:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Graphics_1.Settings.New_Layer_Policy
findWidget('OOF2 Graphics 1').resize(800, 492)
# Open the skeleton info toolbox
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Skeleton Info']).activate() # MenuItemLogger
checkpoint Graphics_1 Skeleton Info sensitized
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(243)
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'','Y Text':''},cbox)
assert tests.gtkMultiTextCompare({'Material':'','Group':'','Shape':'','Homog':'','Dom pixel':'','Area':'','Index':'','Type':''},elbox)
assert tests.sensitizationCheck({'Prev':False,'Clear':False,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",[])
assert tests.chooserListCheck(elbox+":NodeList",[])

# Load a Skeleton
findWidget('OOF2:Introduction Page:Scroll').get_vadjustment().set_value( 1.0000000000000e+00)
findWidget('OOF2:Introduction Page:Scroll').get_vadjustment().set_value( 6.0000000000000e+00)
findWidget('OOF2:Introduction Page:Scroll').get_vadjustment().set_value( 9.0000000000000e+00)
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
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
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
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint Solver page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
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
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
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
# Load a Skeleton
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'','Y Text':''},cbox)
assert tests.gtkMultiTextCompare({'Material':'','Group':'','Shape':'','Homog':'','Dom pixel':'','Area':'','Index':'','Type':''},elbox)
assert tests.sensitizationCheck({'Prev':False,'Clear':False,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",[])

# Rename the pixel groups to reduce verbosity
wevent(findWidget('OOF2:Navigation:PageMenu'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Microstructure']).activate() # MenuItemLogger
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint page installed Microstructure
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
findWidget('OOF2:Microstructure Page:Pane').set_position(189)
findWidget('OOF2:Microstructure Page:Pane:PixelGroups:Rename').clicked()
checkpoint toplevel widget mapped Dialog-Rename pixelgroup RGBColor(red=0.000000, green=0.000000, blue=1.000000)
findWidget('Dialog-Rename pixelgroup RGBColor(red=0.000000, green=0.000000, blue=1.000000)').resize(192, 92)
findWidget('Dialog-Rename pixelgroup RGBColor(red=0.000000, green=0.000000, blue=1.000000):new_name').set_text('')
findWidget('Dialog-Rename pixelgroup RGBColor(red=0.000000, green=0.000000, blue=1.000000):new_name').set_text('b')
findWidget('Dialog-Rename pixelgroup RGBColor(red=0.000000, green=0.000000, blue=1.000000):new_name').set_text('bl')
findWidget('Dialog-Rename pixelgroup RGBColor(red=0.000000, green=0.000000, blue=1.000000):new_name').set_text('blu')
findWidget('Dialog-Rename pixelgroup RGBColor(red=0.000000, green=0.000000, blue=1.000000):new_name').set_text('blue')
findWidget('Dialog-Rename pixelgroup RGBColor(red=0.000000, green=0.000000, blue=1.000000):widget_GTK_RESPONSE_OK').clicked()
checkpoint microstructure page sensitized
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint OOF.PixelGroup.Rename
findWidget('OOF2:Microstructure Page:Pane:PixelGroups:Stack:GroupListScroll:GroupList').get_selection().select_path(Gtk.TreePath([0]))
checkpoint microstructure page sensitized
checkpoint meshable button set
findWidget('OOF2:Microstructure Page:Pane:PixelGroups:Rename').clicked()
checkpoint toplevel widget mapped Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941)
findWidget('Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941)').resize(192, 92)
findWidget('Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941):new_name').set_text('')
findWidget('Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941):new_name').set_text('y')
findWidget('Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941):new_name').set_text('ye')
findWidget('Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941):new_name').set_text('yel')
findWidget('Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941):new_name').set_text('yell')
findWidget('Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941):new_name').set_text('yello')
findWidget('Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941):new_name').set_text('yellow')
findWidget('Dialog-Rename pixelgroup RGBColor(red=1.000000, green=1.000000, blue=0.752941):widget_GTK_RESPONSE_OK').clicked()
checkpoint microstructure page sensitized
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint OOF.PixelGroup.Rename
# Query an element
findGfxWindow('Graphics_1').simulateMouse('down', 31.975, 56.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 31.975, 56.3, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':31.98,'Y Text':56.3},cbox)
assert tests.gtkMultiTextCompare({'Material':'<No material>','Group':'','Dom pixel':'blue','Index':'59','Type':'quad'},elbox)
assert tests.gtkMultiFloatCompare({'Shape':0.418601,'Homog':0.992461,'Area':137.5},elbox)
assert tests.sensitizationCheck({'Prev':False,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",["Segment 135, nodes (36, 44) (length: 15.4029218007)","Segment 134, nodes (36, 37) (length: 15.5)","Segment 70, nodes (37, 45) (length: 12.5)","Segment 128, nodes (44, 45) (length: 6.5)"], tolerance=1.e-6)
assert tests.chooserListCheck(elbox+":NodeList",["Node 36 at (22.0, 50.0) (angle: 54.2461127456)","Node 37 at (37.5, 50.0) (angle: 90)","Node 45 at (37.5, 62.5) (angle: 90)","Node 44 at (31.0, 62.5) (angle: 125.753887254)"], tolerance=1.e-6)

# Query another element
findGfxWindow('Graphics_1').simulateMouse('down', 21.125, 59.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 21.125, 59.1, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':21.12,'Y Text':59.1},cbox)
assert tests.gtkMultiTextCompare({'Material':'<No material>','Group':'','Dom pixel':'yellow','Index':'60','Type':'triangle'},elbox)
assert tests.gtkMultiFloatCompare({'Shape':0.030177,'Homog':0.975899,'Area':115.625000},elbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",["Segment 68, nodes (36, 43) (length: 15.7003184681)", "Segment 135, nodes (36, 44) (length: 15.4029218007)", "Segment 130, nodes (43, 44) (length: 18.5)"], tolerance=1.e-6)
assert tests.chooserListCheck(elbox+":NodeList",["Node 36 at (22.0, 50.0) (angle: 72.988721236)","Node 44 at (31.0, 62.5) (angle: 54.2461127456)", "Node 43 at (12.5, 62.5) (angle: 52.7651660184)"], tolerance=1.e-6)

# Switch to node mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Node').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':21.12,'Y Text':59.1},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Group':'','Mobility':'free','Position':'(22.0, 50.0)','Index':'36'},ndbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",["Element 22", "Element 26", "Element 56", "Element 58", "Element 59", "Element 60"])

# Click on a node
findGfxWindow('Graphics_1').simulateMouse('down', 12.725, 62.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 12.725, 62.6, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':12.72,'Y Text':62.6},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Group':'','Mobility':'free','Position':'(12.5, 62.5)','Index':'43'},ndbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",["Element 26","Element 30","Element 54","Element 60"])

# Click on a different node
findGfxWindow('Graphics_1').simulateMouse('down', 12.375, 75.2, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 12.375, 75.2, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':12.38,'Y Text':75.2},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Group':'','Mobility':'free','Position':'(12.5, 75.0)','Index':'52'},ndbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",["Element 30","Element 35","Element 54","Element 55"])

# Switch to segment mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 4.8000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Segment').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QuerySegment
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':12.38,'Y Text':75.2},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Groups':'','Index':'93'},sgbox)
assert tests.gtkMultiFloatCompare({'Length':12.5},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList", ['Element 35', 'Element 55'])
assert tests.chooserListCheck(sgbox+":NodeList", ['Node 52 at (12.5, 75.0)', 'Node 59 at (12.5, 87.5)'], tolerance=1.e-6)

# Click on a segment
findGfxWindow('Graphics_1').simulateMouse('down', 21.125, 62.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 21.125, 62.95, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QuerySegment
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':21.12,'Y Text':62.95},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Groups':'','Index':'130'},sgbox)
assert tests.gtkMultiFloatCompare({'Length':18.5},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 54","Element 60"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 43 at (12.5, 62.5)","Node 44 at (31.0, 62.5)"], tolerance=1.e-6)

findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 4.9000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 5.0000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 6.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.0000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 4.2000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 0.0000000000000e+00)

# Click on another segment
findGfxWindow('Graphics_1').simulateMouse('down', 10.975, 69.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 10.975, 69.25, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QuerySegment
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':10.97,'Y Text':69.25},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Groups':'','Index':'79'},sgbox)
assert tests.gtkMultiFloatCompare({'Length':12.5},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 30","Element 54"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 43 at (12.5, 62.5)","Node 52 at (12.5, 75.0)"], tolerance=1.e-6)

# Switch to element mode
findWidget('OOF2 Graphics 1').resize(800, 492)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Element').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':10.97,'Y Text':69.25},cbox)
assert tests.gtkMultiTextCompare({'Material':'<No material>','Group':'','Dom pixel':'yellow','Index':'30','Type':'quad'},elbox)
assert tests.gtkMultiFloatCompare({'Shape':0.0,'Homog':1.0,'Area':156.25},elbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",['Segment 78, nodes (42, 51) (length: 12.5)', 'Segment 69, nodes (42, 43) (length: 12.5)', 'Segment 79, nodes (43, 52) (length: 12.5)', 'Segment 80, nodes (51, 52) (length: 12.5)'], tolerance=1.e-6)
assert tests.chooserListCheck(elbox+":NodeList",['Node 42 at (0.0, 62.5) (angle: 90.0)', 'Node 43 at (12.5, 62.5) (angle: 90.0)', 'Node 52 at (12.5, 75.0) (angle: 90.0)', 'Node 51 at (0.0, 75.0) (angle: 90.0)'], tolerance=1.e-6)

# Select elements by dominant pixel and put them in a group
findWidget('OOF2 Graphics 1').resize(800, 492)
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Skeleton Selection']).activate() # MenuItemLogger
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(265)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 0.0000000000000e+00)
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Element:Method:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Element:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['ByDominantPixel']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 2.4000000000000e+01)
findGfxWindow('Graphics_1').simulateMouse('down', 41.10125, 55.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 41.10125, 55.95, 1, False, False)
checkpoint Graphics_1 Element sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Element.ByDominantPixel
wevent(findWidget('OOF2:Navigation:PageMenu'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
checkpoint skeleton page sensitized
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint page installed Skeleton
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(417)
wevent(findWidget('OOF2:Navigation:PageMenu'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint page installed Skeleton Selection
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
findWidget('OOF2:Skeleton Selection Page:Pane').set_position(474)
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:New').clicked()
checkpoint toplevel widget mapped Dialog-Create a new Element group
findWidget('Dialog-Create a new Element group').resize(192, 92)
findWidget('Dialog-Create a new Element group:name').delete_text(0, 11)
findWidget('Dialog-Create a new Element group:name').insert_text('e', 11)
findWidget('Dialog-Create a new Element group:name').insert_text('l', 1)
findWidget('Dialog-Create a new Element group:name').insert_text('s', 2)
findWidget('Dialog-Create a new Element group:name').insert_text('e', 3)
findWidget('Dialog-Create a new Element group:name').insert_text('t', 4)
findWidget('Dialog-Create a new Element group:widget_GTK_RESPONSE_OK').clicked()
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page grouplist Element
checkpoint OOF.ElementGroup.New_Group
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:Add').clicked()
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page groups sensitized Element
checkpoint OOF.ElementGroup.Add_to_Group
# Clear the elemetn selection
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Element:Clear').clicked()
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint Graphics_1 Element sensitized
checkpoint selection info updated Element
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Element.Clear
# Select a circle of nodes
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Select:Node').clicked()
checkpoint Graphics_1 Node sensitized
checkpoint selection info updated Node
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Node:Method:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Node:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Circle']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findGfxWindow('Graphics_1').simulateMouse('down', 49.50125, 99.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 49.50125, 99, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 49.50125, 98.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 49.15125, 96.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 48.80125, 94.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 48.80125, 92.7, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 48.45125, 90.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 48.10125, 89.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 48.10125, 88.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 48.10125, 87.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 48.10125, 86.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 47.75125, 85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 48.10125, 82.9, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 48.80125, 80.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 49.50125, 79.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 49.85125, 77.65, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 50.55125, 76.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 50.90125, 74.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.60125, 72.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.60125, 71, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.60125, 70.3, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.95125, 69.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.95125, 69.95, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.95125, 69.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.95125, 69.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 68.55, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 67.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 67.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 67.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 67.5, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 67.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 67.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 67.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 67.15, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.65125, 66.8, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.65125, 66.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.65125, 66.45, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.00125, 66.1, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 65.75, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.70125, 65.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.70125, 65.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.05125, 65.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.05125, 64.7, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.40125, 64.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.75125, 64.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.75125, 64.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.75125, 64.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.75125, 64.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.75125, 64.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 54.75125, 64.35, 1, False, False)
checkpoint Graphics_1 Node sensitized
checkpoint Graphics_1 Node sensitized
checkpoint selection info updated Node
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Node.Circle
# Put nodes in a group
findWidget('OOF2:Skeleton Selection Page:Mode:Node').clicked()
checkpoint skeleton selection page groups sensitized Node
checkpoint skeleton selection page grouplist Node
checkpoint skeleton selection page selection sensitized Node
checkpoint skeleton selection page updated
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:New').clicked()
checkpoint toplevel widget mapped Dialog-Create a new Node group
findWidget('Dialog-Create a new Node group').resize(192, 92)
findWidget('Dialog-Create a new Node group:name').delete_text(0, 11)
findWidget('Dialog-Create a new Node group:name').insert_text('n', 11)
findWidget('Dialog-Create a new Node group:name').insert_text('d', 1)
findWidget('Dialog-Create a new Node group:name').insert_text('s', 2)
findWidget('Dialog-Create a new Node group:name').insert_text('e', 3)
findWidget('Dialog-Create a new Node group:name').insert_text('t', 4)
findWidget('Dialog-Create a new Node group:widget_GTK_RESPONSE_OK').clicked()
checkpoint skeleton selection page groups sensitized Node
checkpoint skeleton selection page grouplist Node
checkpoint OOF.NodeGroup.New_Group
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:Add').clicked()
checkpoint skeleton selection page grouplist Node
checkpoint skeleton selection page groups sensitized Node
checkpoint OOF.NodeGroup.Add_to_Group
# Clear node selection
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Node:Clear').clicked()
checkpoint skeleton selection page groups sensitized Node
checkpoint skeleton selection page selection sensitized Node
checkpoint skeleton selection page updated
checkpoint Graphics_1 Node sensitized
checkpoint selection info updated Node
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Node.Clear
# Select segments and put them in a group
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Select:Segment').clicked()
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Method:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Circle']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findGfxWindow('Graphics_1').simulateMouse('down', 50.55125, 90.90625, 1, False, False)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:Canvas:vscroll').get_adjustment().set_value( 1.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:Canvas:vscroll').get_adjustment().set_value( 3.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:Canvas:vscroll').get_adjustment().set_value( 6.0000000000000e+00)
findGfxWindow('Graphics_1').simulateMouse('move', 50.55125, 88.45625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 50.90125, 87.75625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.25125, 85.65625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.60125, 83.90625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 51.95125, 82.85625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 82.15625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 81.45625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.30125, 81.10625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.65125, 80.05625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.65125, 79.00625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 52.65125, 77.95625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.00125, 77.60625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.00125, 76.90625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.00125, 76.20625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.00125, 75.50625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 74.80625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 73.75625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 71.65625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.05125, 69.90625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.05125, 68.50625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.40125, 67.10625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.40125, 66.75625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.75125, 65.35625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.10125, 64.30625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 63.60625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.90625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.55625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.55625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.55625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.55625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.55625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.55625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.55625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.20625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.45125, 62.20625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.10125, 62.20625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 55.10125, 61.85625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.75125, 60.80625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.75125, 59.40625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.40125, 58.35625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 54.05125, 57.30625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.70125, 56.25625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.70125, 55.55625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.70125, 54.85625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.70125, 54.85625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 54.85625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 54.50625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 54.50625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 54.15625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 54.15625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 54.15625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 53.35125, 54.15625, 1, False, False)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:Canvas:vscroll').get_adjustment().set_value( 5.0000000000000e+00)
findGfxWindow('Graphics_1').simulateMouse('up', 53.35125, 54.50625, 1, False, False)
checkpoint Graphics_1 Segment sensitized
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Circle
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:Canvas:vscroll').get_adjustment().set_value( 1.0000000000000e+00)
findWidget('OOF2:Skeleton Selection Page:Mode:Segment').clicked()
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page updated
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:New').clicked()
checkpoint toplevel widget mapped Dialog-Create a new Segment group
findWidget('Dialog-Create a new Segment group').resize(192, 92)
findWidget('Dialog-Create a new Segment group:widget_GTK_RESPONSE_OK').clicked()
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page grouplist Segment
checkpoint OOF.SegmentGroup.New_Group
findWidget('OOF2:Skeleton Selection Page:Pane:Groups:Add').clicked()
checkpoint skeleton selection page grouplist Segment
checkpoint skeleton selection page groups sensitized Segment
checkpoint OOF.SegmentGroup.Add_to_Group
# Clear segment selection
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Selection:Segment:Clear').clicked()
checkpoint skeleton selection page groups sensitized Segment
checkpoint skeleton selection page selection sensitized Segment
checkpoint skeleton selection page updated
checkpoint Graphics_1 Segment sensitized
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Select_Segment.Clear
# Create a Material and add it to the yellow pixels
wevent(findWidget('OOF2:Navigation:PageMenu'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Materials']).activate() # MenuItemLogger
checkpoint Materials page updated
checkpoint page installed Materials
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
findWidget('OOF2:Materials Page:Pane').set_position(289)
findWidget('OOF2:Materials Page:Pane:Material:New').clicked()
checkpoint toplevel widget mapped Dialog-New material
findWidget('Dialog-New material').resize(196, 122)
findWidget('Dialog-New material:widget_GTK_RESPONSE_OK').clicked()
checkpoint Materials page updated
checkpoint OOF.Material.New
findWidget('OOF2:Materials Page:Pane:Material:Assign').clicked()
checkpoint toplevel widget mapped Dialog-Assign material material to pixels
findWidget('Dialog-Assign material material to pixels').resize(221, 122)
wevent(findWidget('Dialog-Assign material material to pixels:pixels'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('Dialog-Assign material material to pixels:pixels').get_window())
checkpoint toplevel widget mapped chooserPopup-pixels
findMenu(findWidget('chooserPopup-pixels'), ['yellow']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-pixels') # MenuItemLogger
findWidget('Dialog-Assign material material to pixels:widget_GTK_RESPONSE_OK').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Material.Assign
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:Canvas:vscroll').get_adjustment().set_value( 0.0000000000000e+00)
# Switch back to the Skeleton Info toolbox
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Skeleton Info']).activate() # MenuItemLogger
checkpoint Graphics_1 Skeleton Info sensitized
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(243)
findWidget('OOF2 Graphics 1').resize(838, 524)
findWidget('OOF2 Graphics 1:Pane0').set_position(392)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(710)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(256)
findWidget('OOF2 Graphics 1').resize(850, 609)
findWidget('OOF2 Graphics 1:Pane0').set_position(477)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(722)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(261)
findWidget('OOF2 Graphics 1').resize(856, 693)
findWidget('OOF2 Graphics 1:Pane0').set_position(561)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(728)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(263)
findWidget('OOF2 Graphics 1').resize(857, 705)
findWidget('OOF2 Graphics 1:Pane0').set_position(573)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(729)
findWidget('OOF2 Graphics 1').resize(858, 707)
findWidget('OOF2 Graphics 1:Pane0').set_position(575)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(730)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(264)
findWidget('OOF2 Graphics 1').resize(859, 707)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(731)
findWidget('OOF2 Graphics 1').resize(925, 713)
findWidget('OOF2 Graphics 1:Pane0').set_position(581)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(797)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(288)
findWidget('OOF2 Graphics 1').resize(957, 726)
findWidget('OOF2 Graphics 1:Pane0').set_position(594)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(829)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(300)
findWidget('OOF2 Graphics 1').resize(966, 734)
findWidget('OOF2 Graphics 1:Pane0').set_position(602)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(838)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(303)
findWidget('OOF2 Graphics 1').resize(966, 747)
findWidget('OOF2 Graphics 1:Pane0').set_position(615)
findWidget('OOF2 Graphics 1').resize(967, 753)
findWidget('OOF2 Graphics 1:Pane0').set_position(621)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(839)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(304)
findWidget('OOF2 Graphics 1').resize(967, 756)
findWidget('OOF2 Graphics 1:Pane0').set_position(624)
findWidget('OOF2 Graphics 1').resize(967, 758)
findWidget('OOF2 Graphics 1:Pane0').set_position(626)
findWidget('OOF2 Graphics 1').resize(966, 759)
findWidget('OOF2 Graphics 1:Pane0').set_position(627)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(838)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(303)
findWidget('OOF2 Graphics 1').resize(966, 763)
findWidget('OOF2 Graphics 1:Pane0').set_position(631)
findWidget('OOF2 Graphics 1').resize(964, 781)
findWidget('OOF2 Graphics 1:Pane0').set_position(649)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(836)
findWidget('OOF2 Graphics 1').resize(961, 796)
findWidget('OOF2 Graphics 1:Pane0').set_position(664)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(833)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(301)
findWidget('OOF2 Graphics 1').resize(962, 796)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(834)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(302)
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':10.97,'Y Text':69.25},cbox)
assert tests.gtkMultiTextCompare({'Material':'material','Group':'','Dom pixel':'yellow','Index':'30','Type':'quad'},elbox)
assert tests.gtkMultiFloatCompare({'Shape':0.0,'Homog':1.0,'Area':156.25},elbox)
assert tests.sensitizationCheck({'Prev':False,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",['Segment 78, nodes (42, 51) (length: 12.5)', 'Segment 69, nodes (42, 43) (length: 12.5)', 'Segment 79, nodes (43, 52) (length: 12.5)', 'Segment 80, nodes (51, 52) (length: 12.5)'], tolerance=1.e-6)
assert tests.chooserListCheck(elbox+":NodeList",['Node 42 at (0.0, 62.5) (angle: 90.0)', 'Node 43 at (12.5, 62.5) (angle: 90.0)', 'Node 52 at (12.5, 75.0) (angle: 90.0)', 'Node 51 at (0.0, 75.0) (angle: 90.0)'], tolerance=1.e-6)

# Click on an element
findGfxWindow('Graphics_1').simulateMouse('down', 31.47625, 55.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 31.47625, 55.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 31.47625, 55.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 31.47625, 55.6, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryElement
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':31.48,'Y Text':55.6},cbox)
assert tests.gtkMultiTextCompare({'Material':'<No material>','Group':'elset','Dom pixel':'blue','Index':'59','Type':'quad'},elbox)
assert tests.gtkMultiFloatCompare({'Shape':0.418601,'Homog':0.992461,'Area':137.500000},elbox)
assert tests.sensitizationCheck({'Prev':False,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",["Segment 135, nodes (36, 44) (length: 15.4029218)", "Segment 134, nodes (36, 37) (length: 15.5)", "Segment 70, nodes (37, 45) (length: 12.5)", "Segment 128, nodes (44, 45) (length: 6.5)"], tolerance=1.e-6)
assert tests.chooserListCheck(elbox+":NodeList",['Node 36 at (22.0, 50.0) (angle: 54.2461127456)', 'Node 37 at (37.5, 50.0) (angle: 90.0)', 'Node 45 at (37.5, 62.5) (angle: 90.0)', 'Node 44 at (31.0, 62.5) (angle: 125.753887254)'])

# Click on another element
findGfxWindow('Graphics_1').simulateMouse('down', 42.3, 86.05, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 42.3, 86.4, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 42.3, 86.4, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':42.3,'Y Text':86.4},cbox)
assert tests.gtkMultiTextCompare({'Material':'material','Group':'','Dom pixel':'yellow','Index':'45','Type':'triangle'},elbox)
assert tests.gtkMultiFloatCompare({'Shape':0.133975,'Homog':0.982564,'Area':81.250000},elbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",["Segment 118, nodes (53, 61) (length: 12.747549)","Segment 119, nodes (53, 62) (length: 18.027756)","Segment 109, nodes (61, 62) (length: 12.747549)"], tolerance=1.e-6)
assert tests.chooserListCheck(elbox+":NodeList",["Node 53 at (40.0, 75.0) (angle: 45.0)","Node 62 at (50.0, 90.0) (angle: 45.0)","Node 61 at (37.5, 87.5) (angle: 90)"], tolerance=1.e-6)

# Click on a third element
findGfxWindow('Graphics_1').simulateMouse('down', 52.1, 97.6, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 52.1, 97.6, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':52.1,'Y Text':97.6},cbox)
assert tests.gtkMultiTextCompare({'Material':'material','Group':'','Dom pixel':'yellow','Index':'46','Type':'triangle'},elbox)
assert tests.gtkMultiFloatCompare({'Shape':0.155097,'Homog':0.999600,'Area':62.500000},elbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",["Segment 110, nodes (62, 70) (length: 10)","Segment 120, nodes (62, 71) (length: 16.0078106)","Segment 121, nodes (70, 71) (length: 12.5)"], tolerance=1.e-6)
assert tests.chooserListCheck(elbox+":NodeList",["Node 62 at (50.0, 90.0) (angle: 51.340191)","Node 71 at (62.5, 100.0) (angle: 38.659808)","Node 70 at (50.0, 100.0) (angle: 90)"], tolerance=1.e-6)

# Click on yet another element.  (Why so many?)
findWidget('OOF2 Graphics 1').resize(962, 796)
findGfxWindow('Graphics_1').simulateMouse('down', 51.4, 68.17375, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 51.4, 68.17375, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':51.4,'Y Text':68.17},cbox)
assert tests.gtkMultiTextCompare({'Material':'<No material>','Group':'elset','Dom pixel':'blue','Index':'31','Type':'quad'},elbox)
assert tests.gtkMultiFloatCompare({'Shape':0.580223,'Homog':0.992404,'Area':162.625000},elbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",["Segment 81, nodes (46, 53) (length: 16.007811)","Segment 74, nodes (46, 47) (length: 16)","Segment 82, nodes (47, 54) (length: 7.3824115)","Segment 83, nodes (53, 54) (length: 23.286262)"], tolerance=1.e-6)
assert tests.chooserListCheck(elbox+":NodeList",["Node 46 at (50.0, 62.5) (angle: 128.659808)","Node 47 at (66.0, 62.5) (angle: 61.69924423)","Node 54 at (62.5, 69.0) (angle: 133.232172)","Node 53 at (40.0, 75.0) (angle: 36.408775)"], tolerance=1.e-6)

# Switch to node mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Node').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryNode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_hadjustment().set_value( 0.0000000000000e+00)
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':51.4,'Y Text':68.17},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Group':'','Mobility':'free','Position':'(50.0, 62.5)','Index':'46'},ndbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",['Element 27', 'Element 28', 'Element 31', 'Element 44'])

# Click on a node
findWidget('OOF2 Graphics 1').resize(962, 796)
findGfxWindow('Graphics_1').simulateMouse('down', 36.7, 99.7, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 36.7, 99.7, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 36.7, 99.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 36.7, 99.35, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':36.7,'Y Text':99.35},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Group':'ndset','Mobility':'x only','Position':'(37.5, 100.0)','Index':'69'},ndbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",["Element 40","Element 41"])

# Click on another node
findGfxWindow('Graphics_1').simulateMouse('down', 0.3, 99.58625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 0.3, 99.58625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 0.3, 99.58625, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':0.3,'Y Text':99.59},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'topleft','Group':'','Mobility':'fixed','Position':'(0.0, 100.0)','Index':'66'},ndbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",["Element 38"])

# x=0.3 y=99.59
# Click on another other node
findGfxWindow('Graphics_1').simulateMouse('down', 49.76375, 61.08625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 49.76375, 61.08625, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':49.76,'Y Text':61.09},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Group':'','Mobility':'free','Position':'(50, 62.5)','Index':'46'},ndbox, tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",["Element 27","Element 28","Element 31","Element 44"])

# Click on yet another other node
findWidget('OOF2 Graphics 1').resize(962, 796)
findGfxWindow('Graphics_1').simulateMouse('down', 72.05, 50, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 72.05, 50, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 72.05, 50, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':72.05,'Y Text':50.},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Group':'','Mobility':'free','Position':'(72, 50)','Index':'39'},ndbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",["Element 24","Element 28","Element 57","Element 61","Element 62","Element 63"])

# Switch to segment mode
findWidget('OOF2 Graphics 1').resize(962, 796)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Segment').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QuerySegment
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':72.05,'Y Text':50},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Groups':'','Index':'136','Length':'15.5','Homogeneity':'1.0'},sgbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 61","Element 62"])
assert tests.chooserListCheck(sgbox+":NodeList",['Node 39 at (72.0, 50.0)', 'Node 40 at (87.5, 50.0)'], tolerance=1.e-6)

# Click on a segment
findGfxWindow('Graphics_1').simulateMouse('down', 42.65, 100.75, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 42.65, 100.75, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QuerySegment
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'42.65','Y Text':'100'},cbox,tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'Boundary':'top','Groups':'segmentgroup','Index':'111','Length':'12.5','Homogeneity':'1.0'},sgbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 41"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 69 at (37.5, 100.0)","Node 70 at (50.0, 100.0)"], tolerance=1.e-6)

# Click on another segment
findGfxWindow('Graphics_1').simulateMouse('up', -7.4, 129.07375, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('down', 5.9, 100.37375, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 5.9, 100.37375, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QuerySegment
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'5.9','Y Text':'100'},cbox, tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'Boundary':'top','Groups':'','Index':'102','Length':'12.5','Homogeneity':'1.0'},sgbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 38"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 66 at (0.0, 100.0)","Node 67 at (12.5, 100.0)"], tolerance=1.e-6)

# click on yet another segment
findGfxWindow('Graphics_1').simulateMouse('down', 58.77625, 0.27375, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 58.77625, 0.27375, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QuerySegment
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'58.78','Y Text':'0.2737'},cbox,tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'Boundary':'bottom','Groups':'','Index':'13','Length':'12.5'},sgbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 4"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 4 at (50.0, 0.0)","Node 5 at (62.5, 0.0)"], tolerance=1.e-6)

# Click on yet another other segment
findGfxWindow('Graphics_1').simulateMouse('down', 44.05, 83.27625, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 44.05, 83.27625, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QuerySegment
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'44.05','Y Text':'83.28'},cbox,tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'Boundary':'','Groups':'segmentgroup','Index':'119','Length':'18.027756377319946','Homogeneity':'0.6333333333333334'},sgbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 45","Element 49"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 53 at (40.0, 75.0)","Node 62 at (50.0, 90.0)"], tolerance=1.e-6)

# Click on another segment.  This is getting to be ridiculous
findGfxWindow('Graphics_1').simulateMouse('down', 64.7, 88.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('move', 64.7, 88.85, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 64.7, 88.85, 1, False, False)
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QuerySegment
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'64.7','Y Text':'88.85'},cbox,tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'Boundary':'','Groups':'segmentgroup','Index':'123','Length':'25.124689052802225','Homogeneity':'0.92'},sgbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 47","Element 48"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 62 at (50.0, 90.0)","Node 63 at (75.0, 87.5)"], tolerance=1.e-6)

# Previous button
findWidget('OOF2 Graphics 1').resize(962, 796)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Prev').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'44.05','Y Text':'83.28'},cbox,tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'Boundary':'','Groups':'segmentgroup','Index':'119','Length':'18.027756377319946','Homogeneity':'0.6333333333333334'},sgbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 45","Element 49"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 53 at (40.0, 75.0)","Node 62 at (50.0, 90.0)"], tolerance=1.e-6)

# Previous button again
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Prev').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'58.78','Y Text':'0.2737'},cbox,tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'Boundary':'bottom','Groups':'','Index':'13','Length':'12.5','Homogeneity':'1.0'},sgbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 4"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 4 at (50.0, 0.0)","Node 5 at (62.5, 0.0)"], tolerance=1.e-6)

# Switch to node mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Node').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryNode
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'58.78','Y Text':'0.2737'},cbox,tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'Boundary':'','Group':'','Mobility':'x only','Position':'(62.5, 0)','Index':'5'},ndbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",["Element 4","Element 5"])

# Previous button, triggers switch to segment mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Prev').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Segment').clicked()
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'58.78','Y Text':'0.2737'},cbox,tolerance=1.e-6)
assert tests.gtkMultiTextCompare({'Boundary':'bottom','Groups':'','Index':'13','Length':'12.5'},sgbox,tolerance=1.e-6)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 4"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 4 at (50.0, 0.0)","Node 5 at (62.5, 0.0)"], tolerance=1.e-6)

# Previous button
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Prev').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':5.9,'Y Text':100},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'top','Groups':'','Index':'102'},sgbox,tolerance=1.e-6)
assert tests.gtkMultiFloatCompare({'Length':12.5},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 38"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 66 at (0.0, 100.0)","Node 67 at (12.5, 100.0)"], tolerance=1.e-6)

# Switch to element mode
findWidget('OOF2 Graphics 1').resize(962, 796)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Element').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Skeleton_Info.QueryElement
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':5.9,'Y Text':100},cbox)
assert tests.gtkMultiTextCompare({'Material':'material','Group':'','Dom pixel':'yellow','Index':'38','Type':'quad'},elbox)
assert tests.gtkMultiFloatCompare({'Shape':0.0,'Homog':1.0, 'Area':156.25}, elbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",["Segment 100, nodes (58, 66) (length: 12.5)","Segment 94, nodes (58, 59) (length: 12.5)","Segment 101, nodes (59, 67) (length: 12.5)","Segment 102, nodes (66, 67) (length: 12.5)"], tolerance=1.e-6)
assert tests.chooserListCheck(elbox+":NodeList",["Node 58 at (0.0, 87.5) (angle: 90)","Node 59 at (12.5, 87.5) (angle: 90)","Node 67 at (12.5, 100.0) (angle: 90)","Node 66 at (0.0, 100.0) (angle: 90)"], tolerance=1.e-6)

# Previous button, triggers switch to segment mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Prev').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Segment').clicked()
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':5.9,'Y Text':100},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'top','Groups':'','Index':'102'},sgbox)
assert tests.gtkMultiFloatCompare({'Length':12.5},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 38"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 66 at (0.0, 100.0)","Node 67 at (12.5, 100.0)"], tolerance=1.e-6)

# Previous
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Prev').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':42.65,'Y Text':100},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'top','Groups':'segmentgroup','Index':'111'},sgbox)
assert tests.gtkMultiFloatCompare({'Length':12.5, 'Homogeneity':1.0},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 41"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 69 at (37.5, 100.0)","Node 70 at (50.0, 100.0)"], tolerance=1.e-6)

# Next
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Next').clicked()
checkpoint Graphics_1 Skeleton Info showed position
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiFloatCompare({'X Text':5.9,'Y Text':100},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'top','Groups':'','Index':'102'},sgbox)
assert tests.gtkMultiFloatCompare({'Length':12.5},sgbox)
assert tests.sensitizationCheck({'Prev':True,'Clear':True,'Next':True},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",["Element 38"])
assert tests.chooserListCheck(sgbox+":NodeList",["Node 66 at (0.0, 100.0)","Node 67 at (12.5, 100.0)"], tolerance=1.e-6)

# Clear button
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Clear').clicked()
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint contourmap info updated for Graphics_1
assert not findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'','Y Text':''},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Groups':'','Index':'','Length':'','Homogeneity':''},sgbox)
assert tests.sensitizationCheck({'Prev':False,'Clear':False,'Next':False},tbox)
assert tests.chooserListCheck(sgbox+":ElementList",[])
assert tests.chooserListCheck(sgbox+":NodeList",[])

# Switch to node mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Node').clicked()
assert not findWidget(tbox+":Click:Element").get_active()
assert findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'','Y Text':''},cbox)
assert tests.gtkMultiTextCompare({'Boundary':'','Group':'','Mobility':'','Position':'','Index':''},ndbox)
assert tests.sensitizationCheck({'Prev':False,'Clear':False,'Next':False},tbox)
assert tests.chooserListCheck(ndbox+":ElementList",[])

# Switch to element mode
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Skeleton Info:Click:Element').clicked()
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'','Y Text':''},cbox)
assert tests.gtkMultiTextCompare({'Material':'','Group':'','Shape':'','Homog':'','Dom pixel':'','Area':'','Index':'','Type':''},elbox)
assert tests.sensitizationCheck({'Prev':False,'Clear':False,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",[])
assert tests.chooserListCheck(elbox+":NodeList",[])

# Delete the Skeleton layer
wevent(findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList'), Gdk.EventType.BUTTON_PRESS, button=3, state=0, window=findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList').get_window())
checkpoint toplevel widget mapped PopUp-0
findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList').get_selection().select_path(Gtk.TreePath([10]))
checkpoint OOF.Graphics_1.Layer.Select
findMenu(findWidget('PopUp-0'), ['Delete']).activate() # MenuItemLogger
checkpoint Move Node toolbox writable changed
checkpoint Move Node toolbox info updated
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint Graphics_1 Element sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Layer.Delete
deactivatePopup('PopUp-0') # MenuItemLogger
assert findWidget(tbox+":Click:Element").get_active()
assert not findWidget(tbox+":Click:Node").get_active()
assert not findWidget(tbox+":Click:Segment").get_active()
assert tests.gtkMultiTextCompare({'X Text':'','Y Text':''},cbox)
assert tests.gtkMultiTextCompare({'Material':'','Group':'','Shape':'','Homog':'','Dom pixel':'','Area':'','Index':'','Type':''},elbox)
assert tests.sensitizationCheck({'Prev':False,'Clear':False,'Next':False},tbox)
assert tests.chooserListCheck(elbox+":SegmentList",[])
assert tests.chooserListCheck(elbox+":NodeList",[])

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
