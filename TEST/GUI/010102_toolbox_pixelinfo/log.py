# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# Test the misorientation calculation in the pixel selection toolbox

import tests
findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
findWidget('OOF2').resize(782, 511)
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
checkpoint microstructure page sensitized
checkpoint OOF.Microstructure.Create_From_ImageFile
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
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.0000000000000e+00)
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Layer.New
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint pixel page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Burn
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint OOF.PixelGroup.New
checkpoint microstructure page sensitized
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.PixelGroup.AddSelection
checkpoint microstructure page sensitized
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint pixel page updated
checkpoint pixel page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Burn
checkpoint microstructure page sensitized
checkpoint pixel page updated
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint pixel page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Select.Burn
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint skeleton selection page groups sensitized Element
checkpoint OOF.PixelGroup.New
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.PixelGroup.AddSelection
checkpoint Materials page updated
checkpoint OOF.Material.New
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Material.Assign
checkpoint Materials page updated
checkpoint OOF.Material.New
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Material.Assign
findWidget('OOF2:Materials Page:Pane:Property:PropertyScroll:PropertyTree').expand_row(Gtk.TreePath([5]), open_all=False)
checkpoint Materials page updated
checkpoint property selected
checkpoint OOF.Property.Copy
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Property.Parametrize.Orientation.green
checkpoint property selected
checkpoint Materials page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Material.Add_property
checkpoint Materials page updated
checkpoint property selected
checkpoint OOF.Property.Copy
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Property.Parametrize.Orientation.blue
checkpoint Materials page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Material.Add_property
checkpoint OOF.File.LoadStartUp.Script

findWidget('OOF2 Graphics 1').resize(800, 492)
wevent(findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-TBChooser
findMenu(findWidget('chooserPopup-TBChooser'), ['Pixel Info']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-TBChooser') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(224)
findWidget('OOF2 Graphics 1').resize(800, 501)
findWidget('OOF2 Graphics 1:Pane0').set_position(369)
findWidget('OOF2 Graphics 1').resize(809, 516)
findWidget('OOF2 Graphics 1:Pane0').set_position(384)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(681)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(227)
findWidget('OOF2 Graphics 1').resize(862, 572)
findWidget('OOF2 Graphics 1:Pane0').set_position(440)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(734)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(245)
findWidget('OOF2 Graphics 1').resize(867, 574)
findWidget('OOF2 Graphics 1:Pane0').set_position(442)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(739)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(247)
findWidget('OOF2 Graphics 1').resize(1112, 626)
findWidget('OOF2 Graphics 1:Pane0').set_position(494)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(984)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(330)
findWidget('OOF2 Graphics 1').resize(1221, 635)
findWidget('OOF2 Graphics 1:Pane0').set_position(503)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(1093)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(367)
findWidget('OOF2 Graphics 1').resize(1255, 635)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(1127)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(378)
findWidget('OOF2 Graphics 1').resize(1260, 635)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(1132)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(380)
findWidget('OOF2 Graphics 1').resize(1259, 646)
findWidget('OOF2 Graphics 1:Pane0').set_position(514)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(1131)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.6000000000000e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.0200000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.5600000000000e+02)
# Check initial state.
assert tests.checkInitialState()

# Click on a pixel with no orientation data
findGfxWindow('Graphics_1').simulateMouse('down', 107.2875, 124.875, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 107.2875, 124.875, 1, False, False)
checkpoint Graphics_1 Pixel Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Query
# Check that nothing has changed. 
assert tests.checkInitialState()

# Click on a pixel with an orientation
findGfxWindow('Graphics_1').simulateMouse('down', 46.9125, 73.425, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 46.9125, 73.425, 1, False, False)
checkpoint Graphics_1 Pixel Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Query

findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.5710618765855e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.8710618765855e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.7000000000000e+02)
# Check that orientation is displayed: Abg(45, 105, 0)
# Check that Set Reference button is sensitive.  All other widgets unchanged.
assert tests.checkOrientation("Abg", alpha=45, beta=105, gamma=0)
assert tests.checkOrientationButNoReference()
assert tests.checkMisorientation("???")

# Click "Set Reference Point"
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:setref').clicked()
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Misorientation.Set_Reference
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.7110618765855e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.7810618765855e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.0410618765855e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.5110618765855e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 4.1610618765855e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 4.3700000000000e+02)
assert tests.checkReference(x=46, y=73, oclass="Abg", alpha=45, beta=105, gamma=0)
assert tests.checkMisorientation("0.0")

# Click on another point with a different orientation
findGfxWindow('Graphics_1').simulateMouse('down', 120.9375, 75.525, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 120.9375, 75.525, 1, False, False)
checkpoint Graphics_1 Pixel Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Query
# Check that nothing has changed except the misorientation
assert tests.checkMisorientation("111.55304645", tolerance=1.e-8)
assert tests.checkReference(x=46, y=73, oclass="Abg", alpha=45, beta=105, gamma=0)

# Click on another point that has no misorientation
findGfxWindow('Graphics_1').simulateMouse('down', 16.9875, 59.25, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 16.9875, 59.25, 1, False, False)
checkpoint Graphics_1 Pixel Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Query
# Check that misorientation=0, nothing else changed
assert tests.checkMisorientation("0.0")
assert tests.checkReference(x=46, y=73, oclass="Abg", alpha=45, beta=105, gamma=0)

# Click on a point with no orientation
findWidget('OOF2 Graphics 1').resize(1259, 646)
findGfxWindow('Graphics_1').simulateMouse('down', 90.4875, 121.725, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 90.4875, 121.725, 1, False, False)
checkpoint Graphics_1 Pixel Info updated
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.2300000000000e+02)
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Query
assert tests.checkMisorientation("???")
assert tests.checkOrientation(None)
assert tests.checkReference(x=46, y=73, oclass="Abg", alpha=45, beta=105, gamma=0)

# Click on a misoriented point again
findGfxWindow('Graphics_1').simulateMouse('down', 118.3125, 82.35, 1, False, False)
findGfxWindow('Graphics_1').simulateMouse('up', 118.8375, 82.35, 1, False, False)
checkpoint Graphics_1 Pixel Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Query
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.6300000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 4.3700000000000e+02)
# Check that misorientation is 111.553 again
assert tests.checkMisorientation("111.55304645", tolerance=1.e-8)
assert tests.checkOrientation("Abg", alpha=0, beta=0, gamma=0)
assert tests.checkReference(x=46, y=73, oclass="Abg", alpha=45, beta=105, gamma=0)

# Change the symmetry
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:symmetry:set').clicked()
checkpoint toplevel widget mapped Dialog-Set lattice symmetry for misorientation calculation
findWidget('Dialog-Set lattice symmetry for misorientation calculation').resize(342, 134)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:SpaceGroup:number:slider').get_adjustment().set_value( 4.1805555555556e+00)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:SpaceGroup:number:slider').get_adjustment().set_value( 1.0541666666667e+01)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:SpaceGroup:number:slider').get_adjustment().set_value( 3.5986111111111e+01)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:SpaceGroup:number:slider').get_adjustment().set_value( 6.7791666666667e+01)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:SpaceGroup:number:slider').get_adjustment().set_value( 1.0277777777778e+02)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:SpaceGroup:number:slider').get_adjustment().set_value( 1.4730555555556e+02)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:SpaceGroup:number:slider').get_adjustment().set_value( 1.6002777777778e+02)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:SpaceGroup:number:slider').get_adjustment().set_value( 1.6320833333333e+02)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:SpaceGroup:number:slider').get_adjustment().set_value( 1.6638888888889e+02)
findWidget('Dialog-Set lattice symmetry for misorientation calculation:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Misorientation.Set_Symmetry
assert tests.checkSymmetry("Space Group 166")
assert tests.checkMisorientation("47.31068112772261", tolerance=1.e-8)

# Change symmetry to an equivalent in a different notation
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:symmetry:set').clicked()
checkpoint toplevel widget mapped Dialog-Set lattice symmetry for misorientation calculation
findWidget('Dialog-Set lattice symmetry for misorientation calculation').resize(342, 134)
wevent(findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Schoenflies']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('Dialog-Set lattice symmetry for misorientation calculation:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Misorientation.Set_Symmetry
assert tests.checkSymmetry("Schoenflies D3d")
assert tests.checkMisorientation("47.31068112772261", tolerance=1.e-8)

# Change to another symmetry
findWidget('OOF2 Graphics 1').resize(1259, 646)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:symmetry:set').clicked()
checkpoint toplevel widget mapped Dialog-Set lattice symmetry for misorientation calculation
findWidget('Dialog-Set lattice symmetry for misorientation calculation').resize(221, 128)
wevent(findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:RCFChooser'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['International']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
wevent(findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:International:name'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:International:name').get_window())
checkpoint toplevel widget mapped chooserPopup-name
findMenu(findWidget('chooserPopup-name'), ['23']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-name') # MenuItemLogger
findWidget('Dialog-Set lattice symmetry for misorientation calculation:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Misorientation.Set_Symmetry
assert tests.checkSymmetry("International 23")
assert tests.checkMisorientation("47.31068112772261", tolerance=1.e-8)

findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:symmetry:set').clicked()
checkpoint toplevel widget mapped Dialog-Set lattice symmetry for misorientation calculation
findWidget('Dialog-Set lattice symmetry for misorientation calculation').resize(204, 128)
wevent(findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:International:name'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('Dialog-Set lattice symmetry for misorientation calculation:symmetry:International:name').get_window())
checkpoint toplevel widget mapped chooserPopup-name
findMenu(findWidget('chooserPopup-name'), ['2']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-name') # MenuItemLogger
findWidget('Dialog-Set lattice symmetry for misorientation calculation:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Misorientation.Set_Symmetry
assert tests.checkSymmetry("International 2")
assert tests.checkMisorientation("85.72927190169425", tolerance=1.e-8)

# Manually set query pixel position to one with no misorientation
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 4.3619326555159e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 4.0019326555159e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.8419326555159e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.0419326555159e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 0.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:X').set_text('')
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:X').set_text('1')
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:X').set_text('10')
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:Y').set_text('')
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:Y').set_text('8')
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:Y').set_text('80')
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll:Pixel Info:Update').clicked()
checkpoint Graphics_1 Pixel Info updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Toolbox.Pixel_Info.Query
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 1.1550660792952e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 6.5453744493392e+01)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 2.0791189427313e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.0031718061674e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.9464757709251e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 4.3700000000000e+02)
assert tests.checkMisorientation("0.0")

# Hide the image
findWidget('OOF2 Graphics 1').resize(1259, 646)
wevent(findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList').get_window())
findCellRenderer(findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList'), col=0, rend=0).emit('toggled', Gtk.TreePath(13))
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 0.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.0000000000000e+00)
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint selection info updated Element
checkpoint Graphics_1 Pixel Selection sensitized
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.2300000000000e+02)
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Layer.Hide
assert tests.checkMisorientation("???")

# Unhide the image
findWidget('OOF2 Graphics 1').resize(1259, 646)
wevent(findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList'), Gdk.EventType.BUTTON_PRESS, button=1, state=0, window=findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList').get_window())
findCellRenderer(findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList'), col=0, rend=0).emit('toggled', Gtk.TreePath(13))
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 0.0000000000000e+00)
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 2.0000000000000e+00)
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Element
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Layer.Show
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.2400000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.6000000000000e+02)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 4.3700000000000e+02)
assert tests.checkMisorientation("0.0")

# Delete the image layer
findWidget('OOF2 Graphics 1').resize(1259, 646)
wevent(findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList'), Gdk.EventType.BUTTON_PRESS, button=3, state=0, window=findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList').get_window())
checkpoint toplevel widget mapped PopUp-0
findMenu(findWidget('PopUp-0'), ['Delete']).activate() # MenuItemLogger
deactivatePopup('PopUp-0') # MenuItemLogger
findWidget('OOF2 Graphics 1:Pane0:LayerScroll').get_vadjustment().set_value( 0.0000000000000e+00)
checkpoint Graphics_1 Pixel Info updated
checkpoint selection info updated Pixel Selection
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2:TBScroll').get_vadjustment().set_value( 3.2300000000000e+02)
checkpoint selection info updated Element
checkpoint Graphics_1 Pixel Selection sensitized
checkpoint selection info updated Node
checkpoint selection info updated Segment
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Graphics_1.Layer.Delete
assert tests.checkMisorientation("???")

findMenu(findWidget('OOF2:MenuBar'), ['File', 'Save', 'Python_Log']).activate()
checkpoint toplevel widget mapped Dialog-Python_Log
findWidget('Dialog-Python_Log').resize(192, 122)
findWidget('OOF2').resize(782, 545)
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