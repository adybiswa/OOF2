findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
findWidget('OOF2').resize(782, 511)
findMenu(findWidget('OOF2:MenuBar'), ['Help', 'Tutorials']).activate()
findMenu(findWidget('OOF2:MenuBar'), ['Help', 'Tutorials']).activate()
findMenu(findWidget('OOF2:MenuBar'), ['Help', 'Tutorials', 'Skeleton']).activate()
checkpoint toplevel widget mapped Skeleton
findWidget('Skeleton').resize(500, 300)
findWidget('OOF2').resize(782, 545)
findWidget('Skeleton').resize(500, 300)
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 9.6000000000000e+01,y= 2.0000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
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
findWidget('OOF2:Microstructure Page:Pane').set_position(189)
checkpoint pixel page sensitized
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint Materials page updated
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint boundary page updated
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized
checkpoint Solver page sensitized
checkpoint microstructure page sensitized
checkpoint OOF.Microstructure.Create_From_ImageFile
findWidget('Skeleton:Next').clicked()
findWidget('OOF2:Navigation:Next').clicked()
checkpoint page installed Image
findWidget('OOF2:Image Page:Pane').set_position(546)
findWidget('OOF2:Image Page:Group').clicked()
checkpoint toplevel widget mapped Dialog-AutoGroup
findWidget('Dialog-AutoGroup').resize(207, 92)
findWidget('Dialog-AutoGroup:widget_GTK_RESPONSE_OK').clicked()
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint skeleton selection page groups sensitized
checkpoint microstructure page sensitized
checkpoint meshable button set
checkpoint meshable button set
checkpoint microstructure page sensitized
checkpoint OOF.Image.AutoGroup
findWidget('OOF2:Navigation:PrevHist').clicked()
checkpoint page installed Microstructure
checkpoint meshable button set
checkpoint microstructure page sensitized
findWidget('OOF2:Microstructure Page:Pane').set_position(235)
findWidget('Skeleton:Next').clicked()
# groups created
event(Gdk.EventType.BUTTON_PRESS,x= 8.1000000000000e+01,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
findWidget('OOF2:Skeleton Page:Pane').set_position(417)
checkpoint skeleton page sensitized
checkpoint skeleton page sensitized
findWidget('Skeleton:Next').clicked()
findWidget('Skeleton').resize(500, 302)
findWidget('OOF2:Skeleton Page:New').clicked()
checkpoint toplevel widget mapped Dialog-New skeleton
findWidget('Dialog-New skeleton').resize(346, 254)
findWidget('Dialog-New skeleton:widget_GTK_RESPONSE_OK').clicked()
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized
checkpoint skeleton page sensitized
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint boundary page updated
checkpoint skeleton page info updated
checkpoint skeleton selection page grouplist
checkpoint skeleton page info updated
checkpoint skeleton selection page selection sensitized
checkpoint Solver page sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint OOF.Skeleton.New
checkpoint skeleton page sensitized
findWidget('Skeleton:Next').clicked()
findMenu(findWidget('OOF2:MenuBar'), ['Windows', 'Graphics', 'New']).activate()
checkpoint Move Node toolbox info updated
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Mesh Info sensitized
findWidget('OOF2 Graphics 1:Pane0').set_position(360)
findWidget('OOF2 Graphics 1:Pane0:Pane1').set_position(672)
findWidget('OOF2 Graphics 1:Pane0:Pane1:Pane2').set_position(212)
checkpoint toplevel widget mapped OOF2 Graphics 1
findWidget('OOF2 Graphics 1:Pane0:LayerScroll:LayerList').get_selection().select_path(Gtk.TreePath([10]))
checkpoint OOF.Graphics_1.Layer.Select
findWidget('OOF2 Graphics 1').resize(800, 492)
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Windows.Graphics.New
findWidget('OOF2 Graphics 1').resize(800, 492)
findWidget('Skeleton:Next').clicked()
findWidget('Skeleton').resize(500, 518)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:targets:Heterogeneous Elements:threshold:slider').get_adjustment().set_value( 1.0000000000000e+00)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:alpha:entry').set_text('0.')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:alpha:entry').set_text('0.5')
widget_0=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:alpha:entry')
if widget_0: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_0.get_window())
del widget_0
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
findWidget('OOF2:Skeleton Page:Pane:Modification:Undo').clicked()
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint boundary page updated
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Undo
findWidget('OOF2:Skeleton Page:Pane:Modification:Redo').clicked()
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint boundary page updated
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Redo
findWidget('Skeleton:Next').clicked()
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:targets:Heterogeneous Elements:threshold:slider').get_adjustment().set_value( 9.8611111111111e-01)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:targets:Heterogeneous Elements:threshold:entry').set_text('0.')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:targets:Heterogeneous Elements:threshold:entry').set_text('0.9')
widget_1=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Refine:targets:Heterogeneous Elements:threshold:entry')
if widget_1: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_1.get_window())
del widget_1
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
findWidget('Skeleton:Next').clicked()
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 1.0700000000000e+02,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Snap Nodes']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(437)
checkpoint skeleton page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 8.2000000000000e+01,y= 1.4000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Snap Nodes:targets:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Heterogeneous Elements']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(417)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Snap Nodes:criterion:Average Energy:alpha:slider').get_adjustment().set_value( 4.9397590361446e-01)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Snap Nodes:criterion:Average Energy:alpha:slider').get_adjustment().set_value( 5.1807228915663e-01)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Snap Nodes:criterion:Average Energy:alpha:slider').get_adjustment().set_value( 6.0240963855422e-01)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Snap Nodes:criterion:Average Energy:alpha:slider').get_adjustment().set_value( 7.1084337349398e-01)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Snap Nodes:criterion:Average Energy:alpha:slider').get_adjustment().set_value( 8.4337349397590e-01)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Snap Nodes:criterion:Average Energy:alpha:slider').get_adjustment().set_value( 9.6385542168675e-01)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Snap Nodes:criterion:Average Energy:alpha:slider').get_adjustment().set_value( 1.0000000000000e+00)
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 1.1300000000000e+02,y= 9.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Rationalize']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2').resize(782, 647)
findWidget('OOF2:Skeleton Page:Pane').set_position(285)
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Rationalize:criterion:Average Energy:alpha:entry').set_text('')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Rationalize:criterion:Average Energy:alpha:entry').set_text('0')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Rationalize:criterion:Average Energy:alpha:entry').set_text('0.')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Rationalize:criterion:Average Energy:alpha:entry').set_text('0.8')
widget_2=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Rationalize:criterion:Average Energy:alpha:entry')
if widget_2: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_2.get_window())
del widget_2
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 5.4000000000000e+01,y= 2.0000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint skeleton selection page grouplist
checkpoint page installed Skeleton Selection
findWidget('OOF2:Skeleton Selection Page:Pane').set_position(474)
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
event(Gdk.EventType.BUTTON_PRESS,x= 8.6000000000000e+01,y= 1.2000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Selection Page:Pane:Selection:ElementAction:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Select by Homogeneity']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:ElementHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.ElementSelection.Select_by_Homogeneity
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 5.8000000000000e+01,y= 1.8000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
checkpoint skeleton page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 9.4000000000000e+01,y= 1.6000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Split Quads']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(430)
checkpoint skeleton page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 5.9000000000000e+01,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Split Quads:targets:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Selected Elements']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Split Quads:criterion:Average Energy:alpha:entry').set_text('0.')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Split Quads:criterion:Average Energy:alpha:entry').set_text('0.9')
widget_3=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Split Quads:criterion:Average Energy:alpha:entry')
if widget_3: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_3.get_window())
del widget_3
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 1.0200000000000e+02,y= 1.0000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Rationalize']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(285)
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page selection sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 1.0500000000000e+02,y= 2.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton Selection
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:ElementHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.ElementSelection.Select_by_Homogeneity
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 5.7000000000000e+01,y= 1.8000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
checkpoint skeleton page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 1.0500000000000e+02,y= 1.0000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Anneal']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(437)
checkpoint skeleton page sensitized
findWidget('OOF2 Graphics 1').resize(800, 492)
event(Gdk.EventType.BUTTON_PRESS,x= 7.7000000000000e+01,y= 9.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:targets:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Selected Elements']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:criterion:Average Energy:alpha:entry').set_text('0.')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:criterion:Average Energy:alpha:entry').set_text('0.9')
widget_4=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:criterion:Average Energy:alpha:entry')
if widget_4: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_4.get_window())
del widget_4
widget_5=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:criterion:Average Energy:alpha:entry')
if widget_5: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_5.get_window())
del widget_5
widget_6=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:criterion:Average Energy:alpha:entry')
if widget_6: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_6.get_window())
del widget_6
widget_7=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:criterion:Average Energy:alpha:entry')
if widget_7: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_7.get_window())
del widget_7
event(Gdk.EventType.BUTTON_PRESS,x= 1.3300000000000e+02,y= 9.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:iteration:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Conditional Iteration']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
widget_8=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:criterion:Average Energy:alpha:entry')
if widget_8: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_8.get_window())
del widget_8
findWidget('OOF2').resize(782, 667)
findWidget('OOF2:Skeleton Page:Pane').set_position(293)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:iteration:Conditional Iteration:extra').set_text('')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Anneal:iteration:Conditional Iteration:extra').set_text('3')
checkpoint skeleton page sensitized
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint toplevel widget mapped OOF2 Activity Viewer
findWidget('OOF2 Activity Viewer').resize(400, 300)
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('OOF2 Activity Viewer').resize(400, 300)
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 1.3100000000000e+02,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Rationalize']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(285)
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 6.5000000000000e+01,y= 1.4000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton Selection
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:ElementHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.ElementSelection.Select_by_Homogeneity
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 7.8000000000000e+01,y= 1.2000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
checkpoint skeleton page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 1.4800000000000e+02,y= 8.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Anneal']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(293)
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 1.0800000000000e+02,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Rationalize']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(285)
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 6.9000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton Selection']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton Selection
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:ElementHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.ElementSelection.Select_by_Homogeneity
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:ElementAction:Select by Homogeneity:threshold:entry').set_text('0.')
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:ElementAction:Select by Homogeneity:threshold:entry').set_text('0.8')
widget_9=findWidget('OOF2:Skeleton Selection Page:Pane:Selection:ElementAction:Select by Homogeneity:threshold:entry')
if widget_9: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_9.get_window())
del widget_9
findWidget('OOF2:Skeleton Selection Page:Pane:Selection:ElementHistory:OK').clicked()
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.ElementSelection.Select_by_Homogeneity
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 7.9000000000000e+01,y= 1.3000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
checkpoint skeleton page sensitized
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 7.1000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Pin Nodes']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Pin Nodes
findWidget('OOF2:Pin Nodes Page:Pane').set_position(549)
event(Gdk.EventType.BUTTON_PRESS,x= 4.6000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Pin Nodes Page:Pane:Modify:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Pin Internal Boundary Nodes']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Pin Nodes Page:Pane').set_position(508)
findWidget('OOF2:Pin Nodes Page:Pane:Modify:OK').clicked()
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.PinNodes.Pin_Internal_Boundary_Nodes
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 4.7000000000000e+01,y= 2.2000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Skeleton']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Skeleton
checkpoint skeleton page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 1.2300000000000e+02,y= 1.3000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Swap Edges']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(437)
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Swap Edges:criterion:Average Energy:alpha:entry').set_text('0.')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Swap Edges:criterion:Average Energy:alpha:entry').set_text('0.5')
widget_10=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Swap Edges:criterion:Average Energy:alpha:entry')
if widget_10: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_10.get_window())
del widget_10
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('OOF2:Skeleton Page:Pane:Modification:Undo').clicked()
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint boundary page updated
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Undo
findWidget('OOF2:Skeleton Page:Pane:Modification:Redo').clicked()
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint boundary page updated
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Redo
event(Gdk.EventType.BUTTON_PRESS,x= 4.6000000000000e+01,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Rationalize']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(285)
checkpoint skeleton page sensitized
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page selection sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 1.7200000000000e+02,y= 1.7000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Merge Triangles']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(437)
checkpoint skeleton page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 8.1000000000000e+01,y= 1.1000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Merge Triangles:criterion:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Limited Unconditional']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(390)
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Merge Triangles:criterion:Limited Unconditional:alpha:entry').set_text('0.')
findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Merge Triangles:criterion:Limited Unconditional:alpha:entry').set_text('0.5')
widget_11=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Merge Triangles:criterion:Limited Unconditional:alpha:entry')
if widget_11: event(Gdk.EventType.FOCUS_CHANGE, in_=0, window=widget_11.get_window())
del widget_11
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('OOF2:Skeleton Page:Pane:Modification:Undo').clicked()
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint boundary page updated
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Undo
findWidget('OOF2:Skeleton Page:Pane:Modification:Redo').clicked()
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint boundary page updated
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Redo
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 8.1000000000000e+01,y= 7.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Smooth']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(437)
checkpoint skeleton page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 7.6000000000000e+01,y= 1.7000000000000e+01,button=1,state=0,window=findWidget('OOF2:Skeleton Page:Pane:Modification:Method:Smooth:criterion:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['Limited Unconditional']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('OOF2:Skeleton Page:Pane').set_position(390)
findWidget('OOF2:Skeleton Page:Pane:Modification:OK').clicked()
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint boundary page updated
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint skeleton page sensitized
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint Move Node toolbox writable changed
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Modify
findWidget('OOF2:Skeleton Page:Pane:Modification:Undo').clicked()
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint boundary page updated
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Undo
findWidget('OOF2:Skeleton Page:Pane:Modification:Redo').clicked()
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page grouplist
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page groups sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint contourmap info updated for Graphics_1
checkpoint skeleton selection page selection sensitized
checkpoint skeleton selection page groups sensitized
checkpoint skeleton selection page updated
checkpoint contourmap info updated for Graphics_1
checkpoint boundary page updated
checkpoint skeleton page info updated
checkpoint skeleton page info updated
checkpoint skeleton page sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint Graphics_1 Skeleton Info sensitized
checkpoint Graphics_1 Move Nodes sensitized
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.Redo
findWidget('Skeleton:Next').clicked()
findMenu(findWidget('OOF2:MenuBar'), ['File', 'Save', 'Skeleton']).activate()
checkpoint toplevel widget mapped Dialog-Skeleton
findWidget('Dialog-Skeleton').resize(192, 218)
findWidget('Dialog-Skeleton:filename').set_text('s')
findWidget('Dialog-Skeleton:filename').set_text('sk')
findWidget('Dialog-Skeleton:filename').set_text('ske')
findWidget('Dialog-Skeleton:filename').set_text('skel')
findWidget('Dialog-Skeleton:filename').set_text('skele')
findWidget('Dialog-Skeleton:filename').set_text('skelet')
findWidget('Dialog-Skeleton:filename').set_text('skeleto')
findWidget('Dialog-Skeleton:filename').set_text('skeleton')
findWidget('Dialog-Skeleton:filename').set_text('skeleton.')
findWidget('Dialog-Skeleton:filename').set_text('skeleton.d')
findWidget('Dialog-Skeleton:filename').set_text('skeleton.da')
findWidget('Dialog-Skeleton:filename').set_text('skeleton.dat')
findWidget('Dialog-Skeleton:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.File.Save.Skeleton
findWidget('Skeleton:Next').clicked()
event(Gdk.EventType.BUTTON_PRESS,x= 6.0000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Pin Nodes']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint page installed Pin Nodes
findWidget('OOF2:Pin Nodes Page:Pane:Modify:Unpin All').clicked()
checkpoint contourmap info updated for Graphics_1
checkpoint OOF.Skeleton.PinNodes.UnpinAll
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
findWidget('Dialog-Python_Log').resize(194, 122)
findWidget('Dialog-Python_Log:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.File.Save.Python_Log
findMenu(findWidget('OOF2:MenuBar'), ['File', 'Quit']).activate()
checkpoint OOF.Graphics_1.File.Close
