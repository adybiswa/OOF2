# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# Test the initialization pane on the solver page

import tests
findWidget('OOF2:FE Mesh Page:Pane').set_position(557)
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(106)
checkpoint toplevel widget mapped OOF2
checkpoint page installed Introduction
findWidget('OOF2').resize(782, 545)

event(Gdk.EventType.BUTTON_PRESS,x= 6.4000000000000e+01,y= 2.0000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Solver']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint Solver page sensitized
checkpoint page installed Solver
findWidget('OOF2:Solver Page:VPane').set_position(170)
assert tests.sensitization0()

event(Gdk.EventType.BUTTON_PRESS,x= 8.8000000000000e+01,y= 2.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
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
checkpoint skeleton selection page grouplist Element
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized Element
checkpoint Solver page sensitized
checkpoint OOF.Microstructure.New
event(Gdk.EventType.BUTTON_PRESS,x= 6.9000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
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
checkpoint skeleton selection page selection sensitized Element
checkpoint skeleton selection page updated
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton page sensitized
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
checkpoint OOF.Skeleton.New
checkpoint skeleton page info updated
checkpoint skeleton selection page groups sensitized Element
checkpoint skeleton selection page updated
checkpoint skeleton page sensitized
event(Gdk.EventType.BUTTON_PRESS,x= 6.3000000000000e+01,y= 1.9000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['FE Mesh']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint mesh page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint page installed FE Mesh
findWidget('OOF2:FE Mesh Page:Pane:leftpane').set_position(130)
findWidget('OOF2:FE Mesh Page:New').clicked()
checkpoint toplevel widget mapped Dialog-Create a new mesh
findWidget('Dialog-Create a new mesh').resize(299, 244)
findWidget('Dialog-Create a new mesh:widget_GTK_RESPONSE_OK').clicked()
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
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Mesh.New
event(Gdk.EventType.BUTTON_PRESS,x= 7.3000000000000e+01,y= 5.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Solver']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint Solver page sensitized
checkpoint page installed Solver
assert tests.sensitization0()
assert tests.listCheck()
assert tests.selection(None)

# Define a field
event(Gdk.EventType.BUTTON_PRESS,x= 9.5000000000000e+01,y= 1.9000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Fields & Equations']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint Field page sensitized
checkpoint Field page sensitized
checkpoint page installed Fields & Equations
findWidget('OOF2:Fields & Equations Page:HPane').set_position(470)
findWidget('OOF2:Fields & Equations Page:HPane:Fields:Temperature defined').clicked()
checkpoint Field page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Subproblem.Field.Define
findWidget('OOF2:Navigation:PrevHist').clicked()
checkpoint Solver page sensitized
checkpoint page installed Solver
assert tests.sensitization0()
assert tests.selection(None)
assert tests.listCheck('Temperature', 'Temperature_z')

# Select a field
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll:Initializers').get_selection().select_path(Gtk.TreePath([0]))
checkpoint Solver page sensitized
assert tests.sensitization1()
assert tests.selection(0)

# Unselect the field
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll:Initializers').get_selection().unselect_all()
checkpoint Solver page sensitized
assert tests.sensitization0()
assert tests.selection(None)

# Reselect field
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll:Initializers').get_selection().select_path(Gtk.TreePath([0]))
checkpoint Solver page sensitized
assert tests.sensitization1()
assert tests.selection(0)

# Assign initializer
findWidget('OOF2:Solver Page:VPane:FieldInit:Set').clicked()
checkpoint toplevel widget mapped Dialog-Initialize field Temperature
findWidget('Dialog-Initialize field Temperature').resize(232, 134)
findWidget('Dialog-Initialize field Temperature:initializer:Constant:value').set_text('')
findWidget('Dialog-Initialize field Temperature:initializer:Constant:value').set_text('1')
findWidget('Dialog-Initialize field Temperature:initializer:Constant:value').set_text('12')
findWidget('Dialog-Initialize field Temperature:initializer:Constant:value').set_text('123')
findWidget('Dialog-Initialize field Temperature:widget_GTK_RESPONSE_OK').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Mesh.Set_Field_Initializer
assert tests.sensitization2()
assert tests.selection(0)

# Remove initializer
findWidget('OOF2:Solver Page:VPane:FieldInit:Clear').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
# checkpoint Solver page sensitized
checkpoint OOF.Mesh.Clear_Field_Initializer
assert tests.sensitization1()
assert tests.selection(0)

# Reassign initializer
tree=findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll:Initializers')
column = tree.get_column(0)
tree.row_activated(Gtk.TreePath([0]), column)
checkpoint toplevel widget mapped Dialog-Initialize field Temperature
findWidget('Dialog-Initialize field Temperature').resize(232, 134)
findWidget('Dialog-Initialize field Temperature:widget_GTK_RESPONSE_OK').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
# checkpoint Solver page sensitized
checkpoint OOF.Mesh.Set_Field_Initializer
assert tests.sensitization2()
assert tests.selection(0)

# Assign second initializer
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll:Initializers').get_selection().select_path(Gtk.TreePath([1]))
checkpoint Solver page sensitized
findWidget('OOF2:Solver Page:VPane:FieldInit:Set').clicked()
checkpoint toplevel widget mapped Dialog-Initialize field Temperature_z
findWidget('Dialog-Initialize field Temperature_z').resize(232, 134)
event(Gdk.EventType.BUTTON_PRESS,x= 6.8000000000000e+01,y= 1.3000000000000e+01,button=1,state=0,window=findWidget('Dialog-Initialize field Temperature_z:initializer:RCFChooser').get_window())
checkpoint toplevel widget mapped chooserPopup-RCFChooser
findMenu(findWidget('chooserPopup-RCFChooser'), ['XYTFunction']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-RCFChooser') # MenuItemLogger
findWidget('Dialog-Initialize field Temperature_z').resize(245, 134)
findWidget('Dialog-Initialize field Temperature_z:initializer:XYTFunction:function').set_text('x')
findWidget('Dialog-Initialize field Temperature_z:initializer:XYTFunction:function').set_text('x+')
findWidget('Dialog-Initialize field Temperature_z:initializer:XYTFunction:function').set_text('x+7')
findWidget('Dialog-Initialize field Temperature_z:initializer:XYTFunction:function').set_text('x+')
findWidget('Dialog-Initialize field Temperature_z:initializer:XYTFunction:function').set_text('x+y')
findWidget('Dialog-Initialize field Temperature_z:widget_GTK_RESPONSE_OK').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Mesh.Set_Field_Initializer
assert tests.sensitization2()
assert tests.selection(1)

# Remove first initializer
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll:Initializers').get_selection().select_path(Gtk.TreePath([0]))
checkpoint Solver page sensitized
findWidget('OOF2:Solver Page:VPane:FieldInit:Clear').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
# checkpoint Solver page sensitized
checkpoint OOF.Mesh.Clear_Field_Initializer
assert tests.sensitization3()
assert tests.selection(0)

# Remove all initializers
findWidget('OOF2:Solver Page:VPane:FieldInit:ClearAll').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Mesh.Clear_Field_Initializers
assert tests.sensitization1()
assert tests.selection(0)

# Define another field
findWidget('OOF2:Navigation:NextHist').clicked()
checkpoint Field page sensitized
checkpoint Field page sensitized
checkpoint page installed Fields & Equations
findWidget('OOF2:Fields & Equations Page:HPane:Fields:Displacement defined').clicked()
checkpoint Field page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Subproblem.Field.Define
findWidget('OOF2:Fields & Equations Page:HPane:Fields:Displacement in-plane').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Mesh.Field.In_Plane
findWidget('OOF2:Navigation:PrevHist').clicked()
checkpoint Solver page sensitized
checkpoint page installed Solver
assert tests.sensitization1()
assert tests.listCheck("Temperature", "Temperature_z", "Displacement")
assert tests.selection(0)

# Select new field
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll').get_vadjustment().set_value( 1.0000000000000e+00)
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll').get_vadjustment().set_value( 2.0000000000000e+00)
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll').get_vadjustment().set_value( 8.0000000000000e+00)
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll:Initializers').get_selection().select_path(Gtk.TreePath([2]))
checkpoint Solver page sensitized

# Initialize second field
tree=findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll:Initializers')
column = tree.get_column(1)
tree.row_activated(Gtk.TreePath([2]), column)
checkpoint toplevel widget mapped Dialog-Initialize field Displacement
findWidget('Dialog-Initialize field Displacement').resize(215, 170)
findWidget('Dialog-Initialize field Displacement:initializer:Constant:cx').set_text('')
findWidget('Dialog-Initialize field Displacement:initializer:Constant:cx').set_text('1')
findWidget('Dialog-Initialize field Displacement:initializer:Constant:cy').set_text('')
findWidget('Dialog-Initialize field Displacement:initializer:Constant:cy').set_text('2')
findWidget('Dialog-Initialize field Displacement:widget_GTK_RESPONSE_OK').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
# checkpoint Solver page sensitized
checkpoint OOF.Mesh.Set_Field_Initializer
assert tests.sensitization2()
assert tests.selection(2)

# Apply initializers.  There are no explicit tests here, other the
# checkpoint that confirms that the command has completed, and the log
# file test at the end that checks that the arguments were obtained
# correctly.
findWidget('OOF2:Solver Page:VPane:FieldInit:Apply').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Mesh.Apply_Field_Initializers

# Apply at time.  There are no explicit tests here either.
findWidget('OOF2:Solver Page:VPane:FieldInit:ApplyAt').clicked()
checkpoint toplevel widget mapped Dialog-Initialize Fields at Time
findWidget('Dialog-Initialize Fields at Time').resize(192, 92)
findWidget('Dialog-Initialize Fields at Time:time').set_text('')
findWidget('Dialog-Initialize Fields at Time:time').set_text('2')
findWidget('Dialog-Initialize Fields at Time:time').set_text('23')
findWidget('Dialog-Initialize Fields at Time:time').set_text('2')
findWidget('Dialog-Initialize Fields at Time:time').set_text('')
findWidget('Dialog-Initialize Fields at Time:time').set_text('1')
findWidget('Dialog-Initialize Fields at Time:time').set_text('12')
findWidget('Dialog-Initialize Fields at Time:widget_GTK_RESPONSE_OK').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Mesh.Apply_Field_Initializers_at_Time
# Check that mesh time is 12
assert tests.sensitization2()
assert tests.checkTime(12.0)

# Create second mesh
event(Gdk.EventType.BUTTON_PRESS,x= 4.5000000000000e+01,y= 8.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['FE Mesh']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint mesh page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint page installed FE Mesh
findWidget('OOF2:FE Mesh Page:New').clicked()
checkpoint toplevel widget mapped Dialog-Create a new mesh
findWidget('Dialog-Create a new mesh').resize(299, 244)
findWidget('Dialog-Create a new mesh:widget_GTK_RESPONSE_OK').clicked()
checkpoint mesh bdy page updated
checkpoint mesh bdy page updated
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint mesh page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint Solver page sensitized
checkpoint mesh page sensitized
checkpoint OOF.Mesh.New
event(Gdk.EventType.BUTTON_PRESS,x= 7.9000000000000e+01,y= 1.6000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Solver']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint Solver page sensitized
checkpoint page installed Solver
assert tests.sensitization2()
assert tests.listCheck("Temperature", "Temperature_z", "Displacement")

# Switch to second mesh
event(Gdk.EventType.BUTTON_PRESS,x= 4.6000000000000e+01,y= 1.2000000000000e+01,button=1,state=0,window=findWidget('OOF2:Solver Page:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh<2>']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
checkpoint Solver page sensitized
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll').get_vadjustment().set_value( 0.0000000000000e+00)
assert tests.sensitization0()
assert tests.listCheck()

# Switch back to first mesh
event(Gdk.EventType.BUTTON_PRESS,x= 4.6000000000000e+01,y= 9.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Solver Page:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
checkpoint Solver page sensitized
assert tests.sensitization4()
assert tests.listCheck("Temperature", "Temperature_z", "Displacement")

# Copy initializers to second mesh
findWidget('OOF2:Solver Page:VPane:FieldInit:CopyInit').clicked()
checkpoint toplevel widget mapped Dialog-Select a target Mesh
findWidget('Dialog-Select a target Mesh').resize(192, 152)
event(Gdk.EventType.BUTTON_PRESS,x= 6.4000000000000e+01,y= 1.5000000000000e+01,button=1,state=0,window=findWidget('Dialog-Select a target Mesh:target:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh<2>']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
findWidget('Dialog-Select a target Mesh:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Mesh.Copy_Field_Initializers

# Switch to second mesh
event(Gdk.EventType.BUTTON_PRESS,x= 5.9000000000000e+01,y= 1.2000000000000e+01,button=1,state=0,window=findWidget('OOF2:Solver Page:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh<2>']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
checkpoint Solver page sensitized
# Nothing should have happened, since fields aren't defined.
assert tests.sensitization0()
assert tests.listCheck()

# Define only one field (T) on second mesh.
event(Gdk.EventType.BUTTON_PRESS,x= 5.4000000000000e+01,y= 1.7000000000000e+01,button=1,state=0,window=findWidget('OOF2:Navigation:PageMenu').get_window())
checkpoint toplevel widget mapped chooserPopup-PageMenu
findMenu(findWidget('chooserPopup-PageMenu'), ['Fields & Equations']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-PageMenu') # MenuItemLogger
checkpoint Field page sensitized
checkpoint Field page sensitized
checkpoint page installed Fields & Equations
event(Gdk.EventType.BUTTON_PRESS,x= 6.3000000000000e+01,y= 9.0000000000000e+00,button=1,state=0,window=findWidget('OOF2:Fields & Equations Page:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh<2>']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
checkpoint Field page sensitized
findWidget('OOF2:Fields & Equations Page:HPane:Fields:Temperature defined').clicked()
checkpoint Field page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Subproblem.Field.Define
findWidget('OOF2:Navigation:PrevHist').clicked()
checkpoint Solver page sensitized
checkpoint page installed Solver
assert tests.sensitization0()
assert tests.listCheck("Temperature", "Temperature_z")

# Switch to first mesh
event(Gdk.EventType.BUTTON_PRESS,x= 4.2000000000000e+01,y= 1.9000000000000e+01,button=1,state=0,window=findWidget('OOF2:Solver Page:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
checkpoint Solver page sensitized
assert tests.sensitization4()
assert tests.listCheck("Temperature", "Temperature_z", "Displacement")

# Define initializer for T
findWidget('OOF2:Solver Page:VPane:FieldInit:Scroll:Initializers').get_selection().select_path(Gtk.TreePath([0]))
checkpoint Solver page sensitized
findWidget('OOF2:Solver Page:VPane:FieldInit:Set').clicked()
checkpoint toplevel widget mapped Dialog-Initialize field Temperature
findWidget('Dialog-Initialize field Temperature').resize(232, 134)
findWidget('Dialog-Initialize field Temperature:widget_GTK_RESPONSE_OK').clicked()
checkpoint Solver page sensitized
checkpoint Solver page sensitized
# checkpoint Solver page sensitized
checkpoint OOF.Mesh.Set_Field_Initializer
assert tests.sensitization2()

# Copy initializers to second mesh
findWidget('OOF2:Solver Page:VPane:FieldInit:CopyInit').clicked()
checkpoint toplevel widget mapped Dialog-Select a target Mesh
findWidget('Dialog-Select a target Mesh').resize(192, 152)
event(Gdk.EventType.BUTTON_PRESS,x= 5.6000000000000e+01,y= 1.4000000000000e+01,button=1,state=0,window=findWidget('Dialog-Select a target Mesh:target:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh<2>']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
findWidget('Dialog-Select a target Mesh:widget_GTK_RESPONSE_OK').clicked()
checkpoint OOF.Mesh.Copy_Field_Initializers
assert tests.sensitization2()
assert tests.listCheck("Temperature", "Temperature_z", "Displacement")

# Switch to second mesh
event(Gdk.EventType.BUTTON_PRESS,x= 6.4000000000000e+01,y= 1.7000000000000e+01,button=1,state=0,window=findWidget('OOF2:Solver Page:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh<2>']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
checkpoint Solver page sensitized
assert tests.sensitization2()
assert tests.listCheck("Temperature", "Temperature_z")

# Undefine field on second mesh
findWidget('OOF2:Navigation:NextHist').clicked()
checkpoint Field page sensitized
checkpoint Field page sensitized
checkpoint page installed Fields & Equations
findWidget('OOF2:Fields & Equations Page:HPane:Fields:Temperature defined').clicked()
checkpoint Field page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint Field page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Subproblem.Field.Undefine
findWidget('OOF2:Navigation:PrevHist').clicked()
checkpoint Solver page sensitized
checkpoint page installed Solver
assert tests.sensitization0()
assert tests.listCheck()

# Switch to first mesh
event(Gdk.EventType.BUTTON_PRESS,x= 6.1000000000000e+01,y= 1.8000000000000e+01,button=1,state=0,window=findWidget('OOF2:Solver Page:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
checkpoint Solver page sensitized
assert tests.sensitization4()
assert tests.listCheck("Temperature", "Temperature_z", "Displacement")

# Switch back to second mesh, then delete it.
event(Gdk.EventType.BUTTON_PRESS,x= 6.8000000000000e+01,y= 1.0000000000000e+01,button=1,state=0,window=findWidget('OOF2:Solver Page:Mesh').get_window())
checkpoint toplevel widget mapped chooserPopup-Mesh
findMenu(findWidget('chooserPopup-Mesh'), ['mesh<2>']).activate() # MenuItemLogger
deactivatePopup('chooserPopup-Mesh') # MenuItemLogger
checkpoint Solver page sensitized
findWidget('OOF2:Navigation:PrevHist').clicked()
checkpoint mesh page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint page installed FE Mesh
findWidget('OOF2:FE Mesh Page:Delete').clicked()
checkpoint toplevel widget mapped Questioner
findWidget('Questioner').resize(297, 86)
findWidget('Questioner:Yes').clicked()
checkpoint Field page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint Solver page sensitized
checkpoint Solver page sensitized
checkpoint mesh bdy page updated
checkpoint Field page sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page subproblems sensitized
checkpoint mesh page sensitized
checkpoint Solver page sensitized
checkpoint OOF.Mesh.Delete
findWidget('OOF2:Navigation:NextHist').clicked()
checkpoint Solver page sensitized
checkpoint page installed Solver
assert tests.sensitization4()
assert tests.listCheck("Temperature", "Temperature_z", "Displacement")

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
assert tests.filediff('session.log')

checkpoint_count("Solver page sensitized")
findMenu(findWidget('OOF2:MenuBar'), ['File', 'Quit']).activate()
