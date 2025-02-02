# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.common import debug
from ooflib.common import mainthread
from ooflib.common import utils
from ooflib.common.IO.GUI import gtklogger
from ooflib.common.IO.GUI import parameterwidgets
from ooflib.common.IO.oofmenu import *

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import Gdk

def gtkOOFMenu(menu, accelgroup=None, parentwindow=None):
    # Function to turn an OOFMenu into Gtk.  The leading GtkMenuItem
    # is returned.
    debug.mainthreadTest()
    base = Gtk.MenuItem(label=utils.underscore2space(menu.name))
    menu.parentwindow = parentwindow
    gtklogger.setWidgetName(base, menu.name)
    new_gtkmenu = Gtk.Menu()
    try:
        menu.gtkmenu.append(new_gtkmenu)
    except AttributeError:
        menu.gtkmenu = [new_gtkmenu]

    new_gtkmenu.connect("destroy", menu.gtkmenu_destroyed)
                        
    gtklogger.set_submenu(base, new_gtkmenu)
    menu.setOption('accelgroup', accelgroup)

    for item in menu:
        if not (item.secret or item.getOption('cli_only')):
            item.construct_gui(menu, new_gtkmenu, accelgroup)
    return base


def gtkOOFMenuBar(menu, bar=None, accelgroup=None, parentwindow=None):
    # Function to turn an OOFMenu into a Gtk3 MenuBar.  Reuse the
    # given GtkMenuBar, if one is provided.
    debug.mainthreadTest()
    menu.parentwindow = parentwindow
    if bar is not None:
        # remove old menus from bar
        bar.foreach(Gtk.Widget.destroy)
    else:
        bar = Gtk.MenuBar()
    try:
        menu.gtkmenu.append(bar)
    except AttributeError:
        menu.gtkmenu = [bar]
        
    bar.connect("destroy", menu.gtkmenu_destroyed)
    
    # menu.setOption('accelgroup', accelgroup)

    for item in menu:
        if not (item.secret or item.getOption('cli_only')):
            item.construct_gui(menu, bar, accelgroup)
    return bar

###########################

# Extend the OOFMenu classes so that they can construct the gtk menu

#######################

class MenuCallBackWrapper:
    def __init__(self, menuitem, popup=False):
        self.menuitem = menuitem # An OOFMenuItem, not a GtkMenuItem
        self.popup = popup
    def __call__(self, gtkmenuitem, *args):
        if self.menuitem.gui_callback is None:
            # No special gui callback.
            if self.menuitem.nargs() > 0:
                # Ask for args in a standard dialog box.
                if parameterwidgets.getParameters(
                        title=self.menuitem.gui_title or self.menuitem.name,
                        data={'menuitem':self.menuitem},
                        parentwindow=self.findParentWindow(),
                        *self.menuitem.params):
                    # Call and log the cli callback.
                    self.menuitem.callWithDefaults()
            else:
                # No gui callback and no args.  Call and log the cli callback.
                self.menuitem()
        else:
            # Call, but don't log, the gui callback.  The gui callback
            # will (probably) call and log the cli callback.  If there
            # is no cli callback, nothing will be logged.  This isn't
            # really a problem -- if there is no cli callback, nothing
            # will be done when a script encounters the menu item.
            # Scripts don't call gui callbacks.  Any command that
            # needs to be logged needs to be routed through a non-gui
            # callback, which can be a no-op in text mode.
            self.menuitem.gui_callback(self.menuitem)
    def findParentWindow(self, menuitem=None):
        m = menuitem or self.menuitem
        if m is None:
            return None
        try:
            return m.parentwindow
        except AttributeError:
            return self.findParentWindow(m.parent)

def _menuItemName(self):
    name = utils.underscore2space(self.name)
    # Add an ellipsis if explicitly requested or if there's an
    # automatically generated gui_callback.
    ## TODO: Menu items that take arguments in text mode but get them
    ## automatically in GUI mode should not have an ellipsis appended
    ## to their names in GUI mode.

    if self.ellipsis or (self.params and not self.gui_callback):
        name = name + "..."
    return name

OOFMenuItem.menuItemName = _menuItemName

# Utility function to check if all of this menu item's children
# are visible.  Returns false if there are no children.  This tells
# the parent whether or not to construct a submenu, and prevents the
# construction of empty submenus.  Visibility is a GUI thing.
def _OOFMenuItem_children_visible(self):
    if not self.items:
        return None
    for i in self.items:
        if not (i.secret or i.getOption('cli_only')):
            return 1 # Return true on the first visible item.
    return None # Redundant, None is default return value, but clearer.

OOFMenuItem.children_visible = _OOFMenuItem_children_visible
    
def _OOFMenuItem_construct_gui(self, base, parent_menu, accelgroup,
                               popup=False):
    # "base" is this menu item's OOF menu parent, and "parent_menu" is
    # the to-be-constructed GtkMenuItem's gtk container.
    debug.mainthreadTest()
    if not (self.secret or self.getOption('cli_only')):

        new_gtkitem = Gtk.MenuItem(label=self.menuItemName()) 
        gtklogger.setWidgetName(new_gtkitem, self.name)
        try:
            self.gtkitem.append(new_gtkitem)
        except AttributeError:
            self.gtkitem = [new_gtkitem]
            
        new_gtkitem.connect("destroy", self.gtkitem_destroyed)
        
        parent_menu.insert(new_gtkitem, self.gui_order())

        ## Right justification of help menus is "now considered a bad
        ## idea" according to the gtk documentation, and
        ## MenuItem.set_right_justified is deprecated.
        # if self.help_menu:
        #     base.gtkhelpmenu = 1
        #     new_gtkitem.set_right_justified(True)

        if (self.callback is None and self.gui_callback is None 
            and self.children_visible()):

            new_gtkmenu = Gtk.Menu()
            try:
                self.gtkmenu.append(new_gtkmenu)
            except AttributeError:
                self.gtkmenu=[new_gtkmenu]

            gtklogger.set_submenu(new_gtkitem, new_gtkmenu)
            for item in self.items:
                # recursively construct submenu
                item.construct_gui(self, new_gtkmenu, accelgroup, popup=popup)
        else:                   # no submenu, create command
            gtklogger.connect(
                new_gtkitem, 'activate', MenuCallBackWrapper(self, popup))
            if self.accel is not None and accelgroup is not None:
                new_gtkitem.add_accelerator('activate', accelgroup,
                                            ord(self.accel),
                                            Gdk.ModifierType.CONTROL_MASK,
                                            Gtk.AccelFlags.VISIBLE)
        if not self.enabled():
            new_gtkitem.set_sensitive(False)

OOFMenuItem.construct_gui = _OOFMenuItem_construct_gui


# Destroys all the GUI objects associated with a menu item.
def _OOFMenuItem_destroy_gui(self):
    mainthread.runBlock(self.destroy_gui_thread)

def _OOFMenuItem_destroy_gui_thread(self):
    debug.mainthreadTest()
    if hasattr(self, 'gtkitem'):
        # Iterate over a copy, because destroy() triggers callbacks
        # which modify the list.
        for i in self.gtkitem[:]:
            i.destroy()
    if self.items and hasattr(self, 'gtkmenu'):
        # Copies again, for the same reason as above.
        for m in self.gtkmenu[:]:
            m.destroy()
        for item in self.items:
            item.destroy_gui_thread()
                
OOFMenuItem.destroy_gui = _OOFMenuItem_destroy_gui
OOFMenuItem.destroy_gui_thread = _OOFMenuItem_destroy_gui_thread

# GTK callbacks to clean up the object lists when GUIs are removed.
def _OOFMenuItem_gtkmenu_destroyed(self, gtkmenu):
    debug.mainthreadTest()
    self.gtkmenu.remove(gtkmenu)

OOFMenuItem.gtkmenu_destroyed = _OOFMenuItem_gtkmenu_destroyed

def _OOFMenuItem_gtkitem_destroyed(self, gtkitem):
    debug.mainthreadTest()
    i = self.gtkitem.index(gtkitem)
    del self.gtkitem[i]
    ## TODO: This is ugly.  Use a virtual function instead.
    if hasattr(self, 'handlerid'):
        del self.handlerid[i]
    
OOFMenuItem.gtkitem_destroyed = _OOFMenuItem_gtkitem_destroyed
    

########################

_oldAddItem = OOFMenuItem.addItem

def _newAddItem(self, item):
    return mainthread.runBlock(self.addItem_thread, (item,))
def _addItem_thread(self, item):
    debug.mainthreadTest()
    _oldAddItem(self, item)
    # Check to see if the gui has been constructed yet. The gui
    # objects for the root of the menu have gtkmenu attributes, but
    # not gtkitem attributes.  Other nodes of the tree have gtkitem,
    # but may not have gtkmenu, so it's necessary to check for both.
    if (hasattr(self, 'gtkitem') or hasattr(self, 'gtkmenu')) and \
       self.children_visible():
        # We've been guied, so gui the new children, if they're visible.
        if not hasattr(self, 'gtkmenu'):
            # Make a gtkmenu for each gtkitem.
            self.gtkmenu = [Gtk.Menu() for i in range(len(self.gtkitem))]
            for (i,m) in zip(self.gtkitem, self.gtkmenu):
                gtklogger.set_submenu(i, m)
            for m in self.gtkmenu:
                m.connect("destroy", self.gtkmenu_destroyed)
            
        # At this point, we ourselves are guaranteed both gtkitem and gtkmenu.
        # Build a GUI object for each menu.
        for m in self.gtkmenu:
            item.construct_gui(self, m, item.getOption('accelgroup'))

    # If the parent menu had been desensitized because it had been
    # empty, it has to be sensitized now.
    self.enable_parent_gui()
    
    try:
        for x in self.gtkmenu:
            x.show_all()
    except AttributeError:
        pass
    try:
        for x in self.gtkitem:
            x.show_all()
    except AttributeError:
        pass
    return item

OOFMenuItem.addItem = _newAddItem
OOFMenuItem.addItem_thread = _addItem_thread

_oldRemoveItem = OOFMenuItem.removeItem

def _newRemoveItem(self, name):
    item = self.getItem(name)
    item.destroy_gui()
    _oldRemoveItem(self, name)
    if not self.items:    # desensitize self if it has no more items
        self.sensitize_gui(0)

OOFMenuItem.removeItem = _newRemoveItem
        

########################

class CheckMenuCallBackWrapper(MenuCallBackWrapper):
    def __call__(self, gtkmenuitem, *args):
        active = gtkmenuitem.get_active()
        if self.popup:
            gtkmenuitem.get_parent().destroy()
        return self.menuitem(active)

def _CheckOOFMenuItem_construct_gui(self, base, parent_menu, accelgroup,
                                    popup=False):
    debug.mainthreadTest()
    if not (self.secret or self.getOption('cli_only')):
        new_gtkitem = Gtk.CheckMenuItem(label=self.menuItemName())
        gtklogger.setWidgetName(new_gtkitem, self.name)
        try:
            self.gtkitem.append(new_gtkitem)
        except AttributeError:
            self.gtkitem = [new_gtkitem]
        new_gtkitem.connect("destroy", self.gtkitem_destroyed)
        # Set the state of the button.  This calls the callback, so we do
        # it here before the callback is connected.
        new_gtkitem.set_active(self.value)
        if self.accel is not None and accelgroup is not None:
            new_gtkitem.add_accelerator('activate', accelgroup,
                                        ord(self.accel),
                                        Gdk.ModifierType.CONTROL_MASK,
                                        Gtk.AccelFlags.VISIBLE)

        # Handler IDs are added in the same order as items, so there
        # is item-for-item correspondence of the lists.  They're used
        # to suppress recursion when the state of the check mark is
        # set manually.
        new_handler = gtklogger.connect(
            new_gtkitem, 'activate', CheckMenuCallBackWrapper(self, popup))
        try:
            self.handlerid.append(new_handler)
        except AttributeError:
            self.handlerid = [new_handler]

        if not self.enabled():
            new_gtkitem.set_sensitive(False)

        parent_menu.insert(new_gtkitem, self.gui_order())

CheckOOFMenuItem.construct_gui = _CheckOOFMenuItem_construct_gui

# Redefine the CheckOOFMenuItem __call__ method so that commands
# executed from scripts will set the radio buttons in the GUI
# correctly. 
_old_CheckOOFMenuItem___call__ = CheckOOFMenuItem.__call__

def _CheckOOFMenuItem___call__(self, active):
    _old_CheckOOFMenuItem___call__(self, active)
    # Before calling set_active(), it's necessary to disable the gtk
    # callback mechanism for the button, so that set_active() won't
    # call the callback, which calls this function, which calls
    # set_active(), ad infinitum.
    try:
        gtkitem = self.gtkitem
    except AttributeError:
        pass
    else:
        mainthread.runBlock(_setactive, (gtkitem, self.handlerid, active))
        

def _setactive(gtkitem, handler, active):
    debug.mainthreadTest()
    for (i,h) in zip(gtkitem, handler):
        h.block()
        i.set_active(active)
        h.unblock()

CheckOOFMenuItem.__call__ = _CheckOOFMenuItem___call__

#########################

class RadioMenuCallBackWrapper(CheckMenuCallBackWrapper):
    def __call__(self, gtkmenuitem, *args):
        debug.mainthreadTest()
        # Since the RadioOOFMenuItem takes care of calling the
        # callback for the item that's being turned off, here we only
        # call the callback if an item is being turned on.
        if gtkmenuitem.active:
            if self.popup:
                gtkmenuitem.get_parent().destroy()
            return self.menuitem()


def _RadioOOFMenuItem_construct_gui(self, base, parent_menu, accelgroup,
                                    popup=False):
    debug.mainthreadTest()

    new_gtkitem = Gtk.RadioMenuItem(self.menuItemName())
    gtklogger.setWidgetName(new_gtkitem, self.name)
    try:
        gtkgroup = self.group.gtk
        new_gtkitem.join_group(gtkgroup)
    except AttributeError:
        new_gtkitem.join_group(None)
        self.group.gtk = new_gtkitem
                                       
    # Set the state of the button.  This calls the callback, so we do
    # it here before the callback is connected.
    new_gtkitem.set_active(self.value)
    
    try:
        self.gtkitem.append(new_gtkitem)
    except AttributeError:
        self.gtkitem = [new_gtkitem]
    new_gtkitem.connect("destroy", self.gtkitem_destroyed)
        
    if self.accel is not None and accelgroup is not None:
        new_gtkitem.add_accelerator('activate', accelgroup,
                                    ord(self.accel),
                                    Gdk.ModifierType.CONTROL_MASK,
                                    Gtk.AccelFlags.VISIBLE)
        
    new_handlerid = gtklogger.connect(
        new_gtkitem, 'activate', RadioMenuCallBackWrapper(self, popup))

    try:
        self.handlerid.append(new_handlerid)
    except AttributeError:
        self.handlerid = [new_handlerid]
    
    if self.getOption('disabled'):
        new_gtkitem.set_sensitive(0)
    parent_menu.add(new_gtkitem)

RadioOOFMenuItem.construct_gui = _RadioOOFMenuItem_construct_gui

# See comments above about redefining __call__ for CheckOOFMenuItem.

_old_RadioOOFMenuItem___call__ = RadioOOFMenuItem.__call__

def _RadioOOFMenuItem___call__(self):
    debug.mainthreadTest()
    _old_RadioOOFMenuItem___call__(self)
    mainthread.runBlock(_setactive, (self.gtkitem, self.handlerid, 1))


RadioOOFMenuItem.__call__ = _RadioOOFMenuItem___call__

###################################################

# Redefine 'enable' and 'disable' so that the menus are grayed out.

def _sensitize_gui(self, sensitive):
    mainthread.runBlock(self.sensitize_gui_thread, (sensitive,))

def _sensitize_gui_thread(self, sensitive):
    debug.mainthreadTest()
    try:
        itemlist = self.gtkitem
    except AttributeError:
        pass
    else:
        for i in itemlist:
            i.set_sensitive(sensitive)
OOFMenuItem.sensitize_gui = _sensitize_gui
OOFMenuItem.sensitize_gui_thread = _sensitize_gui_thread

_old_disable = OOFMenuItem.disable
def _OOFMenuItem_disable(self):
    mainthread.runBlock(self.disable_thread)
def disable_thread(self):
    debug.mainthreadTest()
    _old_disable(self)
    self.sensitize_gui(0)
OOFMenuItem.disable = _OOFMenuItem_disable
OOFMenuItem.disable_thread = disable_thread

def _enable_children(self):
    debug.mainthreadTest()
    if self.enabled():
        self.sensitize_gui(1)
        for item in self.items:
            item.enable_children()
OOFMenuItem.enable_children = _enable_children

def _enable_parent_gui(self):
    debug.mainthreadTest()
    if self.enabled():
        self.sensitize_gui(1)
    if self.parent is not None:
            self.parent.enable_parent_gui()
OOFMenuItem.enable_parent_gui = _enable_parent_gui

_old_enable = OOFMenuItem.enable
def _OOFMenuItem_enable(self):
    _old_enable(self)
    mainthread.runBlock(self.enable_children)
OOFMenuItem.enable = _OOFMenuItem_enable

# When a gui callback is added, an automatically disabled menu item
# might become enabled.
_old_add_gui_callback = OOFMenuItem.add_gui_callback
def _OOFMenuItem_add_gui_callback(self, callback):
    _old_add_gui_callback(self, callback)
    if self.enabled():
        self.sensitize_gui(1)
OOFMenuItem.add_gui_callback = _OOFMenuItem_add_gui_callback

####################################################

# def _invokeGUICallback(self):
#     self.gui_callback(self)

# OOFMenuItem.invokeGUICallback = _invokeGUICallback
