# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from . import loggers
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import Gdk
from . import logutils

import sys

# When recording, _wdict[obj] is the name of a variable which, when
# replaying, will be storing a weak reference to obj.  All the
# references are weak so that storing them will not have side effects.
import weakref
_wdict = weakref.WeakKeyDictionary()

class WidgetLogger(loggers.GtkLogger):
    classes = (Gtk.Widget,)

    def location(self, widget, *args):
        # This does not use getWidgetPathStr because it uses both the
        # list of strings and the colon separated string.
        path = logutils.getWidgetPath(widget)
        if path[0] not in logutils.getTopWidgetNames():
            raise logutils.GtkLoggerException(':'.join(path) + 
                                     " is not contained in a top-level widget")
        return "findWidget('%s')" % ':'.join(path)

    def record(self, obj, signal, *args):
        if signal in ('button-press-event', 'button-release-event'):
            evnt = args[0]
            if signal == 'button-press-event':
                eventname = "BUTTON_PRESS"
            else:
                eventname = "BUTTON_RELEASE"
            return [
                ## See comments re event and wevent in replay.py.  The
                ## commented-out lines that use event here don't
                ## always replay correctly, in particular when trying
                ## to detect whether a modifier key has been pressed
                ## before a button event.  Switching to wevent appears
                ## to fix the problem.
                # "event(Gdk.EventType.%s,x=%20.13e,y=%20.13e,button=%d,state=%d,window=%s.get_window())"
                # % (eventname, evnt.x, evnt.y, evnt.button, evnt.state,
                #    self.location(obj, *args))
                "wevent(%(w)s, Gdk.EventType.%(e)s, button=%(b)d, state=%(s)d, window=%(w)s.get_window())"
                % dict(w=self.location(obj, *args),
                       e=eventname,
                       b=evnt.button,
                       s=evnt.state)
                
                ]

        if signal in ('key-press-event', 'key-release-event'):
            evnt = args[0]
            if signal == 'key-press-event':
                eventname = "KEY_PRESS"
            else:
                eventname = "KEY_RELEASE"
            # keymap = Gdk.Keymap.get_default()
            # ok, keymapkeys = keymap.get_entries_for_keyval(evnt.keyval)
            # print "keys=", [k.keycode for k in keymapkeys], \
            #     "event.hardware_keycode=", evnt.hardware_keycode

            # Don't log the modifier keys.  There seem to be problems
            # created by generating the keypress events for them.
            # (For example, recreating a Shift_L event in OOF2 was
            # somehow activating the menu item with the ^A keyboard
            # accelerator.)  We don't need the modifier keypress
            # events as long as the "state" data recorded in other
            # events includes the correct modifiers.
            if is_modifier(evnt.keyval): # evnt.is_modifier isn't set properly!
                return self.ignore
            return [
                # TODO: Including the hardware_keycode seems to be
                # important for getting the delete key to work, but
                # it's not portable.  Do we need to log the delete
                # key?  For example, Gtk.Entry is logged via other
                # signals and doesn't need to handle key press events.
                # When would a delete keypress need to be logged as an
                # event?
                "event(Gdk.EventType.%s, keyval=Gdk.keyval_from_name('%s'), state=%d, window=%s.get_window())"
                % (eventname, Gdk.keyval_name(evnt.keyval), evnt.state,
                   self.location(obj, *args))
                # "wevent(%(w)s, Gdk.EventType.%(e)s, keyval=Gdk.keyval_from_name('%s'), state=%d, window=%(w)s.get_window()"
                # % dict(w=self.location(obj, *args), e=eventname, s=evnt.state)
            ]
        
        if signal == 'motion-notify-event':
            evnt = args[0]
            if logutils.suppress_motion_events(obj):
                return self.ignore
            return [
                "event(Gdk.EventType.MOTION_NOTIFY,x=%20.13e,y=%20.13e,state=%d,window=%s.get_window())"
                % (evnt.x, evnt.y, evnt.state, self.location(obj, *args))
            ]
        
        if signal == 'focus_in_event':
            return [
                "event(Gdk.EventType.FOCUS_CHANGE, in_=1, window=%s.get_window())" % self.location(obj, *args)
            ]

        # If a widget has lost focus because it's been destroyed for
        # some reason, then replaying the focus_out_event will fail.
        # The widget destruction would have been caused by a previous
        # line in the log file, so the widget will have been destroyed
        # before the focus_out_event is replayed.  Therefore it's
        # necessary to check that the widget still exists before
        # issuing the signal.
        if signal == 'focus_out_event':
            try:
                wvar = _wdict[obj]
                lines = []
            except KeyError:
                wvar = loggers.localvar('widget')
                _wdict[obj] = wvar
                # weakRef returns a weak reference to its argument,
                # which is a widget.  If the widget no longer exists,
                # the argument will be None, and weakRef returns a
                # function that returns None, so that the "if..." in
                # the following line will not be satisfied.  This
                # prevents the creation of strong reference to the
                # widget in a local variable, which could have side
                # effects.
                lines = ["%s=weakRef(%s)" % (wvar,self.location(obj,*args))]
            return lines + [
                "if %(w)s(): wevent(%(w)s(), Gdk.EventType.FOCUS_CHANGE, in_=0, window=%(w)s().get_window())" % dict(w=wvar),
                ]

        if signal in ('enter-notify-event', 'leave-notify-event'):
            evnt = args[0]
            device = evnt.get_device()
            if signal == 'enter-notify-event':
                etype = "ENTER_NOTIFY"
            else:
                etype = "LEAVE_NOTIFY"
            return [
                "event(Gdk.EventType.%(etype)s, window=%(widget)s.get_window())" % dict(etype=etype, widget=self.location(obj, *args))
            ]

        if signal == 'destroy':
            return ['%s.destroy()' % self.location(obj, *args)]

        if signal == 'size-allocate':
            # Allocation events should be logged on top level widgets
            # only.  I think.  Gtk will figure out the sizes of
            # everything else.
            if logutils.isTopLevelWidget(obj):
                alloc = obj.get_allocation()
                parent = obj.get_parent()
                return ["%s.size_allocate(Gdk.Rectangle(%d, %d, %d, %d))" \
                       % (self.location(obj, *args),
                          alloc.x, alloc.y, alloc.width, alloc.height)]
        return super(WidgetLogger, self).record(obj, signal, *args)

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

# See comment above about not logging keypress events for modifier
# keys.  GdkEventKey.is_modifier isn't set correctly, so we have to
# detect them the hard way.  This is probably incorrect for non-Latin
# keyboards.

modifiernames = ["Shift_L", "Shift_R",
                 "Control_L", "Control_R",
                 "Meta_L", "Meta_R",
                 "Alt_L", "Alt_R"]
modifierkeyvals = list(map(Gdk.keyval_from_name, modifiernames))

def is_modifier(keyval):
    return keyval in modifierkeyvals
