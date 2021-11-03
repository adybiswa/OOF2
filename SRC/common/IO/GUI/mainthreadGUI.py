# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# Routines for running functions in the main thread.  These override
# the definitions in common/mainthread.py.  See comments there.

from ooflib.SWIG.common import threadstate
from ooflib.common import debug
from ooflib.common import mainthread
from ooflib.common import thread_enable
from gi.repository import GObject
from gi.repository import Gtk
from gi.repository import Gdk
import threading

class OOFIdleCallback:
     def __init__(self, func, args=(), kwargs={}):
          from ooflib.SWIG.common.switchboard import StackWrap
          self.func = StackWrap(func)
          self.args = args
          self.kwargs = kwargs
     def __call__(self):
          try:
               self.func(*self.args, **self.kwargs)
          finally:
               Gdk.flush()      # is this necessary? 
          return False                  # don't repeat

def run_gui(func, args=(), kwargs={}):
     # Unlike runBlock_gui(), below, this function should *not* check
     # thread_enable.query() or mainthread.mainthread().  This
     # function is used by common.oof.front_end() and
     # common.IO.GUI.oofGUI.start() to install idle callbacks that
     # must be installed before the GUI starts and executed
     # afterwards, so the installation must succeed even if threads
     # aren't available yet.
     GObject.idle_add(OOFIdleCallback(func, args, kwargs))

################

# Wrapper for a function to be run as an idle callback on the main
# thread while blocking on a subthread.  That is, the subthread that
# installs the callback blocks until the callback finishes.

class OOFIdleBlockCallback:
     def __init__(self, func, args=(), kwargs={}):
          from ooflib.SWIG.common.switchboard import StackWrap
          self.func = StackWrap(func)
          self.args = args
          self.kwargs = kwargs
          self.event = threading.Event()
          self.result = None
          self.callingthread = threadstate.findThreadNumber()
     def __call__(self):
          # See comment in common.debug.fmsg() about putting
          # debug.fmsg() calls here.
          try:
               self.result = self.func(*self.args, **self.kwargs)
          finally:
               Gdk.flush()
               self.event.set()
          return False              # don't repeat

def runBlock_gui(func, args=(), kwargs={}):
    if thread_enable.query() and not mainthread.mainthread():
        callbackobj = OOFIdleBlockCallback(func, args, kwargs)
        callbackobj.event.clear()
        GObject.idle_add(callbackobj, priority=GObject.PRIORITY_LOW)
        callbackobj.event.wait()
        return callbackobj.result
    else:
        return func(*args, **kwargs)
    
# Override the non-GUI functions.
mainthread.run = run_gui
mainthread.runBlock = runBlock_gui
