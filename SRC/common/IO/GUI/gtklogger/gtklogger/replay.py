# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# Classes and functions for reading and replaying log files.

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gdk
from gi.repository import Gtk
from gi.repository import GLib
import sys
import weakref

from . import core
from . import checkpoint
from . import loggers
from . import logutils

## log files being played back are executed in this namespace, so
## functions used in log files have to be defined here.
checkpoint_count = checkpoint.checkpoint_count

_threaded = True

# replaydelay is the delay time (in milliseconds) that is inserted
# between lines in the log file during playback, if there is no
# explicit delay ('pause' statement).  The timing is not guaranteed to
# be precise.
replaydelay = 0

def set_delay(delay):
    global replaydelay
    replaydelay = delay

# Time in milliseconds to wait before retrying a checkpoint or looking
# for a window.
retrydelay = 100

# maxtries limits the number of times a line that raises a
# GtkLoggerTopFailure exception will be retried.
maxtries = 10

## See the README file for a description of the arguments to replay().

def replay(filename, beginCB=None, finishCB=None, debugLevel=2,
           threaded=False, exceptHook=None, rerecord=None, checkpoints=True,
           comment_gui=False):
    logutils.set_replaying(True)
    GLib.idle_add(
        function=GUILogPlayer(filename, beginCB, finishCB, debugLevel,
                              threaded, exceptHook, rerecord, checkpoints,
                              comment_gui))

# A GUILogPlayer reads a log file of saved gui events and simulates them.

class GUILogPlayer:
    current = None
    def __init__(self, filename, beginCB=None, finishCB=None, debugLevel=2,
                 threaded=False, exceptHook=None, rerecord=None,
                 checkpoints=True, comment_gui=False):
        global _threaded
        logutils.set_debugLevel(debugLevel)
        _threaded = threaded
        self.beginCB = beginCB
        self.finishCB = finishCB
        self.exceptHook = exceptHook
        self.checkpoints = checkpoints  # if False, checkpoints will be ignored
        try:
            file = open(filename, "r")
        except IOError:
            self.aborted = True
            raise
        self.linerunners = []

        if rerecord:
            core.start(rerecord, comment_gui=comment_gui)

        # More than one execution line can come from a single source
        # line, if, for example, automatic pauses are inserted.  The
        # execution line number is used to know when to run a line.
        # The source line number is useful for debugging.
        lineno = 0                      # number of the line being executed
        srcline = 1                     # source line number
        pausenext = False               # add a pause before the next line?
        lines = file.readlines()
        self._nfilelines = len(lines)
        tobecontinued = ""
        for line in lines:
            line = line.rstrip()
            if not line:
                self.linerunners.append(CommentLine(self, srcline, lineno, ""))
                lineno += 1
                pausenext = False
            elif line[-1] == '\\':
                # Lines ending in a backslash aren't processed
                # immediately.  They're prepended to the next line
                # instead.
                tobecontinued += line[:-1]
            else:                   # line isn't continued
                line = tobecontinued + line
                tobecontinued = ""
                if line.lstrip()[0] == "#": # line is a comment 
                    self.linerunners.append(
                        CommentLine(self, srcline, lineno, line.rstrip()))
                    lineno += 1
                    pausenext = False
                else:               # not a comment 
                    try:
                        words = line.split(None, 1) # look for keyword
                    except:
                        words = None
                    if words and words[0] == 'pause':
                        self.linerunners.append(
                            PauseLine(self, srcline, lineno, eval(words[1])))
                        lineno += 1
                        pausenext = False
                    elif words and words[0] == "checkpoint":
                        if self.checkpoints:
                            self.linerunners.append(
                                CheckPointLine(self, srcline, lineno,
                                               words[1].rstrip()))
                            lineno += 1
                        pausenext = False
                    elif words and words[0] == "postpone":
                        self.linerunners.append(
                            PostponeLine(self, srcline, lineno, words[1]))
                        lineno += 1
                    elif logutils.recording() and words and words[0]=="assert":
                        # When rerecording, don't actually run the
                        # tests that may have been inserted in the
                        # log file.
                        ## TODO: When rerecording, assert statements
                        ## should be copied into the log file *after*
                        ## immediately subsequent checkpoints.  That
                        ## is, checkpoints that arise after an
                        ## assertion and before any user action should
                        ## precede the assertion in the log file.
                        self.linerunners.append(
                            CommentLine(self, srcline, lineno, line.rstrip()))
                        lineno += 1
                        pausenext = False
                    else:               # not a special line
                        if pausenext and replaydelay > 0:
                            self.linerunners.append(
                                PauseLine(self, srcline, lineno, replaydelay))
                            lineno += 1
                        self.linerunners.append(
                             PerformLine(self, srcline, lineno, line.rstrip()))
                        lineno += 1
                        pausenext = True
            srcline += 1
        file.close()
        GUILogPlayer.current = self
        self.aborted = False

            
    def __del__(self):
        GUILogPlayer.current = None
    def __call__(self):
        if self.beginCB is not None:
            self.beginCB()
            self.beginCB = None
        if len(self.linerunners) > 0:
            self.linerunners[0].start()
        return False                    # only call this callback once
    def stop(self):
        if logutils.replaying():
            logutils.set_replaying(False)
            if self.finishCB:
                self.finishCB()
    def abort(self):
        self.aborted = True
        self.stop()
    def getLine(self, n):
        if n >= 0 and n < len(self.linerunners):
            return self.linerunners[n]
    def nlines(self):
        return self._nfilelines

# A GUILogLineRunner is in charge of executing a single line of the
# gui log file.

class GUILogLineRunner:
    def __init__(self, logrunner, srcline, lineno):
        self.lineno = lineno            # line number (not counting comments)
        self.srcline = srcline          # line number in source file
        self.logrunner = logrunner
        self.ntries = 0
        self.status = "initialized"
        # insertedLines contains GUILogLineRunners for postponed lines
        # that have been run just before this line.
        self.insertedLines = []
    def start(self):
        # Install ourself as an idle callback.  The callback is set to
        # a low priority, because we're simulating a user's mouse
        # clicks and keyboard events, and users are slow.  If we run
        # at normal priority, the simulated events interfere with
        # normal gtk operation.
        if self.status == "initialized":
            self.status = "installed"
            if logutils.debugLevel() >= 4:
                print("Installing", self.srcline, file=sys.stderr)
            GLib.idle_add(function=self, priority=GLib.PRIORITY_LOW)
    def nextLine(self):
        line = self.logrunner.getLine(self.lineno+1)
        if line:
            return line
        if _postponed:
            return _postponed[-1]
    def nlines(self):
        return self.logrunner.nlines()
    def ready(self):
        # See comments in __call__.
        if not self.insertedLines:
            pl = self.logrunner.getLine(self.lineno-1) # previous line
            return (not pl or
                    pl.status == "done" or
                    (pl.status == "running" and
                     pl.runLevel < logutils.run_level()))
        for line in self.insertedLines:
            if not (line.status == "done" or
                    line.status == "running" and
                    line.runLevel < logutils.run_level()):
                return False
        return True

    def __call__(self):
        # Execute our line of the gui log, *if* the previous line has
        # completed.  Figuring out if the previous line has completed
        # is non-trivial, because the previous line may have emitted a
        # gtk signal that caused a modal dialog box to open, in which
        # case its "emit" call won't return until the box has closed!
        # *This* line contains the commands that operate the dialog
        # box, and must be issued even though the previous command
        # hasn't returned.  If the previous line hasn't returned it
        # must have called Dialog.run or started up a new gtk main
        # loop, so by keeping track of Gtk.main_level() and the number
        # of open dialogs, we can tell when it's time to execute our
        # line.  (This is why we must redefine the Dialog class.)
        if self.logrunner.aborted:
            # The previous line raised an exception, so don't run this
            # line, even though it's already been installed as an idle
            # callback.
            self.status = "aborted"
            return False

        # Run this line, but only if there are no postponed lines
        # ready to go, and if this line is also ready.
        if not self.run_postponed() and self.ready():
            assert self.status in ("repeating", "installed")
            # Add the idle callback for the next line *before*
            # executing our line, because we might not return
            # until after the next line is done!  This is why we
            # need a separate idle callback for each line.
            if self.status != "repeating" and self.nextLine() is not None:
                self.nextLine().start()

            try:
                if self.status == "installed":
                    self.status = "running"
                    self.report()
                self.runLevel = logutils.run_level()
                # self.playback performs some suitable action and
                # returns True if the idle callback should be
                # repeated, and False if it shouldn't.  It can also
                # reinstall the callback, and should set self.status
                # to "done" if the task is finished.
                try:
                    result = self.playback()
                    if self.nextLine() is None:
                        self.logrunner.stop()
                    return result
                except logutils.GtkLoggerTopFailure:
                    # GtkLoggerTopFailures occur when a previous log
                    # line tried to open a window, but the window
                    # hasn't appeared by the time that a subsequent
                    # log line tries to access the window.  A properly
                    # placed checkpoint can ensure that the subsequent
                    # line doesn't execute too soon.  In case that
                    # doesn't work, or if the checkpoint is missing, a
                    # GtkLoggerTopFailure is ignored unless it is
                    # repeated multiple (maxtries) times.
                    self.ntries += 1
                    if self.ntries == maxtries:
                        if logutils.debugLevel() >= 1:
                            print("Failed to find top-level widget after", \
                                  self.ntries, "attempts.", file=sys.stderr)
                        self.status = "aborted"
                        self.logrunner.abort()
                        return False
                    # Keep trying.  By reinstalling ourself in the
                    # idle callback table and returning False
                    # (meaning, "don't repeat this callback") we move
                    # to the back of the queue.  This allows the
                    # widget we are waiting for to appear, we hope.
                    self.status = "repeating"
                    GLib.timeout_add(interval=retrydelay, function=self,
                                     priority=GLib.PRIORITY_LOW)
                    return False


                except Exception as exc:
                    # Any type of exception other than GtkLoggerTopFailure
                    # is fatal.
                    self.status = "aborted"
                    self.logrunner.abort()
                    if self.logrunner.exceptHook:
                        if not self.logrunner.exceptHook(exc, self.srcline):
                            raise exc
                    else:
                        raise exc
            finally:
                Gdk.flush()

        # We're still waiting for the previous line to execute. It's
        # probably getting GtkLoggerTopFailure exceptions (see above).
        # We put ourself at the back of the execution queue (by
        # reinstalling and returning False) so that the previous line
        # will run first.
        if logutils.debugLevel() >= 4:
            print("Reinstalling", self.srcline, file=sys.stderr)
        GLib.timeout_add(interval=retrydelay, function=self,
                         priority=GLib.PRIORITY_LOW)
        return False

    def run_postponed(self):
        ## Attempt to run a postponed line.  Return True if successful.
        if not _postponed:
            return False                # no postponed lines to run
        line = _postponed[-1]
        if line.ready():
            # Put the postponed line in the list of lines that must
            # run before this line is run.
            self.insertedLines.append(line)
            # Start the idle callback that will run the postponed line.
            line.start()
            return True
        return False                    # postponed line isn't ready

class PerformLine(GUILogLineRunner):
    def __init__(self, logrunner, srcline, lineno, line):
        GUILogLineRunner.__init__(self, logrunner, srcline, lineno)
        self.line = line
    def report(self):
        print("////// %d/%d %s" %(self.srcline, self.nlines(),
                                                 self.line), file=sys.stderr)
    def playback(self):
        if logutils.recording():
            ## Hack opportunity: if it's necessary to modify some
            ## lines in existing log files, check for the lines here,
            ## and call loggers._writeline with the modified version.
            ## This will allow log files to be modified by rerecording
            ## them.
            loggers._writeline(self.line)
                
        if logutils.debugLevel() >= 4:
            print("Executing", self.srcline, self.line, file=sys.stderr)
        # Exec'ing the line with an explicitly provided dictionary
        # allows variables created on one line to be available on a
        # later line.  Otherwise, the variable's scope would just be
        # this function call, which wouldn't be very useful.
        exec(self.line, sys.modules[__name__].__dict__)
        if logutils.debugLevel() >= 4:
            print("Finished", self.srcline, self.line, file=sys.stderr)
        self.status = "done"
        return False

_postponed = []

class PostponeLine(GUILogLineRunner):
    def __init__(self, logrunner, srcline, lineno, line):
        GUILogLineRunner.__init__(self, logrunner, srcline, lineno)
        self.line = line
        self.runlevel = logutils.run_level()
    def playback(self):
        if logutils.recording():
            loggers._writeline("postpone " + self.line)
        _postponed.append(PostponedLine(self))
        self.status = "done"
        return False
    def report(self):
        print("////// %d/%d postponing %s" % (self.srcline,
                                                             self.nlines(),
                                                             self.line), file=sys.stderr)

class PostponedLine(PerformLine):
    def __init__(self, ppl):
        # ppl is a PostponeLine object
        PerformLine.__init__(self, ppl.logrunner, ppl.srcline, ppl.lineno,
                             ppl.line)
        self.ppl = ppl
    def ready(self):
        # Overrides PerformLine.ready()
        return logutils.run_level() <= self.ppl.runlevel
    def start(self):
        _postponed.remove(self)
        PerformLine.start(self)
    def playback(self):
        # This is just like PerformLine.playback, but it doesn't echo
        # the line when rerecording, because PostponeLine has already
        # echoed it.
        if logutils.debugLevel() >= 4:
            print("Executing", self.srcline, self.line, file=sys.stderr)
        exec(self.line, sys.modules[__name__].__dict__)
        if logutils.debugLevel() >= 4:
            print("Finished", self.srcline, self.line, file=sys.stderr)
        self.status = "done"
        return False
    def run_postponed(self):
        # Postponed lines never wait for other postponed lines.
        return False
    def report(self):
        print("////// %d/%d (postponed) %s" % (self.srcline,
                                                              self.nlines(),
                                                              self.line), file=sys.stderr)

class CommentLine(GUILogLineRunner):
    def __init__(self, logrunner, srcline, lineno, comment):
        self.comment = comment
        GUILogLineRunner.__init__(self, logrunner, srcline, lineno)
    def playback(self):
        if logutils.recording():
            loggers._writeline(self.comment)
        self.status = "done"
        return False
    def report(self):
        print("###### %d/%d %s" % (self.srcline, self.nlines(),
                                                  self.comment), file=sys.stderr)

class PauseLine(GUILogLineRunner):
    # Special handler for lines of the form "pause <time>".  Such
    # lines allow threads started by earlier commands to complete.
    # The pause is guaranteed to be at least <time> milliseconds long.
    def __init__(self, logrunner, srcline, lineno, delaytime):
        GUILogLineRunner.__init__(self, logrunner, srcline, lineno)
        self.delaytime = delaytime
    def playback(self):
        if logutils.recording():
            loggers._writeline("pause" + self.delaytime)
        if _threaded and self.delaytime > 0:
            if self.status == "running":
                self.status = "repeating"
                if logutils.debugLevel() >= 4:
                    print(self.srcline, \
                          "Pausing", self.delaytime, "milliseconds", file=sys.stderr)
                GLib.timeout_add(interval=self.delaytime, function=self,
                                 priority=GLib.PRIORITY_LOW)
            elif self.status == "repeating":
                if logutils.debugLevel() >= 4:
                    print("Done pausing", self.srcline, file=sys.stderr)
                self.status = "done"
        else:
            # not threaded, no need to wait for background tasks
            self.status = "done"
        return False
    def report(self):
        pass

class CheckPointLine(GUILogLineRunner):
    def __init__(self, logrunner, srcline, lineno, comment):
        GUILogLineRunner.__init__(self, logrunner, srcline, lineno)
        self.comment = comment
    def playback(self):
        # Do NOT echo to the logfile, even if rerecording.  The point
        # of rerecording is to allow checkpoints to be added by the
        # code.
        if checkpoint.check_checkpoint(self.comment):
            if logutils.debugLevel() >= 4:
                print("Reached checkpoint", self.srcline, file=sys.stderr)
            self.status = "done"
        else:
            self.status = "repeating"
            if logutils.debugLevel() >= 4:
                print("Waiting on checkpoint", self.srcline, file=sys.stderr)
            GLib.timeout_add(interval=retrydelay, function=self,
                             priority=GLib.PRIORITY_LOW)
        return False
    def report(self):
        print("////// %d/%d checkpoint %s" %(self.srcline,
                                                            self.nlines(),
                                                            self.comment), file=sys.stderr)
####################

## Functions used within log files.

# findWidget. et al are defined in logutils so that they can easily be
# used elsewhere, too. 
findWidget = logutils.findWidget
findAllWidgets = logutils.findAllWidgets
findMenu = logutils.findMenu
findCellRenderer = logutils.findCellRenderer
setComboBox = logutils.setComboBox

# Utility function for creating a Gdk.Event object. "etype" should be
# a GdkEventType, such as Gdk.EventType.BUTTON_PRESS.  kwargs contains
# attributes that will be assigned to the event.  It almost certainly
# should include the "window" attribute, which must be set to a
# Gdk.Window.  For Gtk.Widgets, this is just Widget.get_window().

def buildEvent(etype, **kwargs):
    ev = Gdk.Event.new(etype)
    if hasattr(ev, 'time'):
        ev.time = Gtk.get_current_event_time()
    if hasattr(ev, 'set_device'):
        disp = Gdk.Display.get_default()
        ev.set_device(disp.list_devices()[0])
    for arg, val in kwargs.items():
        if logutils.debugLevel() > 0:
            if not hasattr(ev, arg):
                print("Event", etype, "has no attribute", arg, file=sys.stderr)
        setattr(ev, arg, val)
    return ev

# event() and wevent() can be used in log files to recreate events.
# The gtk documentation is vague on how to do this correctly, because
# apparently it's not a good idea.  wevent() uses GtkWidget.event(),
# which ensures that the event is associated with the correct widget.
# However the docs say to use Gtk.main_do_event() instead, so that the
# event will behave as if it's in the event queue.  However,
# main_do_event() takes a window and a position as arguments, not a
# widget, and sometimes the wrong widget responds when replaying a log
# file.

def event(etype, **kwargs):
    ev = buildEvent(etype, **kwargs)
    Gtk.main_do_event(ev)

def wevent(widget, etype, **kwargs):
    ev = buildEvent(etype, **kwargs)
    widget.event(ev)


# Pop-up menus sometimes need to be explicitly closed, but it's hard
# to tell when.  deactivatePopup() closes a popup if it can be found,
# and silently returns if it can't.

def deactivatePopup(name):
    try:
        menu = logutils.getTopLevelWidget(name)
    except logutils.GtkLoggerTopFailure:
        return
    menu.deactivate()

# weakRef returns a weak reference to its argument, or if the argument
# is None, it returns a function that returns None.  This way, a log
# file can contain lines like this:
#   widget = weakRef(findWidget(...))
#   if widget(): ...
# even if findWidget(...) might return None.  This also makes it easy to
# avoid using strong references in log files, which might have side effects.

def weakRef(obj):
    if obj is None:
        return noneFunc
    return weakref.ref(obj)

def noneFunc():
    return None

####################

## replayDefine adds an object to the namespace used while replaying
## log files.  If an externally defined GtkLogger needs to invoke an
## externally defined function during replay, that function (or the
## module containing it) should be injected into the namespace using
## replayDefine.

def replayDefine(obj, name=None):
    nm = name or obj.__name__
    sys.modules[__name__].__dict__[nm] = obj

