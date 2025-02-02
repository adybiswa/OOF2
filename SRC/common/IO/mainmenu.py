# -*- python -*-


# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

#  The main oof menu -- with recording capability and so forth.

# Lots of miscellaneous commands are defined in this file because they
# don't merit a file of their own.

## TODO: Add a command that resets everything, deletes all
## Microstructures, resets the factory settings for Properties, etc.

from ooflib.SWIG.common import config
from ooflib.SWIG.common import guitop
from ooflib.SWIG.common import ooferror
from ooflib.SWIG.common import progress
from ooflib.SWIG.common import switchboard
from ooflib.SWIG.common import threadstate
from ooflib.common import debug
from ooflib.common import runtimeflags
from ooflib.common import subthread
from ooflib.common import thread_enable
from ooflib.common import utils
from ooflib.common.IO import filenameparam
from ooflib.common.IO import oofmenu
from ooflib.common.IO import parameter
from ooflib.common.IO import reporter
from ooflib.common.IO import scriptloader
from ooflib.common.IO.gfxmanager import gfxManager
from ooflib.common.IO import xmlmenudump

import code
import sys
import tempfile
import atexit
import os
import os.path

# Parameter = parameter.Parameter
StringParameter = parameter.StringParameter
IntParameter = parameter.IntParameter
FloatParameter = parameter.FloatParameter

OOFMenuItem = oofmenu.OOFMenuItem
OOFRootMenu = oofmenu.OOFRootMenu
CheckOOFMenuItem = oofmenu.CheckOOFMenuItem

OOF = OOFRootMenu(
    'OOF',
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/oof.xml'))

################

# Automatically log all menu commands to a temp file. The location of
# the temp file is taken from the OOFTMP environment variable, if it's
# defined.  Otherwise it's determined by the tempfile.mkstemp
# function, which looks in the environment variables TMPDIR, TEMP, and
# TMP, followed by the directories /tmp, /var/tmp, and /usr/tmp.

ooftmpdir = os.getenv('OOFTMP')
if ooftmpdir is not None:
    fd, logpath = tempfile.mkstemp(prefix='oof2-', suffix='.py', dir=ooftmpdir)
else:
    fd, logpath = tempfile.mkstemp(prefix='oof2-', suffix='.py')
tmplogfile = os.fdopen(fd, 'w')

def _tmplog(s):
    if tmplogfile:
        print(s, file=tmplogfile)
        tmplogfile.flush()
    
OOF.addLogger(oofmenu.MenuLogger(_tmplog))

# Remove the log file automatically if the program exits cleanly
def cleanlog():
    global tmplogfile, logpath
    if tmplogfile:
        try:
            tmplogfile.close()
            os.remove(logpath)
        except:
            pass
        logpath = ""
        tmplogfile = None
atexit.register(cleanlog)


################

## File menu

_filemenu = OOF.addItem(OOFMenuItem(
    'File',
    help="Commands for saving and loading data, and quitting.",
    discussion=xmlmenudump.emptyDiscussion))

_loadmenu = _filemenu.addItem(OOFMenuItem(
    'Load',
    ordering = 0,
    help="Commands for loading datafiles and scripts.",
    discussion="<para>Commands to load datafiles and scripts.</para>"))

# Commands in _startupmenu are identical to commands in _loadmenu, but
# they have their own copies of the parameters.  This prevents startup
# file names from setting the default values of parameters in the load
# menu.
_startupmenu = _filemenu.addItem(OOFMenuItem(
    'LoadStartUp',
    secret=True,
    help="Load start-up datafiles and scripts.",
    discussion="""<para> Commands to load datafiles and scripts from
    command line arguments at start-up time. These commands are the
    same as the ones in <xref linkend="MenuItem-OOF.File.Load"/> but
    have their own copies of the parameters to avoid affecting default
    values. </para>""",
    xrefs=["Section-Running"]
))

class PScriptLoader(scriptloader.ScriptLoader):
    # A ScriptLoader that supports a progress bar.
    def __init__(self, filename, **kwargs):
        self.prog = progress.getProgress(os.path.basename(filename),
                                          progress.DEFINITE)
        scriptloader.ScriptLoader.__init__(
            self,
            filename=filename,
            locals=sys.modules['__main__'].__dict__,
            **kwargs)
    def progress(self, current, total):
        self.prog.setFraction((1.0*current)/total)
        if current <= total:
            self.prog.setMessage("Read %d/%d lines" % (current, total))
        else:
            self.prog.setMessage("Done")
    def stop(self):                     # called by ScriptLoader. Abort loop?
        return self.prog.stopped()
    def done(self):                     # called by ScriptLoader when finished
        self.prog.finish()
        scriptloader.ScriptLoader.done(self)

subScriptErrorHandler = None       # redefined if GUI is loaded

def loadscript(menuitem, filename):
    if filename is not None:
        debug.fmsg('reading', filename, 'in thread',
                   threadstate.findThreadNumber())
        kwargs = {}
        if subScriptErrorHandler:
            kwargs['errhandler'] = subScriptErrorHandler
        interp = PScriptLoader(filename, **kwargs)
        interp.run()
        if interp.error:
            # If the interpreter raised an exception and we're in
            # batch mode, the shell error status won't be set unless a
            # new exception is raised here.  The old exception has
            # already been handled by the time we get to this point.
            # interp.error is the result of sys.exc_info() when the
            # error occurred.
            errorname = interp.error[0].__name__
            if errorname.lower()[0] in "aeiou":
                article = "an"
            else:
                article = "a"
            raise ooferror.PyErrUserError(
                f"Script {filename} raised {article} "
                f"{interp.error[0].__name__} exception")
        debug.fmsg('finished reading', filename)

_loadmenu.addItem(OOFMenuItem(
    'Script',
    callback=loadscript,
    params=[filenameparam.ReadFileNameParameter('filename', 'logfile',
                                                tip="Name of the file.",
                                                ident="load")],
    no_log=1,
    ellipsis=1,
    accel='l',
    help="Execute a Python script.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/loadscript.xml'),
    xrefs=["Section-Running"]
    ))

_startupmenu.addItem(OOFMenuItem(
    'Script',
    callback=loadscript,
    params=[filenameparam.ReadFileNameParameter('filename', 'logfile',
                                                tip="Name of the file.",
                                                ident="load")],
    no_log=1,
    ellipsis=1,
    accel='l',
    disabled=config.nanoHUB(),  # loading arbitrary scripts is a
                                # security hole on nanoHUB
    help="Execute a Python script.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/loadscript.xml'),
    ))

def loadmodule(menuitem, module):
    if module is not None:
        exec(f"import {module}")

_loadmenu.addItem(OOFMenuItem(
    'Module',
    callback=loadmodule,
    params=[parameter.StringParameter("module",
                                      tip="The name of a Python module")],
    ellipsis=1,
    help="Load a Python module.",
    discussion="""<para>
    Load a Python module, such as an OOF extension module.  The module
    must be located within the Python search path.  The search path
    can be adjusted by setting the <varname>PYTHONPATH</varname>
    environment variable.
    </para>"""))

# TODO: A "Load/Extension" command, to make loading extension modules
# easier.  It would list all of the extensions found in
# SRC/EXTENSIONS, plus any others found in additional paths that the
# user could define.

# "Data" files are distinguished from scripts in that they're not read
# by the Python interpreter directly, and therefore can't contain
# arbitrary Python code.  They can only contain menu commands from the
# OOF.LoadData menu.  The command arguments can be constants or
# variables and functions that are defined in the main OOF namespace,
# or lists and tuples thereof.

def loaddata(menuitem, filename):
    if filename is not None:
        from ooflib.common.IO import datafile
        datafile.readDataFile(filename, OOF.LoadData)

_loadmenu.addItem(OOFMenuItem(
    'Data',
    callback=loaddata,
    threadable=oofmenu.THREADABLE,
    params=[filenameparam.ReadFileNameParameter('filename', ident="load",
                                                tip="Name of the file.")],
    ellipsis=1,
    help="Load a data file.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/loaddatafile.xml'),
    xrefs=["Section-Running"]
    ))

_startupmenu.addItem(OOFMenuItem(
    'Data',
    callback=loaddata,
    threadable=oofmenu.THREADABLE,
    params=[filenameparam.ReadFileNameParameter('filename', ident="load",
                                                tip="Name of the file.")],
    ellipsis=1,
    help="Load a data file.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/loaddatafile.xml'),
    ))

OOF.addItem(oofmenu.OOFMenuItem(
    "LoadData", secret=1,
    help="Commands used in data files.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/loaddata.xml'),
    post_hook=None, # Don't include checkpoints in gui logs
    xrefs=["MenuItem-OOF.File.Load.Data", "MenuItem-OOF.File.Save"]
    ))


def saveLog(menuitem, filename, mode):
    file = open(filename, mode.string())
    menuitem.root().saveLog(file)
    file.close()

_savemenu = _filemenu.addItem(OOFMenuItem(
    'Save',
    ordering=1,
    help='Create data files and scripts.',
    discussion=xmlmenudump.emptyDiscussion))

_savemenu.addItem(OOFMenuItem(
    'Python_Log',
    callback=saveLog,
    ordering=10,
    params=[filenameparam.WriteFileNameParameter('filename', ident="load",
                                                 tip="Name of the file."),
            filenameparam.WriteModeParameter(
                'mode',
                tip="Whether to overwrite or append to an existing file.")],
    accel='s',
    ellipsis=1,
    no_log=1,
    help="Save the current session as a Python script.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/savelog.xml'),
    xrefs=["MenuItem-OOF"]
    ))

#################################

def quitCmd(menuitem):
    from ooflib.common import quit
    quit.quit()

_filemenu.addItem(OOFMenuItem(
    'Quit',
    callback=quitCmd,
    accel='q',
    ordering=10000,
    help="Don't give up so easily!",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/quit.xml'),
    threadable = oofmenu.UNTHREADABLE,
    no_log=1
    ))

##################################

settingsmenu = OOF.addItem(OOFMenuItem(
    'Settings',
    help="Global settings",
    discussion="""
    <para>
    Commands for setting parameters that don't belong anywhere else.
    </para>"""))

fontmenu = settingsmenu.addItem(OOFMenuItem(
    "Fonts",
    ordering=1,
    help="Set fonts used in the GUI.",
    discussion=xmlmenudump.emptyDiscussion
    ))

def setFont(menuitem, fontname):
    switchboard.notify('change font', fontname)

fontmenu.addItem(OOFMenuItem(
    "Widgets",
    callback=setFont,
    params=[parameter.StringParameter('fontname', tip="The name of a font.")],
    help="Set the font to use for labels, menus, buttons, etc. in the graphical user interface.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/widgetfont.xml')
    ))

# Any Gtk widget that wants a fixed width font should set its CSS name
# to "fixedfont".
# The fixed font size is stored as a global here just
# so that the GUI can retrieve its initial value.
fixedFontSize = 12

def setFixedFont(menuitem, fontsize):
    global fixedFontSize
    fixedFontSize = fontsize
    switchboard.notify('change fixed font', fontsize)

fontmenu.addItem(OOFMenuItem(
    "Fixed",
    callback=setFixedFont,
    params=[parameter.PositiveIntParameter('fontsize', fixedFontSize,
                                           tip='Font size, in pixels.')],
    help="Set the size of fixed-width font to use in text displays.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/textfont.xml')
    ))

def setTheme(menuitem, theme):
    switchboard.notify('change theme', theme)

settingsmenu.addItem(OOFMenuItem(
    "Theme",
    callback=setTheme,
    ordering=2,
    params=[parameter.StringParameter('theme',
                                      tip="The name of a gnome theme.")],
    help="Set the look and feel of the graphical user interface.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/theme.xml')
    ))

bufsizemenu = settingsmenu.addItem(OOFMenuItem(
    "UndoBuffer_Size",
    ordering=8,
    help="Set the size of history buffers.",
    discussion="""<para>
    Many operations in &oof2; save data in a buffer so that the
    operation can be reversed with an <guibutton>Undo</guibutton>
    button.  This menu contains commands that set the size of the
    buffer.  Using a larger buffer make more operations undoable, but
    uses more memory.
    </para>"""
))

gfxdefaultsmenu = settingsmenu.addItem(OOFMenuItem(
    "Graphics_Defaults",
    ordering=4,
    help="Set various default parameters for graphics displays.",
    discussion="""<para>
    This menu contains commands for setting the default values of
    various parameters controlling how things are displayed in the
    graphics window.  Put these commands into your &oof2rc; file
    to set defaults for every &oof2; session.
    </para>""",
    xrefs=["Chapter-Graphics"]
))
gfxdefaultsmenu.addItem(OOFMenuItem(
    "Pixels",
    ordering=1,
    help="Set default parameters for displaying pixels.",
    discussion="""<para>

    This menu contains commands for setting the default values of
    various parameters controlling how pixels (from &images; and
    &micros;) are displayed in the graphics window.  Put
    these commands into your &oof2rc; file
    to set defaults for every &oof2; session.

    </para>"""))
gfxdefaultsmenu.addItem(OOFMenuItem(
    "Skeletons",
    ordering=2,
    help="Set default parameters for displaying Skeletons.",
    discussion="""<para>

    This menu contains commands for setting the default values of
    various parameters controlling how &skels; and &skel;
    components are displayed in the graphics window.  Put these
    commands into your &oof2rc; file
    to set defaults for every &oof2; session.

    </para>"""))
gfxdefaultsmenu.addItem(OOFMenuItem(
    "Meshes",
    ordering=3,
    help="Set default parameters for displaying Meshes.",
    discussion="""<para>

    This menu contains commands for setting the default values of
    various parameters controlling how &meshes; and &mesh;
    components are displayed in the graphics window.  Put these
    commands into your &oof2rc; file
    to set defaults for every &oof2; session.

    </para>"""))

from ooflib.SWIG.common import crandom

def _randomseed(menuitem, seed):
    crandom.rndmseed(seed)

settingsmenu.addItem(oofmenu.OOFMenuItem(
    'Random_Seed',
    callback=_randomseed,
    ordering=3.5,
    params=[parameter.IntParameter('seed', 17, tip=parameter.emptyTipString)],
    help="Seed the random number generator.",
    discussion=xmlmenudump.loadFile("DISCUSSIONS/common/menu/randomseed.xml")
))

def _setdigits(menuitem, digits):
    runtimeflags.setDigits(digits)

settingsmenu.addItem(oofmenu.OOFMenuItem(
    'GUI_Digits',
    callback=_setdigits,
    ordering=3,
    params=[parameter.NonNegativeIntParameter(
        'digits', runtimeflags.digits(),
        tip='Number of digits to show after the decimal point.')],
    help="Set the precision for numbers in the GUI.",
    discussion=xmlmenudump.loadFile("DISCUSSIONS/common/menu/digits.xml")
))

################################

def _annotatelogmenucallback(menutiem, message):
    reporter.report(message)

_annotatelogmenu = _filemenu.addItem(OOFMenuItem(
    'Annotate_Log',
    help="Write a message to the log file",
    discussion=xmlmenudump.loadFile("DISCUSSIONS/common/menu/annotatelog.xml"),
    callback = _annotatelogmenucallback,
    ordering=200,
    params=[parameter.StringParameter(name="message",
                                      tip="Data to include in the log file."),]
    ))

##################################

## Subwindows menu.

_windowmenu = OOFMenuItem(
    'Windows',
    help="Menus for opening and raising windows.",
    discussion=xmlmenudump.emptyDiscussion)
OOF.addItem(_windowmenu)

def dummy(menuitem): pass   # Dummy callback, so trivial menu items get logged.

# Add an entry for the main window, which, when clicked, raises it.
_windowmenu.addItem(OOFMenuItem(
    "OOF2",
    callback=dummy,
    help="Raise the main OOF2 window.",
    discussion="""<para>
    Every &oof2; subwindow has this menu available. It's a good way to
    locate the main &oof2; window, if it's out of sight.
    </para>"""))

# The Console is a no-op in text mode.

def consolation(menuitem):
    print("There, there, I'm sure everything will be fine.")

_windowmenu.addItem(OOFMenuItem(
    'Console',
    callback=consolation,
    help="Open or raise the Python console interface.",
    no_log=1,
    disabled=config.nanoHUB(),  # executing arbitrary python is a
                                # security hole on nanoHUB.
    discussion="""<para>
    The &oof2; <link linkend='Section-Windows-Console'>Console</link>
    provides a way of executing arbitrary Python code while running
    &oof2; in graphics mode.  All of the OOF2 objects that are
    available in scripts are available in the Console.
    </para>"""))

##################

_graphicsmenu = OOFMenuItem(
    'Graphics',
    help="Graphical display of &oof2; objects.",
    discussion="""<para>
    Graphics windows are discussed in <xref
    linkend='Chapter-Graphics'/>.
    </para>""" )

def openGfx(menuitem):
    window = gfxManager.openWindow()
    window.drawAtTime(time=window.latestTime(), zoom=True)
    
_graphicsmenu.addItem(OOFMenuItem(
    "New",
    callback=openGfx,
    help="Open a new graphics window.",
    accel='g',
    discussion="""<para>Create a new <link
    linkend='Chapter-Graphics'>Graphics Window</link>.</para>"""
))

_windowmenu.addItem(_graphicsmenu)

_windowmenu.addItem(OOFMenuItem(
    'Activity_Viewer',
    callback=dummy,
    help="Open or raise the Activity Viewer window.",
    discussion="""<para>
Raise the <link linkend='Section-Windows-ActivityViewer'>Activity
Viewer</link> window, if it is open. If not, open it.
</para>"""
    ))

#################################

## Help menu

helpmenu = OOF.addItem(OOFMenuItem(
    'Help',
    help_menu=1,
    help="Tutorials, helpful utilities, and debugging tools.",
    discussion=
    """<para>
    Tutorials, helpful utilties, debugging tools, and documentation
    tools.  
    </para>"""
))

debugmenu = helpmenu.addItem(OOFMenuItem(
    'Debug',
    help="Tools for debugging.",
    discussion= """<para>Tools for figuring out what's going on when
    it's not going well.  Mostly of interest to the developers.  Some
    of these commands only appear in the GUI if &oof2; is started with
    the <link
    linkend="Section-Running"><userinput>--debug</userinput></link>
    option.</para>""",
    ordering=10000
    ))

def set_debug(menuitem, state):
    if state:
        debug.set_debug_mode()
    else:
        debug.clear_debug_mode()
        
debugmenu.addItem(CheckOOFMenuItem(
    'Debug',
    debug.debug(),
    callback=set_debug,
    help='Turn debugging mode on and off.',
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/debug.xml')
    ))

debugmenu.addItem(CheckOOFMenuItem(
    'Verbose_Switchboard',
    switchboard.switchboard.verbose,
    callback=switchboard.verbose,
    help='Print all switchboard calls as they occur.',
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/verbosesb.xml'),
    xrefs=["MenuItem-OOF.Help.Debug.Switchboard_Stack_Tracking"]
    ))

debugmenu.addItem(CheckOOFMenuItem(
    'Switchboard_Stack_Tracking',
    switchboard.useMessageStackFlag,
    callback=switchboard.useMessageStackCB,
    help='Keep track of current switchboard calls.',
    discussion="""<para>
    Keep track of which switchboard calls have led to other
    switchboard calls, making it easier to find cause and effect when
    debugging.  See class <classname>MessageStack</classname> in
    <filename>SRC/common/switchboard.spy</filename> in the source
    code.
    </para>""",
    xrefs=["MenuItem-OOF.Help.Debug.Verbose_Switchboard"]
))
    

if debug.debug():
    debugmenu.addItem(oofmenu.OOFMenuItem(
        "Sandbox",
        callback=None,
        accel='d',
        threadable=oofmenu.UNTHREADABLE,
        help="Open the sandbox window.",
        discussion="""<para>A place for developers to play with gtk
        code in the &oof2; environment.</para>"""
    ))
        
####

def _startMemMonitor(menuitem, filename):
    utils.startMemoryMonitor(filename)

def _stopMemMonitor(menuitem):
    utils.stopMemoryMonitor()

memmenu = debugmenu.addItem(OOFMenuItem(
    'Memory_Monitor',
    help="Debug memory use",
    discussion="""

    <para> A tool for debugging memory issues.  Place calls to <code
    language='python'>utils.memusage(comment)</code> in Python files
    at the points where you want memory usage to be printed, rebuild
    &oof2;, and call <xref
    linkend="MenuItem-OOF.Help.Debug.Memory_Monitor.Start"/> and <xref
    linkend="MenuItem-OOF.Help.Debug.Memory_Monitor.Stop"/> while
    running it.  </para>

    <para>This currently does not work on Macintosh.</para>

    <para>See <filename>SRC/common/utils.py</filename>.</para>
    
    """
))

memmenu.addItem(OOFMenuItem(
    'Start',
    callback=_startMemMonitor,
    params=[
        filenameparam.WriteFileNameParameter(
            "filename", tip="Log file name.")],
    help="Start logging memory use.",
    discussion=xmlmenudump.emptyDiscussion
    ))

memmenu.addItem(OOFMenuItem(
    'Stop',
    callback=_stopMemMonitor,
    help="Stop logging memory use.",
    discussion=xmlmenudump.emptyDiscussion
))

####

def setWarnPopups(menuitem, value):
    reporter.messagemanager.set_warning_pop_up(value)

helpmenu.addItem(CheckOOFMenuItem(
    'Popup_warnings',1,
    ordering=-8,
    callback=setWarnPopups,
    help="Display warnings in a pop-up window or just in the message window?",
    discussion="""<para>
    If <command>Popup_warnings</command> is
    <userinput>true</userinput>, warning messages will appear in an
    annoying pop-up window.  If it's <userinput>false</userinput>,
    they'll appear only in the <link
    linkend='Section-Windows-Messages'>Messages</link> window.  The
    default value is <userinput>true</userinput>.
    </para>"""))

## TODO: Remove No_Warnings.  Instead, add Warning_Mode, which can be
## set to IGNORE, NOTIFY, or FATAL.

def setWarnErrors(menuitem, value):
    reporter.messagemanager.set_warning_error(value)

helpmenu.addItem(CheckOOFMenuItem(
        'No_Warnings', 0,
        ordering=-7,
        callback=setWarnErrors,
        help="Treat warnings as errors.",
        discussion="""<para>
        If <command>No_Warnings</command> is
        <userinput>true</userinput>, warning messages are treated as
        errors and will abort the current calculation.  The default
        value is <userinput>false</userinput>.
        </para>"""))

# def testBars1(menuitem):
#     import time
#     prog = progress.getProgress("main", progress.DEFINITE)
#     yprog = progress.getProgress("why", progress.DEFINITE)
#     xmax = 100
#     ymax = 10000
#     for x in xrange(xmax+1):
# #        reporter.report("x=", x)
#         time.sleep(0.1)
#         prog.setMessage("xprog: " + `xmax-x`)
#         prog.setFraction(float(x)/xmax)
#         if prog.stopped():
#             break
#         for y in xrange(ymax+1):
#             yprog.setMessage("yprog: " + `x` + '/' + `y`)
#             yprog.setFraction(float(y)/ymax)
#             if yprog.stopped():
#                 break
#     yprog.finish()
#     prog.finish()

# def testBars2(menuitem):
#     import time
#     from ooflib.SWIG.common import progress
#     prog = progress.getProgress("main", progress.INDEFINITE)
#     xmax = 1000
#     for x in xrange(xmax):
# #        reporter.report("x=", x)
#         time.sleep(0.1)
#         prog.setMessage("testBars: " + `x`)
#         prog.pulse()
#         if prog.stopped():
#             break
#     prog.finish()

# debugmenu.addItem(OOFMenuItem('Bar1', callback=testBars1))
# debugmenu.addItem(OOFMenuItem('Bar2', callback=testBars2))

####################################

## Profiling functions, in the Debug menu

profmenu = OOFMenuItem('Profile')

## TODO: The profiling menu is commented out because it doesn't work
## anymore.  Fix it and uncomment the next line.
# debugmenu.addItem(profmenu)
prof = None

def profile_start(menuitem, filename, fudge):
    global prof
##    import hotshot
##    prof = hotshot.Profile(filename)
##    prof.start()
    if thread_enable.enabled():
        reporter.warn("Multithreaded profiling is unreliable!\nUse the --unthreaded startup option.")
    from ooflib.common.EXTRA import profiler
    prof = profiler.Profiler(filename, fudge=fudge)

def profile_stop(menuitem):
    global prof
    prof.stop()
##    prof.close()
    prof = None
    
profmenu.addItem(OOFMenuItem(
    'Start',
    callback=profile_start,
    threadable = oofmenu.UNTHREADABLE,
    params=[StringParameter('filename', 'prof.out', tip="File name."),
            FloatParameter('fudge', 2.89e-6, tip="Fudge factor.")],
    help="Begin measuring execution time.",
    discussion=xmlmenudump.loadFile('DISCUSSIONS/common/menu/profileStart.xml')
    ))
    
profmenu.addItem(OOFMenuItem(
    'Stop',
    callback=profile_stop,
    threadable = oofmenu.UNTHREADABLE,
    help="Stop measuring execution time.",
    discussion="""<para>Stop profiling, and save the data in the file
    specified in <xref
    linkend='MenuItem-OOF.Help.Debug.Profile.Start'/>.</para>"""
                             ))

def proffudge(menuitem, iterations):
    from ooflib.common.EXTRA import profiler
    fudge = profiler.calibrate_profiler(iterations)
    helpmenu.Debug.Profile.Start.get_arg('fudge').value = fudge
    reporter.report('fudge =', fudge)

profmenu.addItem(OOFMenuItem(
    'FudgeFinder',
    callback=proffudge,
    threadable = oofmenu.UNTHREADABLE,
    params=[IntParameter('iterations', 1000, tip="Number of iterations.")],
    help='Find the factor to compensate for time spent in the profiler itself.',
    discussion="""
    <para>Find the machine dependent fudge factor that the profiler
    uses to compensate for function calling overhead in the profiler
    itself, by measuring how long it takes to call the profiler
    <varname>iterations</varname> times.  </para>
    """
    ))




####################################

## The following functions are visible only if --debug is provided on
## the command line at start up.  Users shouldn't be interested in
## them.  Some of them *are* used in the test suites, so the functions
## have to be present even if -debug isn't used.

def _noop(menuitem):
    pass

debugmenu.addItem(OOFMenuItem('NoOp', no_doc=True,
                              secret=not debug.debug(),
                              callback=_noop))

errmenu = debugmenu.addItem(OOFMenuItem('Error', no_doc=1, 
                                        secret=not debug.debug()))


def _warning(menuitem):
        reporter.warn("You'd better be home by 11, young lady!")

errmenu.addItem(OOFMenuItem('Warning', callback=_warning,
                            help='Actual numbers may vary.'))

from ooflib.SWIG.common import cdebug
def _segfault(menuitem, delay):
    cdebug.segfault(delay)

errmenu.addItem(OOFMenuItem('SegFault', callback=_segfault,
                             threadable=oofmenu.THREADABLE,
                             params=[IntParameter('delay', 10,
                                                  tip="Delay time.")],
                             help='For external use only.  Slippery when wet.'))

def _divzero(menuitem):
    x = 0
    y = 1/x

errmenu.addItem(OOFMenuItem('DivideByZero', callback=_divzero,
                            help="Not recommended"))

def _pyerror(menuitem):
    raise RuntimeError("Oops!")

errmenu.addItem(OOFMenuItem('PyError', callback=_pyerror,
                             threadable=oofmenu.THREADABLE,
                             help='Do not taunt PyError.'))

def _pyerror2(menuitem):
    raise ooferror.PyErrPyProgrammingError("Do not!")

errmenu.addItem(OOFMenuItem('PyError2', callback=_pyerror2,
                            threadable=oofmenu.THREADABLE,
                            help='Do not taunt PyError!'))

def _cerror(menuitem):
    cdebug.throwException()

errmenu.addItem(OOFMenuItem('CError', callback=_cerror,
                             threadable=oofmenu.THREADABLE))

def _cpyerror(menuitem):
    cdebug.throwPythonException()

errmenu.addItem(OOFMenuItem('CPyError', callback=_cpyerror,
                            threadable=oofmenu.THREADABLE))

def _cpycerror(menuitem):
    cdebug.throwPythonCException()

errmenu.addItem(OOFMenuItem("CPyCError", callback=_cpycerror,
                            threadable=oofmenu.THREADABLE))


def loop(menuitem):
    while True:
        pass
    debug.fmsg("What am I doing here?")

errmenu.addItem(OOFMenuItem('Infinite_Loop', callback=loop,
                             threadable=oofmenu.THREADABLE,
                             help="I hope you have lots of time."))

# This was used to introduce a delay in a script at some point.  I
# don't remember why.  It shouldn't be cluttering up the menus or the
# manual.

# def spinCycle(menuitem, nCycles):
#     cdebug.spinCycle(nCycles)

# debugmenu.addItem(OOFMenuItem(
#     'SpinCycle', callback=spinCycle,
#     params=[IntParameter('nCycles', 100000, tip="How many cycles to run.")],
#     help="Eat up some cpu cycles.",
#     discussion="<para>I don't remember why this was needed.</para>"))

import os
from ooflib.SWIG.common import lock
import time

rw = lock.RWLock()

lockmenu = debugmenu.addItem(OOFMenuItem("LockTest", no_doc=True,
                                         secret=not debug.debug()))

def _py_read(menuitem, seconds):
    global rw
    rw.read_acquire()
    print("Got read permission for %d seconds." % seconds)
    time.sleep(seconds)
    print("Releasing read.")
    rw.read_release()

def _py_write(menuitem, seconds):
    global rw
    rw.write_acquire()
    print("Got write permission for %d seconds." % seconds)
    time.sleep(seconds)
    print("Releasing write.")
    rw.write_release()

lockmenu.addItem(OOFMenuItem('RWLock_read', callback=_py_read,
                             no_doc=1,
                             threadable=oofmenu.THREADABLE,
                             params=[IntParameter('seconds', 10,
                                                  tip="Sleeping time.")],
                             help='Safe when used as directed.  For entertainment purposes only.'))

lockmenu.addItem(OOFMenuItem('RWLock_write', callback=_py_write,
                             no_doc=1,
                             threadable=oofmenu.THREADABLE,
                             params=[IntParameter('seconds', 10,
                                                  tip="Sleeping time.")],
                             help='Packaged by weight, contents may settle during shipping.'))


def _wait(menuitem, seconds):
    cdebug.wait(seconds)

lockmenu.addItem(OOFMenuItem('Wait', callback=_wait,
                             no_doc=1,
                             threadable=oofmenu.THREADABLE,
                             params=[IntParameter('seconds', 10,
                                                  tip="Waiting time.")],
                             help='For internal use only.'))


def _random(menuitem, n):
    print([crandom.irndm() for i in range(n)], file=sys.stderr)

debugmenu.addItem(OOFMenuItem(
        'Random',
        callback=_random,
        secret=not debug.debug(),
        params=[IntParameter('n', 10, tip='How many numbers to generate.')],
        help='Generate some random numbers.',
    discussion="""<para>For debugging the random number generator.
    Being able to generate reproducible random numbers is important
    for the test suite.</para>"""
))
