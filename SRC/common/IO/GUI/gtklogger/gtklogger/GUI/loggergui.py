# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

## This is a gui for gtklogger, to be used when recording log files.
## It must be run as a separate process, not as part of the program
## being instrumented.  The output from gtklogger should be piped to
## this program.  This program takes the name of the log file as a
## command line argument.

## The gui allows comments to be inserted into the log file as the log
## is recorded, making it easier to instrument it later.
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GObject
from gi.repository import Gtk
import sys

class GUI:
    def __init__(self):
        global logfile, logfilename
        window = Gtk.Window(Gtk.WindowType.TOPLEVEL,
                            title="gtklogger:" + logfilename)
        window.connect('delete-event', self.quit)
        box = gtk.VBox()
        window.add(box)

        logscroll = Gtk.ScrolledWindow()
        logscroll.set_shadow_type(Gtk.ShadowType.IN)
        box.pack_start(logscroll, expand=True, fill=True)
        self.logtextview = Gtk.TextView(editable=False, cursor_visible=False)
        self.logtextview.set_wrap_mode(Gtk.WrapMode.WORD)
        logscroll.add(self.logtextview)

        commentbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,
                             spacing=2)
        box.pack_start(commentbox, expand=False, fill=False)
        
        self.commentText = Gtk.Entry()
        commentbox.pack_start(self.commentText,
                              expand=True, fill=True, padding=0)
        
        self.commentButton = Gtk.Button("Comment")
        commentbox.pack_start(self.commentButton, expand=False, fill=False,
                              padding=0)
        self.commentButton.connect('clicked', self.commentCB)

        self.clearButton = Gtk.Button('Clear')
        commentbox.pack_start(self.clearButton, expand=False, fill=False,
                              padding=0)
        self.clearButton.connect('clicked', self.clearCB)

        window.show_all()

    def addLine(self, line):
        buffer = self.logtextview.get_buffer()
        buffer.insert(buffer.get_end_iter(), line)
        mark = buffer.create_mark(None, buffer.get_end_iter())
        self.logtextview.scroll_mark_onscreen(mark)
        buffer.delete_mark(mark)
    
    def quit(self, *args):
        gtk.main_quit()
        logfile.close()
    
    def clearCB(self, button):
        self.commentText.set_text("")

    def commentCB(self, button):
        global logfile
        text = '# ' + self.commentText.get_text()
        print >> logfile, text
        self.addLine(text+'\n')

inbuf = ""
def inputhandler(source, condition):
    global inbuf
    line = sys.stdin.readline()
    if not line:
        gui.quit()
        return False
    if line[-1] == '\n':
        fullline = inbuf + line
        gui.addLine(fullline)
        print >> logfile, fullline,
        logfile.flush()
        inbuf = ""
    else:
        inbuf += line
    return True

def start(name):
    global logfilename, logfile, gui
    logfilename = name
    logfile = file(logfilename, "w")
    gobject.io_add_watch(sys.stdin, gobject.IO_IN, inputhandler)
    gui = GUI()
    gtk.main()
    
if __name__ == "__main__":
    start(sys.argv[1])
