# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.common import config
from ooflib.SWIG.common import progress
from ooflib.common import debug
from ooflib.common import enum
from ooflib.common import utils
from ooflib.common import version
from ooflib.common.IO import mainmenu
from ooflib.common.IO import menuparser
from ooflib.common.IO import oofmenu
from ooflib.common.IO import parameter
import os.path

##############################

datafileversion = 1.0

##############################

# Data file formats

if not config.nanoHUB():
    class DataFileFormat(enum.EnumClass(
        ('script',
 'A fully functioning Python script.  Flexible and editable, but insecure.'),
        ('ascii', 'An ASCII file with Python-like syntax that will NOT be parsed by the Python interpreter.  Editable and secure, but inflexible.'),
        ('binary', 'A binary file. Inflexible and uneditable, but secure, compact and not subject to round-off error.')
        )):
        tip = "Types of oof2 data files."
        discussion = "<para>Formats for writing data files.</para>"
        xrefs=["Section-Concepts-FileFormats"]

else:                           # in nanoHUB mode scripts aren't allowed
    class DataFileFormat(enum.EnumClass(
        ('ascii', 'An ASCII file with Python-like syntax that will NOT be parsed by the Python interpreter.  Editable and secure, but inflexible.'),
        ('binary', 'A binary file. Inflexible and uneditable, but secure, compact and not subject to round-off error.')
        )):
        tip = "Types of oof2 data files."
        discussion = "<para>Formats for writing data files.</para>"
        xrefs=["Section-Concepts-FileFormats"]
utils.OOFdefine('DataFileFormat', DataFileFormat)


class DataFileFormatExt(
    enum.subClassEnum(DataFileFormat, ('abaqus', 'An ABAQUS-style text file'))):
    tip = "Types of oof2 data files."
    discussion = """<para> Types of oof2 data files, extended for
    &skels; and &meshes;.  </para>"""
    
utils.OOFdefine('DataFileFormatExt', DataFileFormatExt)

# These constants or objects are also instances of DataFileFormat
if not config.nanoHUB():
    SCRIPT = DataFileFormatExt("script")
else:
    SCRIPT = None
ASCII = DataFileFormatExt("ascii")
BINARY = DataFileFormatExt("binary")
ABAQUS = DataFileFormatExt("abaqus")

##############################

def versionCB(menuitem, number, format):
    if format == BINARY:
        menuitem.parser.binaryMode()

versionCmd = oofmenu.OOFMenuItem(
    'FileVersion',
    callback=versionCB,
    params=[parameter.FloatParameter('number',
                                     tip='file format version number'),
            enum.EnumParameter('format', DataFileFormat,
                               tip='format for the data file.')],
    help="Identify data file format.  Used internally in data files.",
    discussion="""
    <para>&oof2; data files must begin with a FileVersion command.
    The <varname>number</varname> parameter is used to maintain
    compatibility with older data files.  For now, its value should be
    <userinput>1.0</userinput>.  The <varname>format</varname>
    parameter must be one of the values discussed in <xref
    linkend='Section-Concepts-FileFormats'/>.</para>
    """)

mainmenu.OOF.LoadData.addItem(versionCmd)

class AsciiDataFile:
    def __init__(self, file, format):
        self.format = format
        self.file = file
        self.nargs = 0
        self.buffer = ""
    def startCmd(self, command):
        path = command.path()
        if self.format == ASCII:
            path = '.'.join(path.split('.')[2:])
        self.buffer = path + "("
        self.nargs = 0
    def endCmd(self):
        self.file.write(self.buffer)
        self.file.write(")\n")
        self.file.flush()
        self.buffer = ""
    def discardCmd(self):
        self.buffer = ""
        self.nargs = 0
    def argument(self, name, value):
        if self.nargs > 0:
            self.buffer += ", "
        self.buffer += "%s=%s" % (name, repr(value))
        self.nargs += 1
    def comment(self, remark):
        self.file.write("# %s\n" % remark)
    def close(self):
        self.file.close()
    def flush(self):
        self.file.flush()

def writeDataFile(filename, mode, format):
    if format == BINARY:
        mode += 'b'
    file = open(filename, mode)
    if format == SCRIPT:
        versioncmd = "OOF.LoadData.FileVersion"
    else:
        versioncmd = "FileVersion"
    header = "# OOF version %s\n%s(number=%s, format=%s)\n" \
               % (version.version, versioncmd, datafileversion, repr(format))

    if format != BINARY:
        file.write(header)
        return AsciiDataFile(file, format)

    from ooflib.common.IO import binarydata    # avoid import loop
    file.write(bytes(header, "UTF-8"))
    return binarydata.BinaryDataFile(file)

def readDataFile(filename, menu):
    prog = progress.getProgress(os.path.basename(filename), progress.DEFINITE)
    source = None
    try:
        source = menuparser.ProgFileInput(filename, prog)
        parser = menuparser.MenuParser(source, menu)
        parser.run()
    finally:
        if source is not None:  # in case an error occured in creating source
            source.close()
        prog.finish()
