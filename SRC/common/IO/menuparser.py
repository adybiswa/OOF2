# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

# The MenuParser class reads a file or other input source and executes
# the OOFMenu commands it contains.  The parser has two Modes: ascii
# and binary.  In ascii mode the input must be ascii characters.  In
# binary mode the input is (surprise) binary.  In neither mode is the
# input eval'd by the Python interpreter.

# The inputsource argument to the MenuParser's __init__ is an object
# that has two functions, getLine() and getBytes(nbytes), which return
# strings from the input.  getLine() is only used in ascii mode and
# getBytes is only used in binary mode, so it's not strictly necessary
# for the inputsource object to provide both functions.

# The MenuParser switches between modes by creating new objects of the
# MenuParserMode class.  MenuParserMode subclasses must provide
# functions getMenuItem and getArguments.  getMenuItem takes an
# OOFMenu as an argument, reads some input (using inputsource) and
# returns an OOFMenuItem.  It should return None if there's no menu
# item to be read.  getArguments takes an OOFMenuItem argument and
# returns both a list of non-keyword arguments and a dictionary of
# keyword arguments.

from ooflib.SWIG.common import ooferror
from ooflib.common import debug
from ooflib.common import utils
import os
import sys

class MenuParser:
    mode_binary = 0
    mode_ascii = 1
    def __init__(self, inputsource, menu, mode=mode_ascii):
        self.inputsource = inputsource
        self.menu = menu
        if mode is MenuParser.mode_ascii:
            self.asciiMode()
        else:
            self.binaryMode()
    def getLine(self):
        return self.inputsource.getLine()
    def getBytes(self, n):
        return self.inputsource.getBytes(n)
    def asciiMode(self):
        self.mode = AsciiMenuParser(self)
    def binaryMode(self):
        from ooflib.common.IO import binarydata
        self.mode = binarydata.BinaryMenuParser(self)
    def run1(self):
        menuitem = self.mode.getMenuItem(self.menu)
        if menuitem is None:
            return 0
        args, kwargs = self.mode.getArguments(menuitem)
        if args:
            raise ooferror.PyErrDataFileError(
                "All arguments to menu commands must be keyword arguments!")
        menuitem.parser = self
        menuitem(**kwargs)
        menuitem.parser = None
        return 1
    def run(self):
        self.menu.root().quietmode(True) # don't echo commands in debug mode
        self.menu.root().haltLog()
        try:
            while self.run1():
                pass
        finally:
            self.menu.root().resumeLog()
            self.menu.root().quietmode(False)


#######################

class InputSource:
    def getLine(self):
        pass
    def getBytes(self, n):
        pass

## TODO PYTHON3? There maybe should be separate FileInput classes for
## ascii and binary files.  The binary ones should be opened in 'rb'
## mode and the ascii ones in 'r' mode.  The binary FileInput will
## have getBytes() and the ascii one will have getLine().  Maybe.  One
## problem with the current setup is that the lines retrieved by
## file.readline() are bytes objects, not strings, if the file was
## opened with the 'b' option.  When they're echoed to the screen in
## debug mode they look ugly, with an extra 'b' and quotation marks.
## The trouble with doing this is that we don't know whether a file
## should be binary or not until after the first line is read.

class FileInput(InputSource):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'rb')
        self.bytecount = 0
        self.totalbytes = os.stat(filename).st_size
    def getLine(self):
        line = self.file.readline()
        self.bytecount += len(line)
        if debug.debug():
            displaylen = 80     # chars to display per line
            if isinstance(line, bytes):
                dline = line.decode() # for display in debug mode
            if len(dline) <= displaylen:
                shortline = dline
            else:
                taillen = 5     # chars to show at end of line
                dots = "..."
                j = displaylen - taillen - len(dots)
                shortline = dline[0:j] + dots + dline[-taillen-1:-1]
            debug.msg(f"{self.filename}: {shortline}")
        return line
    def getBytes(self, n):
        b = self.file.read(n)
        self.bytecount += len(b)
        if len(b) != n:
            raise ooferror.PyErrDataFileError(
                "Premature EOF at byte %d! (%d missing)" %
                (self.bytecount, n-len(b)))
        return b
    def close(self):
        self.file.close()

class ProgFileInput(FileInput):
    ## FileInput with a ProgressBar
    def __init__(self, filename, progress):
        self.progress = progress
        self._error = False
        FileInput.__init__(self, filename)
    def reportProgress(self):
        self.progress.setFraction((1.0*self.bytecount)/self.totalbytes)
        self.progress.setMessage("Read %d/%d bytes" %
                                 (self.bytecount, self.totalbytes))
    def getLine(self):
        if self.progress.stopped():
            self._error = True
            raise ooferror.PyErrDataFileError("Interrupted!")
        line = FileInput.getLine(self)
        self.reportProgress()
        return line
    def getBytes(self, n):
        if self.progress.stopped():
            self._error = True
            raise ooferror.PyErrDataFileError("Interrupted!")
        b = FileInput.getBytes(self, n)
        self.reportProgress()
        return b
    def error(self):
        return self._error

class StringInput(InputSource):
    def __init__(self, strng):
        self.string = strng
        self.position = 0
    def getLine(self):
        result = self.string
        self.string = ""
        return result
    def getBytes(self, n):
        end = self.position + n
        if end > len(self.string):
            end = len(self.string)
        result = self.string[self.position:end]
        self.position = end
        return result

#######################

class MenuParserMode:
    # The derived classes must provide the following functions:
    def __init__(self, masterparser):
        pass
    def getMenuItem(self, menu):
        raise PyErrUserError(
            f"Somebody forgot to define {self.__class__.__name__}.getMenuItem()"
        )
    def getArguments(self, menuitem):
        # Returns a tuple containing the name of the argument and its
        # value.  It doesn't return the *string* containing the value,
        # because that wouldn't work for a BinaryFileMenuParser.
        # Returns None if there are no more arguments.
        raise "Somebody forgot to define %s.getArguments()" \
              % self.__class__.__name__


###########################
###########################
###########################

CMDSEP = "."                            # separates command and subcommand
ASSIGN = "="                            # assigns argument values
ARGSEP = ","                            # separates arguments
BGNARG = "("                            # begins arguments
ENDARG = ")"                            # ends arguments
SQUOTE = "'"                            # single quote
DQUOTE = '"'                            # double quote
COMMENT = "#"
ESCAPE = "\\"                           # continuation at EOL, quote special
BGNLIST = '['
ENDLIST = ']'
BGNTUPLE = '('
ENDTUPLE = ')'
BGNINDEX = '['
ENDINDEX = ']'

def legalname(name):
    a = name[0]
    if not (a.isalpha() or a == "_"):
        return False
    for c in name[1:]:
        if not (c.isalnum() or c == "_"):
            return False
    return True

def string2number(strng):
    try:
        return int(strng)
    except ValueError:
        return float(strng)


class AsciiMenuParser(MenuParserMode):
    # The parser does *NOT* understand backslashes correctly, but
    # since it's supposed to be used to read data files, not general
    # python files, that's not a big deal.  Backslashes are only
    # understood in the context of quoted strings, for escaping
    # internal quotation marks.

    # The parser is always in one of these states:
    state_idle = 0                      # none of the below
    state_cmd = 1                       # processing menu items
    state_arg = 2                       # looking for argument name=value

    def __init__(self, masterparser):
        # ascii mode stuff
        self.masterparser = masterparser
        self.buffer = ""
        self.bufpos = 0                 # position in buffer
        self.buflen = 0
        self.parendepth = 0
        self.state = AsciiMenuParser.state_idle
        self.storedTokens = []

    def fetchLine(self):
        self.buffer = self.masterparser.getLine().decode("UTF-8")
        self.bufpos = 0
        self.buflen = len(self.buffer)

    def nextToken(self):
        # Retrieve the next unit of information ('token') from the input.
        if self.storedTokens:
            return self.storedTokens.pop()
        return self._nextToken()

    def pushbackToken(self, token):
        # Restore a unit of information to the input.  It will be
        # retrieved on the next call to nextToken().  An arbitrary
        # number of tokens can be pushed back.
        self.storedTokens.append(token)

    def skipSpace(self):
        while self.bufpos < self.buflen and self.buffer[self.bufpos].isspace():
            self.bufpos += 1

    def clearBuffer(self):
        self.buffer = ""
        self.buflen = 0
        self.bufpos = 0

    def _nextToken(self):
        # Do the actual work of retrieving information from the input.
        # The token is removed from self.buffer and returned.

        # Make sure the buffer has something in it.  Get more input if needed.
        while self.bufpos == self.buflen:
            self.fetchLine()
            if not self.buffer:           # no more input
                return None
            self.buflen = len(self.buffer)
            self.bufpos = 0
            self.skipSpace()            # adjusts bufpos

        self.skipSpace()
        if self.bufpos == self.buflen:
            return self._nextToken()
        
        # Discard comments.
        if self.buffer[self.bufpos] == COMMENT:
            self.clearBuffer()
            return self._nextToken()

        # Special characters are tokens all by themselves, unless
        # they're quotation marks or group delimiters, in which case
        # the whole quoted string or group is a token.
        c = self.buffer[self.bufpos]
        if c in specialchars[self.state]:
            if c in quotechars:
                return self.processQuote()
            self.bufpos += 1
            return c

        # current char is not a special character.  Token is all chars to
        # next special character.
        end = self.bufpos + 1
        while end < self.buflen and \
                  not self.buffer[end] in specialchars[self.state]:
            end += 1
        token = self.buffer[self.bufpos:end] # don't include special char
        self.bufpos = end
        return token.rstrip()

    def processQuote(self):
        quotechar = self.buffer[self.bufpos]
        quote = ""
        while True:
            # look for closing quote
            end = self.bufpos + 1
            while end < self.buflen and self.buffer[end] != quotechar:
                end += 1
            if end == self.buflen:      # keep looking!
                quote += self.buffer[self.bufpos:end]
                self.fetchLine() # look at next line
                if not self.buffer:
                    raise ooferror.PyErrDataFileError(
                        "unmatched quotation marks")
            else:                       # found closing quote
                quote += self.buffer[self.bufpos:end+1]
                self.bufpos = end + 1
                if quote[-2] != ESCAPE:
                    return quote
                else:
                    quote = quote[:-2] + quote[-1] # remove ESCAPE
                # keep looking for more input

    def getMenuItem(self, menu):
        ident = self.getIdentifier()
        if ident is None:
            return None
        menuitem = getattr(menu, ident)
        return self.getSubMenuItem(menuitem)

    def getSubMenuItem(self, menu):
        ident = self.getIdentifier()
        if ident is None:
            return menu
        menuitem = getattr(menu, ident)
        return self.getSubMenuItem(menuitem)

    def getIdentifier(self):
        token = self.nextToken()
        if not token:
            return None                 # EOF
        if self.state is AsciiMenuParser.state_idle:
            if not legalname(token):
                raise ooferror.PyErrDataFileError(f"Illegal command: '{token}'")
            self.state = AsciiMenuParser.state_cmd
            return token
        if self.state is AsciiMenuParser.state_cmd:
            if token[0] == CMDSEP:
                self.state = AsciiMenuParser.state_idle
                return self.getIdentifier()
            if token[0] == BGNARG:
                self.parendepth += 1
                self.state = AsciiMenuParser.state_arg
                return None

    def getArguments(self, menuitem):
        # Returns list of args and dictionary of kwargs
        args = []
        kwargs = {}
        if self.state is not AsciiMenuParser.state_arg:
            return args, kwargs
        while True:
            token0 = self.nextToken()
            if token0 is None:
                raise ooferror.PyErrDataFileError("Premature EOF in data file?")
            if token0 in endSequence:
                # Does no checking for matching () or [] pairs!
                self.parendepth -= 1
                if self.parendepth == 0:
                    self.state = AsciiMenuParser.state_idle
                return args, kwargs
            if token0 == ARGSEP:
                continue
            token1 = self.nextToken()
            if token1 != ASSIGN:        # not a keyword argument
                self.pushbackToken(token1) # to be read again
                args.append(self.getArgumentValue(token0))
            else:                       # key word argument
                if not legalname(token0):
                    raise ooferror.PyErrDataFileError(
                        f"Illegal argument name: '{token0}'")
                token2 = self.nextToken()
                kwargs[token0] = self.getArgumentValue(token2)
            
    def getArgumentValue(self, token):
        if token[0] in quotechars:      # it's a string
            return token[1:-1]          # strip the quotation marks

        if token == BGNLIST:
            self.parendepth += 1
            return list(self.getArguments(None)[0])
        if token == BGNTUPLE:
            self.parendepth += 1
            return tuple(self.getArguments(None)[0])

        try:                            # is it a number?
            val = string2number(token)
        except ValueError:              # no, it's not
            pass
        else:
            return val                  # yes, it's a number

        # Is it None?
        if token == 'None':
            return None
        if token == 'True':
            return True
        if token == 'False':
            return False
        
        # Is it a function or variable defined in the OOF namespace?
        try:
            argval = utils.OOFeval_r(token)
        except KeyError:
            raise ooferror.PyErrDataFileError(
                f"Incomprehensible argument: {token}")

        # If it's a function, the next token is an open paren.
        nexttoken = self.nextToken()
        if nexttoken == BGNARG:
            self.parendepth += 1
            args, kwargs = self.getArguments(argval)
            return argval(*args, **kwargs)

        if nexttoken == BGNINDEX:
            self.parendepth += 1
            args, kwargs = self.getArguments(argval)
            return argval[args[0]]

        self.pushbackToken(nexttoken)   # to be read again
        return argval                   # arg was an OOF namespace variable

###################
    
# The parser's state affects the set of characters which have special
# meaning to it.  In particular, "." is the command separator in
# command and idle modes, but is a decimal point in argument mode.
# (Perhaps there should be a special number mode.)  The sets of
# special characters are stored in a dictionary keyed by the parser
# state.

specialchars = {}

specialchars[AsciiMenuParser.state_cmd] = (
    CMDSEP, BGNARG, ENDARG, SQUOTE, DQUOTE, COMMENT)
#  Are quotes special in state_cmd?  They should never be encountered.

specialchars[AsciiMenuParser.state_idle] = \
                                specialchars[AsciiMenuParser.state_cmd]

specialchars[AsciiMenuParser.state_arg] = (
    ASSIGN, ARGSEP, BGNARG, ENDARG, SQUOTE, DQUOTE, COMMENT,
    BGNLIST, ENDLIST, BGNTUPLE, ENDTUPLE, BGNINDEX, ENDINDEX)

quotechars = (SQUOTE, DQUOTE)
endSequence = (ENDLIST, ENDTUPLE, ENDINDEX)


                
