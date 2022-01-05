# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# This file looks for all subdirectories of the current directory and
# runs the gui test contained in each one.  The tests are run in
# alphabetical order of the subdirectory name.  It is assumed that
# each subdirectory contains a file named TESTFILE (defined below to
# "test.log", nee "log.py").  The test is run by
# executing
#         oof2 --pathdir <subdirectory> --replay <subdirectory>/TESTFILE
# and testing its return value. The subdirectory is added to the
# python path so that the log file can contain import statements that
# load tests from other files in the subdirectory.

# Actually, the tests are run in a temporary directory so that any
# files created by the test don't overwrite anything.  This means that
# *this* directory (OOF2/TEST/GUI) is also added to the path.  The
# temp directory also contains a symbolic link to OOF2/TEST/UTILS,
# which is added to the path. It contains links to
# OOF2/TEST/GUI/examples and OOF2/TEST/GUI/TEST_DATA, but they're not
# in the path.

# To temporarily skip a subdirectory, add a file called SKIP to it.

# The subdirectory can contain a file called "args" which contains a
# single line of arguments to be added to the oof2 command.  It can
# also contain a file named 'cleanup.py' which will be run after the
# test, if the test is successful.  cleanup.py is run in the
# guitests.py environment.

# Any test that calls sys.exit() with an non-zero status is considered
# a failure.  If a test is *supposed* to return a non-zero status,
# that status should be put in a file called 'exitstatus' in the
# test subdirectory.

TESTFILE = "test.log"

import getopt
import os
import string
import subprocess
import sys
import tempfile

delaystr = None
debug = False
no_checkpoints = False
sync = False
unthreaded = False
#forever = False

global tmpdir
tmpdir = None

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

def run_tests(dirs, rerecord, forever):
    homedir = os.getcwd()
    global tmpdir
    tmpdir = tempfile.mkdtemp(prefix='oof2temp_')
    print >> sys.stderr, "Using temp dir", tmpdir

    linkfile(homedir, 'examples')
    linkfile(homedir, 'TEST_DATA')
    os.symlink(os.path.abspath(os.path.join('..', 'UTILS')),
               os.path.join(tmpdir, 'UTILS'))
    os.chdir(tmpdir)
    try:
        if forever:
            counter = 0
            while 1:
                print >> sys.stderr, "******* %d ********" % counter
                counter += 1
                ok = really_run_tests(homedir, dirs, rerecord)
        else:
            really_run_tests(homedir, dirs, rerecord)
    except:
        print >> sys.stderr, "Not removing temp directory", tmpdir
        raise
    else:   # Successful execution. 
        pass
        # Remove everything from the temp directory.  There may be
        # *.pyc files left, as well as the subdirectories linked above,
        # even if everything ran correctly.
        for f in os.listdir(tmpdir):
            os.remove(f)
        # Remove the temp directory
        os.rmdir(tmpdir)

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

def really_run_tests(homedir, dirs, rerecord):
    nskipped = 0
    nrun = 0
    for directory in dirs:
        originaldir = os.path.join(homedir, directory)
        # Check that the directory and log file exist, and that
        # there's no SKIP file, before bothering to make the symlink
        # to the directory.
        if not os.path.isdir(originaldir):
            print >> sys.stderr, "Can't find directory", directory
            return
        if os.path.exists(os.path.join(originaldir, 'SKIP')) and len(dirs) > 1:
            print >> sys.stderr, " **** Skipping", directory, "****"
            nskipped += 1
            continue
        if not os.path.exists(os.path.join(originaldir, TESTFILE)):
            print >> sys.stderr, " **** Skipping", directory, "(No log file!) ****"
            nskipped += 1
            continue

        # Ok, everything's there.  Get ready to run this test.  Make a
        # symlink to the test directory.
        ## TODO: Is the symlink really necessary? We could provide a
        ## full path to TESTFILE.  The test directory is already in
        ## PYTHONPATH.
        testdir = os.path.join(tmpdir, directory)
        os.symlink(os.path.join(homedir, directory), testdir)

        # Read extra oof2 args from the args file, if it exists.
        if os.path.exists(os.path.join(directory, 'args')):
            argfile = open(os.path.join(directory, 'args'))
            extraargs = argfile.readline().rstrip().split()
            argfile.close()
        else:
            extraargs = []
        # Read the expected exit status from the exitstatus file, if
        # it exists.
        if os.path.exists(os.path.join(originaldir, 'exitstatus')):
            exitstatfile = open(os.path.join(directory, 'exitstatus'))
            exitstatus = int(exitstatfile.readline())
            exitstatfile.close()
        else:
            exitstatus = 0
        
        global delaystr
        if delaystr:
            extraargs += ["--replaydelay=", delaystr]
        if debug:
            extraargs += ["--debug"]
        if no_checkpoints:
            extraargs += ["--no-checkpoints"]
        if sync:
            extraargs += ["--gtk=", "--sync"]
        if unthreaded:
            extraargs += ["--unthreaded"]
        if rerecord:
            replayarg = 'rerecord'
        else:
            replayarg = 'replay'

        cmd = ["oof2",
               "--no-rc",       # .oof2rc might affect tests.  Don't use it.
               "--pathdir", ".",
               "--pathdir", "%s" % directory,
               "--pathdir", "%s" % homedir,
               "--pathdir", "UTILS",
               "--%s" % replayarg,
               os.path.join(directory, TESTFILE)] + extraargs

        print >> sys.stderr, "-------------------------"
        print >> sys.stderr, "--- Running %s" % ' '.join(cmd)
        os.putenv('OOFTESTDIR', directory)
        result = subprocess.call(cmd)
        print >> sys.stderr, "--- Return value =", result
        if result < 0:
            print >> sys.stderr, "Child was terminated by signal", -result
            print >> sys.stderr, "Test", directory, "failed!"
            sys.exit(result)

        if result != exitstatus:
            print "Test %s failed! Status=%d, expected=%d" \
                % (directory, result, exitstatus)
            sys.exit(result)
        print >> sys.stderr, "--- Finished %s" % directory

        cleanupscript = os.path.join(directory, 'cleanup.py')
        if os.path.exists(cleanupscript):
            execfile(cleanupscript)

        os.remove(testdir)
        nrun += 1
          
    print >> sys.stderr, "%d test%s ran successfully!" % (nrun, "s"*(nrun!=1))
    print >> sys.stderr, "Skipped %d test%s." % (nskipped, "s"*(nskipped!=1))

excluded = ['CVS','TEST_DATA', 'examples']

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

def get_dirs():
    files = [f for f in os.listdir('.')
             if os.path.isdir(f) and f not in excluded]
    files.sort()
    return files

def checkdir(directory, dirs):
    if directory not in dirs:
        print >> sys.stderr, "There is no directory named", directory
        sys.exit(1)

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

def removefile(filename):
    fullname = os.path.normpath(os.path.join(tmpdir, filename))
    print >> sys.stderr, "Removing file", fullname
    if os.path.exists(fullname):
        os.remove(fullname)

def linkfile(homedir, filename):
    os.symlink(os.path.join(homedir, filename),
               os.path.join(tmpdir, filename))

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

def printhelp():
    print >> sys.stderr, \
"""
Usage:  python guitests.py [options] [test directories]

Options are:
   --list       List test names in order, but don't run any of them.
   --from=dir   Start tests at directory dir.
   --after=dir  Start tests at the first one following dir.
   --to=dir     End tests at directory dir.
   --delay=ms   Specify delay (in milliseconds) between lines of each test.
   --debug      Run tests in debug mode.
   --unthreaded Run tests in unthreaded mode.
   --sync       Run tests in X11 sync mode (very slow over a network!).
   --rerecord   Re-record log files, and ignore 'assert' statements in them.
                This is useful if new checkpoints have been added.
   --no-checkpoints Ignore checkpoints in log files (not very useful).
   --forever    Repeat tests until they fail.
   --help       Print this message.
"""

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

def run(homedir):
    os.chdir(homedir)
    try:
        optlist, args = getopt.getopt(sys.argv[1:], '', 
                                      ['delay=', 'debug',
                                       'list',
                                       'from=', 'after=', 'to=',
                                       'rerecord', 'no-checkpoints',
                                       'sync', 'unthreaded',
                                       'forever', 'help'])
    except getopt.error, message:
        print message
        sys.exit(1)
    fromdir = None
    afterdir = None
    todir = None
    rerecord = False
    forever = False
    listtests = False
    for opt in optlist:
        if opt[0] == "--debug":
            debug = True
        elif opt[0] == "--delay":
            delaystr = opt[1]
        elif opt[0] == '--from':
            # normpath is necessary here because if the shell's
            # filename completion was used to construct the argument,
            # the directory name may have a trailing slash, and the
            # index() calls below will fail.
            fromdir = os.path.normpath(opt[1])
        elif opt[0] == '--after':
            afterdir = os.path.normpath(opt[1])
        elif opt[0] == '--to':
            todir = os.path.normpath(opt[1])
        elif opt[0] == '--rerecord':
            rerecord = True
        elif opt[0] == '--no-checkpoints':
            no_checkpoints = True
        elif opt[0] == '--unthreaded':
            unthreaded = True
        elif opt[0] == '--sync':
            sync = True
        elif opt[0] == '--forever':
            forever = True
        elif opt[0] == '--list':
            listtests = True
        elif opt[0] == '--help':
            printhelp()
            sys.exit(0)

    if listtests:
        dirs = get_dirs()
        print "\n".join(dirs)
        sys.exit(0)

    if args:         # test directories were explicitly listed on command line
        run_tests([os.path.normpath(a) for a in args], rerecord, forever)
    else:
        dirs = get_dirs()
        if afterdir:
            if fromdir:
                print >> sys.stderr, "You cannot use both --from and --after!"
                sys.exit(0)
            # Start at the directory following afterdir
            fromdir = dirs[dirs.index(afterdir)+1]

        if fromdir and not todir:
            checkdir(fromdir, dirs)
            start = dirs.index(fromdir)
            run_tests(dirs[start:], rerecord, forever)
        elif todir and not fromdir:
            checkdir(todir, dirs)
            end = dirs.index(todir)
            run_tests(dirs[:end+1], rerecord, forever)
        elif todir and fromdir:
            checkdir(fromdir, dirs)
            checkdir(todir, dirs)
            start = dirs.index(fromdir)
            end = dirs.index(todir)
            run_tests(dirs[start:end+1], rerecord, forever)
        else:                           # use all test directories
            run_tests(dirs, rerecord, forever)
                         
#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

if __name__ == "__main__":
    homedir = os.path.realpath(sys.path[0])
    run(homedir)
    
