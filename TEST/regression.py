# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

# Top-level regression test file for the OOF application.  Knows about
# all the test suites in this directory, and what order to run them in
# in order to get a proper regression test.

import sys, os, getopt, copy, unittest

test_module_names = [
    "fundamental_test",
    "microstructure_test",
    "image_test",
    "pixel_test",
    "activearea_test",
    "microstructure_extra_test",
    "matrix_test",
    "matrix_method_test",
    "misorientation_test",
    "skeleton_basic_test",
    "skeleton_select_test",
    "skeleton_bdy_test",
    "skeleton_periodic_test",
    "skeleton_periodic_bdy_test",
    "skeleton_selectionmod_test",
    "skeleton_extra_test",
    "material_property_test",
    "pixel_extra_test",
    "mesh_test",
    "subproblem_test",
    "solver_test",
    "boundary_condition_test",
    "aniso_test",
    "nonlinear_linear_test",
    "nonlinear_floatbc_test",
    "nonconstant_property_test",
    "nonlinear_property_test",
    "nonlinear_plane_flux_test",
    "nonlinear_timedependent_tests",
    "nonlinear_K_timedep_tests",
    "amr_test",
    "output_test",
    "pyproperty_test",
    "scheduled_output_test",
    "time_dependent_bc_test",
    "subproblem_test_extra",
    "r3tensorrotationbug",
    "polefigure_test",
    "zstrain_test",
    # "interface_test"
    ]



# The startup sequence for regression.py has to imitate the executable
# oof2 script. That one imports the contents of the math module into
# the main oof namespace, so we have to do it here too.  Not importing
# math here will make some tests fail.
from math import *

def stripdotpy(name):
    if name.endswith(".py"):
        return name[:-3]
    return name

testcount = 1

def run_modules(test_module_names, oofglobals, backwards):
    logan = unittest.TextTestRunner()
    if backwards:
        test_module_names.reverse()
    for m in test_module_names:
        try:
            ldict = {}
            exec(f"from oof2.TEST import {m} as test_module", globals(), ldict)
            test_module = ldict["test_module"]
        except ImportError:
            print(f"Import error: {m}", file=sys.stderr)
            print(f"path is {sys.path}")
        else:
            print("Running test module %s." % m)
            # Make sure all the goodies in the OOF namespace are available.
            test_module.__dict__.update(oofglobals)
            if hasattr(test_module, "initialize"):
                test_module.initialize()
            for t in test_module.test_set:
                global testcount
                print("\n *** Running test %d: %s ***\n" % \
                    (testcount, t.id()), file=sys.stderr)
                testcount += 1
                res = logan.run(t)
                if not res.wasSuccessful():
                    return False
            # res = test_module.run_tests()
            # if res==0: # failure.
            #     return False
    return True

def printhelp():
    print(f"""
    Usage : {os.path.split(sys.argv[0])[1]} [options] [test names]

Options are:
   --list             List test names in order, but don't run any of them.
   --from   testname  Start with the given test.
   --after  testname  Start after the given test.
   --to     testname  Stop at the given test.
   --forever          Repeat tests until they fail.
   --backwards        Run tests in reverse order.
   --oofargs args     Pass arguments to oof2.
   --debug            Run oof2 in debug mode.
   --help             Print this message.
The options --from, --after, and --to cannot be used if test names are 
explicitly listed after the options.
""", file=sys.stderr)

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

def run(homedir):
    global test_module_names
    try:
        opts,args = getopt.getopt(sys.argv[1:],"f:a:t:o:",
                                  ["from=", "after=", "to=", "oofargs=",
                                   "forever", "debug", "backwards",
                                   "help", "list"])
    except getopt.GetoptError as err:
        print(str(err))
        printhelp()
        sys.exit(2)

    oofargs = []
    
    fromtogiven = False
    startaftergiven = False
    forever = False
    debug = False
    backwards = False

    for o,v in opts:
        if o in ("-f", "--from"):
            if startaftergiven:
                print("You can't use both --from and --after.", file=sys.stderr)
                sys.exit(1)
            v = stripdotpy(v)
            test_module_names = test_module_names[test_module_names.index(v):]
            fromtogiven = True
            startaftergiven = True
        if o in ("-a", "--after"):
            if startaftergiven:
                print("You can't use both --from and --after.", file=sys.stderr)
                sys.exit(1)
            v = stripdotpy(v)
            test_module_names = \
                test_module_names[test_module_names.index(v)+1:]
            fromtogiven = True
            startaftergiven = True
        elif o in ("-t", "--to"):
            v = stripdotpy(v)
            test_module_names = \
                test_module_names[:test_module_names.index(v)+1]
            fromtogiven = True
        elif o in ("-o","--oofargs"):
            oofargs = v.split()
        elif o == "--forever":
            forever = True
        elif o == "--debug":
            debug = True
        elif o == "--backwards":
            backwards = True
        elif o == "--list":
            print("\n".join(test_module_names))
            sys.exit(0)
        elif o == "--help":
            printhelp()
            sys.exit(0)

    if fromtogiven:
        if args:
            print("You can't explicitly list the tests *and* use --from, --after, or --to.")
            sys.exit(1)
    elif args:
        test_module_names = [stripdotpy(a) for a in args]
        

    # Effectively pass these through.
    sys.argv = [sys.argv[0]] + oofargs

    try:
        import oof2
        if oof2.__file__ not in sys.path:
            sys.path.append(os.path.dirname(oof2.__file__))
        from ooflib.common import oof
    except ImportError:
        print("OOF is not correctly installed on this system.")
        sys.exit(4)

    sys.argv.extend(["--text", "--quiet", "--seed=17"])
    if debug:
        sys.argv.append("--debug")

    # Set the time zone here so that pdf files generated in the tests
    # are in the same zone as the reference files.  The pdf files
    # created by Cairo include a creation time, which is ignored
    # during comparison, but they also include a file length, which
    # changes if the time format changes.  Setting TZ here means that
    # the creation time will always include time zone information, and
    # will always use the same number of characters.
    os.environ["TZ"] = "Etc/UTC"

    oof.run(no_interp=1)

    # Make a temp directory and cd to it, but put the current
    # directory in the path first, so imports will still work.  By
    # cd'ing to a temp directory, we ensure that all files written
    # during the tests won't clobber or be clobbered by files written
    # by another test being run in the same file system.
    import tempfile
    sys.path[0] = os.path.realpath(sys.path[0])
    tmpdir = tempfile.mkdtemp(prefix='oof2temp_')
    print("Using temp dir", tmpdir, file=sys.stderr)
    os.chdir(tmpdir)
    # Tell file_utils where the home directory is, since reference
    # files are named relative to it.  See comment in
    # fundamental_test.py about using the absolute path name here.
    from oof2.TEST.UTILS import file_utils
    file_utils.set_reference_dir(homedir)

    # utils.OOFglobals() returns OOF namespace objects that we will be
    # making available to each test script.  If test scripts modify
    # globals (eg, by using utils.OOFdefine or the scriptloader), we
    # don't want those modifications to affect later test scripts.
    # Therefore we create a pristine copy of globals now, and use it
    # instead of utils.OOFglobals() later.
    from ooflib.common import utils
    oofglobals = copy.copy(utils.OOFglobals())
    ok = False
    try:
        if forever:
            count = 0
            ok = False
            while run_modules(test_module_names, oofglobals, backwards):
                count += 1
                print("******* Finished", count, \
                    "iteration%s"%("s"*(count>1)), "*******", file=sys.stderr)
        else:
            ok = run_modules(test_module_names, oofglobals, backwards)
    finally:
        if ok:
            print("All tests completed successfully!", file=sys.stderr)
            if not debug:
                os.rmdir(tmpdir)
            else:
                print("Temp dir", tmpdir, "was not removed.", file=sys.stderr)
        else:
            print("Test failed. Temp dir", tmpdir, "was not removed.", file=sys.stderr)


#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

if __name__=="__main__":
    homedir = os.path.realpath(sys.path[0])
    run(homedir)
    OOF.File.Quit()
