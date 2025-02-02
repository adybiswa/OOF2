# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from generics import *

def activeAreaStatusCheck(nactive, ntotal, override=False):
    if override:
        return gtkTextviewCompare(
            'OOF2:Active Area Page:Pane:Status',
            "OVERRIDE: all %d pixels are active" % ntotal)
    return gtkTextviewCompare(
        'OOF2:Active Area Page:Pane:Status',
        '%d of %d pixels are active (%g%%)' % (nactive, ntotal,
                                                (100.*nactive)/ntotal))

def pixelSelectionCheck(n, ms='small.ppm'):
    return pixelSelectionSizeCheck(ms, n)

def activeAreaMSCheck(msname, n):
    from ooflib.common.IO import whoville
    ms = whoville.getClass('Microstructure')[msname].getObject()
    npix = ms.size()[0]*ms.size()[1]
    return npix - ms.activearea.size()  # aa.size() is no. of inactive pixels

def activeAreaCheck(n):
    return activeAreaMSCheck('small.ppm', n)

def activeAreaOverrideCheck(o):
    from ooflib.common.IO import whoville
    ms = whoville.getClass('Microstructure')['small.ppm'].getObject()
    return ms.activearea.getOverride() == o

def activeAreaPageSensitivityCheck0():
    return (sensitizationCheck({'OOF2:Active Area Page:Microstructure' : 1})
            and
            sensitizationCheck({"Store" : 1,
                                "Rename" : 0,
                                "Delete" : 0,
                                "Restore" : 0,
                                "Modify:Method" : 1,
                                "Modify:Prev" : 0,
                                "Modify:OK" : 1,
                                "Modify:Next" : 0,
                                "Modify:Undo" : 0,
                                "Modify:Redo" : 0,
                                "Modify:Override" : 1
                                },
                               base="OOF2:Active Area Page:Pane"))

def activeAreaPageSensitivityCheck1():
    return (sensitizationCheck({'OOF2:Active Area Page:Microstructure' : 1})
            and
            sensitizationCheck({"Store" : 1,
                                "Rename" : 1,
                                "Delete" : 1,
                                "Restore" : 1,
                                "Modify:Method" : 1,
                                "Modify:Prev" : 0,
                                "Modify:OK" : 1,
                                "Modify:Next" : 0,
                                "Modify:Undo" : 1,
                                "Modify:Redo" : 0,
                                "Modify:Override" : 1
                                },
                               base="OOF2:Active Area Page:Pane"))
            

