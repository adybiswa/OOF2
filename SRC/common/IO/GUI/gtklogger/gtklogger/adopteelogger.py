# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from . import loggers

class AdopteeLogger(loggers.GtkLogger):
    #  Handles non-Widgets that were "adopted" by adoptGObject().
    def location(self, obj, *args):
        parent = getattr(obj, 'oofparent')
        parentcode = loggers.findLogger(parent).location(parent)
        strargs = [repr(x) for x in obj.oofparent_access_args] + \
                  ["%s=%s" % (name, repr(val))
                   for name, val in obj.oofparent_access_kwargs.items()]
        if hasattr(obj, 'oofparent_access_method'):
            return '%s.%s(%s)' % (parentcode, obj.oofparent_access_method,
                                  ','.join(strargs))
        else: # must have oofparent_access_function instead
            return '%s(%s)' % (obj.oofparent_access_function,
                               ','.join([parentcode]+strargs)) 

