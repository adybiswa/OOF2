# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

## Checkpointing.  When recording, calling "checkpoint(<comment>)"
## increments the counter and writes "checkpoint <comment>" in the log
## file.  When replaying, a "checkpoint <comment>" line in the log
## file won't finish until the same checkpoint is reached by the
## executing code.  There is no constraint on the order in which
## checkpoints are reached when running in threaded mode.

from . import logutils
import sys

_checkpointdict = {}

try:
    import threading
    _checkpointlock = threading.Lock()
except ImportError:
    import dummy_threading
    _checkpointlock = dummy_threading.Lock()

def checkpoint(comment):
    _checkpointlock.acquire()
    try:
        if logutils.recording():        # recording
            print("checkpoint", comment, file=logutils.logfile())
            logutils.logfile().flush()
            if logutils.debugLevel() >= 2:
                print("////// checkpoint", comment, file=sys.stderr)
        if logutils.replaying():
            try:
                _checkpointdict[comment] += 1
            except KeyError:
                _checkpointdict[comment] = 1
    finally:
        _checkpointlock.release()

def reset_checkpoints():
    global _checkpointdict
    _checkpointlock.acquire()
    try:                                # is this being absurdly careful?
        _checkpointdict = {}
    finally:
        _checkpointlock.release()

def check_checkpoint(comment):
    # Called by CheckPointLine to see if its checkpoint has been reached.
    _checkpointlock.acquire()
    try:
        try:
            count = _checkpointdict[comment]
        except KeyError:
            return False
        if count > 1:
            _checkpointdict[comment] = count-1
        else:
            del _checkpointdict[comment]
        return True
    finally:
        _checkpointlock.release()


class CheckPointException(Exception):
    pass

def checkpoint_count(comment, n=0):          # for debugging
    # If checkpoints have been changed in a program, and you want to
    # retrofit a script to use the new checkpoints, you can call this
    # function from the script to make sure that the checkpoints have
    # been inserted into the script properly.  It raises an exception
    # unless there have been n more calls to checkpoint() from the
    # code than there have been 'checkpoint' lines in the script.
    _checkpointlock.acquire()
    try:
        count = _checkpointdict.get(comment, 0)
        print("****** checkpoint_count", comment, count)
        if count != n:
            raise CheckPointException(comment)
    finally:
        _checkpointlock.release()
