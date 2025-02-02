# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

## Routines for submitting functions to be run in a subthread.  A
## "subthread" is a thread that's not associated directly with a
## menuitem or worker, but may have been spawned by such a thread.
## When threads are enabled, a thread is launched executing the
## proposed function. If threads are not enabled, the function is
## simply called on the main thread.

from ooflib.SWIG.common import lock
from ooflib.SWIG.common import ooferror
from ooflib.SWIG.common import threadstate
from ooflib.common import debug
from ooflib.common import excepthook
from ooflib.common import thread_enable
from ooflib.common import ooferrorwrappers
import sys
import threading

class StopThread(Exception):
    def __init__ (self):
        Exception.__init__(self)
    

class MiniThread(threading.Thread):
    def __init__(self, function, args=(), kwargs={}):
        from ooflib.SWIG.common.switchboard import StackWrap
        self.function = StackWrap(function)
        self.args = args
        self.kwargs = kwargs
        self.immortal = False
        threading.Thread.__init__(self)
        self.threadstate = None

    def immortalize(self):
        self.immortal = True
        
    def run(self):
        from ooflib.common.IO import reporter
        miniThreadManager.add(self)
        try:
            try:
                self.threadstate = threadstate.ThreadState()
                hook = excepthook.assign_excepthook(excepthook.OOFexceptHook())
                self.function(*self.args, **self.kwargs)
                excepthook.remove_excepthook(hook)
            except StopThread:
                excepthook.remove_excepthook(hook)
                return
            except ooferrorwrappers.PyOOFError as exception:
                debug.fmsg("Caught a PyOOFError!", exception, type(exception))
                reporter.error(exception.cexcept)
                sys.excepthook(*sys.exc_info())
            except Exception as exception:
                debug.fmsg("Caught something else!")
                reporter.error(exception)
                sys.excepthook(*sys.exc_info())
        finally:
            miniThreadManager.remove(self)
            self.threadstate = None

    def stop_it(self):
        if not self.immortal:
            threadstate.cancelThread(self.threadstate)

def execute(function, args=(), kwargs={}):
    if thread_enable.query():
        littlethread = MiniThread(function, args, kwargs)
        littlethread.start()
    else:
        function(*args, **kwargs)

def execute_immortal(function, args=(), kwargs={}):
    if thread_enable.query():
        littlethread = MiniThread(function, args, kwargs)
        littlethread.immortalize()
        littlethread.start()
    else:
        function(*args, **kwargs)


## The purpose of the MiniThreadManager is to administer the running
## (mini) threads, so that when quitting time comes all minithreads
## are asked to stop properly, releasing their locks, and freeing the
## main thread. The main thread is left to execute other quitting
## tasks.

class MiniThreadManager:
    def __init__(self):
        self.listofminithreads =[]
        self.lock = lock.SLock()

    def add(self, minithread):
        self.lock.acquire()
        try:
            self.listofminithreads.append(minithread)
        finally:
            self.lock.release()


    def remove(self,minithread):
        self.lock.acquire()
        try:
            self.listofminithreads.remove(minithread)
        finally:
            self.lock.release()

    def stopAll(self):
        threadlist = []
        self.lock.acquire()
        try:
            threadlist = self.listofminithreads[:]
        finally:
            self.lock.release()
        for minithread in threadlist:
            minithread.stop_it()
            
    def waitForAllThreads(self):
        threadlist = []
        self.lock.acquire()
        try:
            threadlist = self.listofminithreads[:]
        finally:
            self.lock.release()
#         debug.fmsg("waiting for subthreads", 
#                    [ts.threadstate.id() for ts in threadlist 
#                     if not ts.immortal])
        for minithread in threadlist:
            if not minithread.immortal:
                minithread.join()

    # Return the calling thread's MiniThread object, if it has one, or
    # None.
    def getMiniThread(self):
        callers_ts = threadstate.findThreadState()
        if callers_ts:
            self.lock.acquire()
            try:
                for mini in self.listofminithreads:
                    if mini.threadstate == callers_ts:
                        return mini
            finally:
                self.lock.release()
        
    def quit(self):
        if len(self.listofminithreads) > 0:
            ## TODO: Why is this done differently than for Worker
            ## threads?  Here, stopAll calls pthread_cancel, which we
            ## thought doesn't always work.  And waiting for threads
            ## is a bad idea if this function is called on the main
            ## thread, since it will block the main thread.
            self.stopAll()
            self.waitForAllThreads()
            
miniThreadManager = MiniThreadManager()

