# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modifed
# versions of this software, you first contact the authors at
# oof_manager@nist.gov.

target_sources(oof2common
  PRIVATE
  bitoverlay.C
  bitoverlay.h)

oof_swig_sources(
  SWIGFILES
  bitoverlay
  LIBRARIES
  oof2common)

add_subdirectory(GUI)
