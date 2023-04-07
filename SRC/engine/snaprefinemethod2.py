# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

from ooflib.SWIG.common import config
from ooflib.SWIG.common import switchboard
from ooflib.common import debug
from ooflib.common import utils
from ooflib.common import primitives
from ooflib.common import enum
from ooflib.common.IO import xmlmenudump

# TODO PYTHON3: rules should return multiple choices and SnapRefine
# can choose the best one, using ProvisionalElements.

Point = primitives.Point

class RefinementRuleSet:
    allRuleSets = []
    ruleDict = {}
    def __init__(self, name, parent=None, help=None):
        self.rules = {}
        self._name = name
        self._help = help
        RefinementRuleSet.allRuleSets.append(self)
        RefinementRuleSet.ruleDict[name] = self
        if parent:
            self.parent = RefinementRuleSet.ruleDict[parent]
        else:
            self.parent = None
        updateRuleEnum(name, help) 
    def name(self):
        return self._name
    def help(self):
        return self._help
    def __repr__(self):
        return "getRuleSet('%s')" % self.name()
    def addRule(self, rule, signature):
        self.rules[signature] = rule
    def getRule(self, signature):
        try:
            return self.rules[signature]
        except KeyError:
            if self.parent:
                return self.parent.getRule(signature)
            raise
    def __getitem__(self, signature):
        return self.getRule(signature)
        
def getRuleSet(name):
    return RefinementRuleSet.ruleDict[name]

utils.OOFdefine('getSnapRefineRuleSet', getRuleSet)

#########################################

# The names of the refinement rule sets are stored in an Enum so that
# the UI can handle them correctly.

class RuleSet(enum.EnumClass(*[(r.name(), r.help())
                               for r in RefinementRuleSet.allRuleSets])):
    pass
##    def __repr__(self):
##	return "RuleSet('%s')" % self.name
utils.OOFdefine('SnapRefineRuleSet', RuleSet)

RuleSet.tip = "Refinement rule sets."
RuleSet.discussion = xmlmenudump.loadFile('DISCUSSIONS/engine/enum/ruleset.xml')

def updateRuleEnum(name, help=None):
    enum.addEnumName(RuleSet, name, help)

# This function is used to get the default value for the RuleSet
# parameter in the refinement menu item.
def conservativeRuleSetEnum():
    return RuleSet(RefinementRuleSet.allRuleSets[0].name())

##########################################

class RefinementRule:
    def __init__(self, ruleset, signature, function):
        self.function = function
        ruleset.addRule(self, signature)
    def apply(self, element, rotation, edgenodes, newSkeleton, alpha):
        return self.function(element, rotation, edgenodes, newSkeleton, alpha)

##########################################

# This function is called with an old element and an integer
# 'rotation'.  It returns copies (children in the SkeletonSelectable
# sense) of the nodes of the old element, with the order shifted
# around by the rotation.  The new refined elements will be built from
# these base nodes.

def baseNodes(element, rotation):
    nnodes = element.nnodes()
    return [element.nodes[(i+rotation)%nnodes].children[-1]
            for i in range(nnodes)]

class ProvisionalRefinement:
   def __init__(self, newbies = [], internalNodes = []):
       self.newbies = newbies   # new elements
       self.internalNodes = internalNodes
   def energy(self, skeleton, alpha):
       energy = 0.0
       for element in self.newbies:
           energy += element.energyTotal(skeleton, alpha)
       return energy/len(self.newbies)
   def accept(self, skeleton):
       return [element.accept(skeleton) for element in self.newbies]

def theBetter(skeleton, candidates, alpha):
   energy_min = 100000.                # much larger than any possible energy
   theone = None
   for candi in candidates:
       energy = candi.energy(skeleton, alpha)
       if energy < energy_min:
           energy_min = energy
           theone = candi
   # Before returning the chosen refinement, we need to remove any internal
   # nodes created for the refinements that were not chosen.
   destroyedNodes = {}
   for candi in candidates:
       if candi is theone or not candi.internalNodes:
           continue
       for n in candi.internalNodes:
           if n not in theone.internalNodes and n not in destroyedNodes:
               n.destroy(skeleton)
               destroyedNodes[n] = 1
                
   return theone.accept(skeleton)

##########################################

# A missing function in the liberal rule set will be found in
# conservative ruleset. 
conservativeRuleSet = RefinementRuleSet(
    'conservative',
    help='Preserve topology: quads are refined into quads and triangles into triangles (whenever possible).')

liberalRuleSet = RefinementRuleSet(
    'liberal',
    parent='conservative',
    help="If there's a choice, choose the refinement that minimizes E, without trying to preserve topology.")

#########################################

# ruleZero applies to both triangles and quads which don't need refining.

def ruleZero(element, rotation, edgenodes, newSkeleton, alpha):
    return (newSkeleton.newElement(nodes=baseNodes(element, rotation),
                                   parents=[element]),)

# def unrefinedelement(element, signature_info, newSkeleton):
#     if config.dimension() == 2:
#         bNodes = baseNodes(element, 0)
#     elif config.dimension() == 3:
#         bNodes = baseNodes(element)
#     el = newSkeleton.newElement(nodes=bNodes,
#                                 parents=[element])
#     return (el,)

##########################################

# Unrefined triangle

RefinementRule(liberalRuleSet, (0,0,0), ruleZero)

#########

#          2
#         /|\
#        / | \
#       /  |  \
#      /   |   \
#     /    |    \
#    /     |     \
#   /______|______\
#   0      a      1
#

def rule100(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    return (newSkeleton.newElement(nodes=[na, n2, n0], parents=[element]),
            newSkeleton.newElement(nodes=[na, n1, n2], parents=[element]))
RefinementRule(liberalRuleSet, (1,0,0), rule100)

#          2                   2                   2       
#         /\                  /|\                  /\       
#        /  \                / | \                /  \      
#       /    \              /  |  \              /    \     
#      /      b            /   |   b            /    . b    
#     /      / \          /    | /  \          /  .   / \   
#    /      /   \        /     |/    \        /.     /   \  
#   /______/_____\      /______/______\      /______/_____\ 
#   0      a      1     0      a      1     0      a      1
#

def rule110(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[(rotation+1)%3][0]
    refine0 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, n1, n2], parents=[element]),
         ProvisionalTriangle(nodes=[na, nb, n2], parents=[element]),
         ProvisionalTriangle(nodes=[na, n1, nb], parents=[element])])
    refine1 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, nb, n2], parents=[element]),
         ProvisionalTriangle(nodes=[n0, na, nb], parents=[element]),
         ProvisionalTriangle(nodes=[na, n1, nb], parents=[element])])
    refine2 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[na, n1, nb], parents=[element]),
         ProvisionalQuad(nodes=[n0, na, nb, n2], parents=[element])])
    return theBetter(newSkeleton, (refine0, refine1, refine2), alpha)

RefinementRule(liberalRuleSet, (1,1,0), rule110)

    

#             2                            2
#             /\                           /\         
#            /  \                         /  \        
#           /    \                       /    \       
#          /      \                    c/      \b     
#       c /________\b                  / \    / \     
#        /\        /\                 /   \  /   \    
#       /  \      /  \               /     \/d    \   
#      /    \    /    \             /      |       \  
#     /      \  /      \           /       |        \ 
#    /________\/________\         /________|_________\
#    0        a          1        0        a         1

def rule111(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[(rotation+1)%3][0]
    nc = edgenodes[(rotation+2)%3][0]
    center = element.center()
    nd = newSkeleton.newNode(center.x, center.y)
    refine0 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, nc], parents=[element]),
         ProvisionalTriangle(nodes=[na, n1, nb], parents=[element]),
         ProvisionalTriangle(nodes=[nc, nb, n2], parents=[element]),
         ProvisionalTriangle(nodes=[na, nb, nc], parents=[element])])
    refine1 = ProvisionalRefinement(
        [ProvisionalQuad(nodes=[n0, na, nd, nc], parents=[element]),
         ProvisionalQuad(nodes=[na, n1, nb, nd], parents=[element]),
         ProvisionalQuad(nodes=[nd, nb, n2, nc], parents=[element])],
        internalNodes = [nd])
    return theBetter(newSkeleton, (refine0, refine1), alpha)

RefinementRule(liberalRuleSet, (1,1,1), rule111)


#             2
#             /\
#            /||\
#           / || \
#          / |  | \       There are a limited number of slopes
#         /  /  \  \      that are convenient in ascii art.
#        /  |    |  \     Pretend that the lines from a to 2 and b to 2
#       /   /    \   \    are straight.
#      /   |     |    \
#     /   /       \    \
#    /___/_________\____\
#   0    a         b    1

#          2
#         /|\
#        / | \
#       /  |  \
#      /  c|   \
#     /   /\    \
#    /   /  \    \ 
#   /___/____\____\     
#   0   a     b   1

def rule200(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    center = element.center()
    nc = newSkeleton.newNode(center.x, center.y)
    refine0 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, n2], parents=[element]),
         ProvisionalTriangle(nodes=[na, nb, n2], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, n2], parents=[element])])
    refine1 = ProvisionalRefinement(
        [ProvisionalQuad(nodes=[n0, na, nc, n2], parents=[element]),
         ProvisionalTriangle(nodes=[na, nb, nc], parents=[element]),
         ProvisionalQuad(nodes=[nb, n1, n2, nc], parents=[element])],
        internalNodes=[nc])
    return theBetter(newSkeleton, (refine0, refine1), alpha)

RefinementRule(liberalRuleSet, (2,0,0), rule200)



#           2
#           /\
#          /  \
#         /    \c
#        /    /|\
#       /    / | \      
#      /    /  |  \ 
#     /    /   |   \
#    /____/____|____\
#   0    a     b     1

def rule210(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nc, n2], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]))

RefinementRule(liberalRuleSet, (2,1,0), rule210)

#           2
#           /\
#          /  \
#        c/    \ 
#        /|\    \
#       / | \    \      
#      /  |  \    \ 
#     /   |   \    \
#    /____|____\____\
#   0     a     b    1

def rule201(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+2)%3][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nc], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, n2, nc], parents=[element]))

RefinementRule(liberalRuleSet, (2,0,1), rule201)


#            2                     2                     2             
#           / \                   / \                   / \            
#          /   \                 /   \                 /   \           
#        d/_____\c             d/     \c             d/     \c         
#        /|     |\             /|\    |\             /|    /|\         
#       / |     | \           / | \   | \           / |   / | \        
#      /  |     |  \         /  |  \  |  \         /  |  /  |  \       
#     /   |     |   \       /   |   \ |   \       /   | /   |   \      
#    /____|_____|____\     /____|____\|____\     /____|/____|____\     
#   0     a     b    1    0     a     b    1    0     a     b    1     

def rule211(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    nd = edgenodes[(rotation+2)%3][0]
    refine0 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, nd], parents=[element]),
         ProvisionalQuad(nodes=[na, nb, nc, nd], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element]),
         ProvisionalTriangle(nodes=[nd, nc, n2], parents=[element])])
    refine1 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, nd], parents=[element]),
         ProvisionalTriangle(nodes=[na, nb, nd], parents=[element]),
         ProvisionalQuad(nodes=[nb, nc, n2, nd], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element])])
    refine2 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, nd], parents=[element]),
         ProvisionalQuad(nodes=[na, nc, n2, nd], parents=[element]),
         ProvisionalTriangle(nodes=[na, nb, nc], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element])])
    return theBetter(newSkeleton, (refine0, refine1, refine2), alpha)

RefinementRule(liberalRuleSet, (2,1,1), rule211)

#           2
#           /\
#          /  \
#         /    \d
#        /    / \
#       /    /   \c     
#      /    /    /\ 
#     /    /    /  \
#    /____/____/____\
#   0     a    b     1

def rule220(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    nd = edgenodes[(rotation+1)%3][1]
    return (
        newSkeleton.newElement(nodes=[n0, na, nd, n2], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]))

RefinementRule(liberalRuleSet, (2,2,0), rule220)

#           2                     2                     2              
#           /\                    /\                    /\           
#          /  \                  /  \                  /  \          
#        e/____\d              e/____\d              e/    \d        
#        /|   / \              /    / \              /|   / \        
#       / |  /   \c           /    /   \c           / |  /   \c      
#      /  | /    /\          /    /    /\          /  | /    /\      
#     /   |/    /  \        /    /    /  \        /   |/    /  \     
#    /____/____/____\      /____/____/____\      /____/____/____\    
#   0     a    b     1    0     a    b     1    0     a    b     1    

def rule221(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    nd = edgenodes[(rotation+1)%3][1]
    ne = edgenodes[(rotation+2)%3][0]
    refine0 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, ne], parents=[element]),
         ProvisionalTriangle(nodes=[na, nd, ne], parents=[element]),
         ProvisionalTriangle(nodes=[ne, nd, n2], parents=[element]),
         ProvisionalQuad(nodes=[na, nb, nc, nd], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element])])
    refine1 = ProvisionalRefinement(
        [ProvisionalQuad(nodes=[n0, na, nd, ne], parents=[element]),
         ProvisionalQuad(nodes=[na, nb, nc, nd], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element]),
         ProvisionalTriangle(nodes=[ne, nd, n2], parents=[element])])
    refine2 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, ne], parents=[element]),
         ProvisionalQuad(nodes=[na, nd, n2, ne], parents=[element]),
         ProvisionalQuad(nodes=[na, nb, nc, nd], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element])])
    return theBetter(newSkeleton, (refine0, refine1, refine2), alpha)


RefinementRule(liberalRuleSet, (2,2,1), rule221)


#            2                      2         
#           /\                     /\         
#          /  \                   /  \        
#        e/____\d               e/    \d      
#        /\    /\               /\    /\      
#       /  \  /  \             /  \  /  \     
#     f/____\g____\c         f/____\g____\c   
#     /\    /\    /\         /     /\     \   
#    /  \  /  \  /  \       /     /  \     \  
#   /____\/____\/____\     /____ /____\ ____\ 
#  0      a     b     1   0      a     b     1
#
#            2                      2         
#           /\                     /\          
#          /  \                   /  \         
#        e/____\d               e/____\d       
#        /     /\               /      \     
#       /     /  \             /        \      
#     f/_____g    \c         f/__________\c    
#     /\     \    /\         /\          /\    
#    /  \     \  /  \       /  \        /  \   
#   /____\ ____\/____\     /____\ ____ /____\  
#  0      a     b     1   0      a     b     1 
#
# There are two orientations for the first diagram in the second row
# and three for the second.  It is possible to create other
# configurations by dividing some of the quadrilaterals into
# triangles.  I'm not sure how many configurations should be included.

def rule222(element, rotation, edgenodes, newSkeleton, alpha):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    nd = edgenodes[(rotation+1)%3][1]
    ne = edgenodes[(rotation+2)%3][0]
    nf = edgenodes[(rotation+2)%3][1]
    center = element.center()
    ng = newSkeleton.newNode(center.x, center.y)
    refine0 = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, nf], parents=[element]),
         ProvisionalTriangle(nodes=[na, ng, nf], parents=[element]),
         ProvisionalTriangle(nodes=[na, nb, ng], parents=[element]),
         ProvisionalTriangle(nodes=[nb, nc, ng], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element]),
         ProvisionalTriangle(nodes=[nf, ng, ne], parents=[element]),
         ProvisionalTriangle(nodes=[ng, nd, ne], parents=[element]),
         ProvisionalTriangle(nodes=[ng, nc, nd], parents=[element]),
         ProvisionalTriangle(nodes=[ne, nd, n2], parents=[element])],
        internalNodes=[ng])
    refine1 = ProvisionalRefinement(
        [ProvisionalQuad(nodes=[n0, na, ng, nf], parents=[element]),
         ProvisionalQuad(nodes=[n1, nc, ng, nb], parents=[element]),
         ProvisionalQuad(nodes=[n2, ne, ng, nd], parents=[element]),
         ProvisionalTriangle(nodes=[na, nb, ng], parents=[element]),
         ProvisionalTriangle(nodes=[nc, nd, ng], parents=[element]),
         ProvisionalTriangle(nodes=[ne, nf, ng], parents=[element])]
        internalNodes=[ng])
    refine2a = ProvisionalRefinement(
        [ProvisionalQuad(nodes=[na, nb, ng, nf], parents=[element]),
         ProvisionalQuad(nodes=[nb, nc, nd, ng], parents=[element]),
         ProvisionalQuad(nodes=[ne, nf, ng, nd], parents=[element]),
         ProvisionalTriangle(nodes=[n0, na, nf], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element]),
         ProvisionalTriangle(nodes=[ne, nd, n2], parents=[element])]
        internalNodes=[ng])
    refine2b = ProvisionalRefinement(
        [ProvisionalQuad(nodes=[na, nb, nc, ng], parents=[element]),
         ProvisionalQuad(nodes=[ng, nc, nd, ne], parents=[element]),
         ProvisionalQuad(nodes=[na, ng, ne, nf], parents=[element]),
         ProvisionalTriangle(nodes=[n0, na, nf], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element]),
         ProvisionalTriangle(nodes=[ne, nd, n2], parents=[element])]
        internalNodes=[ng])
    refine3a = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, nf], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element]),
         ProvisionalTriangle(nodes=[ne, nd, n2], parents=[element]),
         ProvisionalQuad(nodes=[na, nb, nc, nf], parents=[element]),
         ProvisionalQuad(nodes=[nf, nc, nd, ne], parents=[element])])
    refine3b = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, nf], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element]),
         ProvisionalTriangle(nodes=[ne, nd, n2], parents=[element]),
         ProvisionalQuad(nodes=[na, nb, nc, nd], parents=[element]),
         ProvisionalQuad(nodes=[na, nd, ne, nf], parents=[element])])
    refine3b = ProvisionalRefinement(
        [ProvisionalTriangle(nodes=[n0, na, nf], parents=[element]),
         ProvisionalTriangle(nodes=[nb, n1, nc], parents=[element]),
         ProvisionalTriangle(nodes=[ne, nd, n2], parents=[element]),
         ProvisionalQuad(nodes=[na, nb, ne, nf], parents=[element]),
         ProvisionalQuad(nodes=[nb, nc, nd, ne], parents=[element])])

    return theBetter(newSkeleton,
                     (refine0, refine1, refine2a, refine2b,
                      refine3a, refine3b, refine3c),
                     alpha)
    

#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#
#=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=##=--=#

# Quads
#
# Unrefined quadrilateral
    
RefinementRule(liberalRuleSet, (0,0,0,0), ruleZero)


#            D
#  3-------------------2
#  |\                 /|
#  | \               / |
#  |  \             /  |
#  |   \           /   |
# E|    \         /    |C
#  |     \       /     |
#  |      \     /      |
#  |       \   /       |
#  |        \ /        |
#  0---------a---------1
#       A         B

#  3-------------------2
#  |\ .                |
#  | \  .              |
#  |  \   .            |
#  |   \    .          |
#  |    \     .        |     TODO: WHY?
#  |     \      .      |
#  |      \       .    |
#  |       \        .  |
#  |        \         .|
#  0---------a---------1

#  3-------------------2
#  |\                  |
#  | \                 |
#  |  \                |
#  |   \               |
#  |    \              |
#  |     \             |
#  |      \            |
#  |       \           |
#  |        \          |
#  0---------a---------1

#  3-------------------2
#  |                . /|
#  |              .  / |
#  |            .   /  |
#  |          .    /   |
#  |        .     /    |    TODO: WHY?
#  |      .      /     |
#  |    .       /      |
#  |  .        /       |
#  |.         /        |
#  0---------a---------1

#  3-------------------2
#  |                  /|
#  |                 / |
#  |                /  |
#  |               /   |
#  |              /    |
#  |             /     |
#  |            /      |
#  |           /       |
#  |          /        |
#  0---------a---------1

def rule1000(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    rcats=[cats[(i+rotation)%4] for i in range(4)]
    A=rcats[0][0]
    B=rcats[0][1]
    C=rcats[1][0]
    D=rcats[2][0]
    E=rcats[3][0]
    na = edgenodes[rotation][0]
    if B!=C and C==D and D!=E and E==A:
        return (
            newSkeleton.newElement(nodes=[n0, na, n3], parents=[element]),
            newSkeleton.newElement(nodes=[na, n1, n3], parents=[element]),
            newSkeleton.newElement(nodes=[n1, n2, n3], parents=[element]))
    elif B==C and C==D and D!=E and E==A:
        return (
            newSkeleton.newElement(nodes=[n0, na, n3], parents=[element]),
            newSkeleton.newElement(nodes=[na, n1, n2, n3], parents=[element]))
    elif B==C and C!=D and D==E and E!=A:
        return (
            newSkeleton.newElement(nodes=[n3, n0, n2], parents=[element]),
            newSkeleton.newElement(nodes=[n0, na, n2], parents=[element]),
            newSkeleton.newElement(nodes=[na, n1, n2], parents=[element]))
    elif B==C and C!=D and D==E and E==A:
        return (
            newSkeleton.newElement(nodes=[na, n1, n2], parents=[element]),
            newSkeleton.newElement(nodes=[n3, n0, na, n2], parents=[element]))
    else:
        return (
            newSkeleton.newElement(nodes=[n3, n0, na], parents=[element]),
            newSkeleton.newElement(nodes=[n2, n3, na], parents=[element]),
            newSkeleton.newElement(nodes=[n1, n2, na], parents=[element]))
RefinementRule(liberalRuleSet, (1,0,0,0), rule1000)

#       E
#  3----------2
#  |          |
#  |          |D
#  |          |
# F|          |
#  |          b
#  |          |
#  |          |C
#  |          |
#  0----a-----1
#    A     B
#

#  3------------2
#  |\ .         |
#  | \  .       |
#  |  \   .     |
#  |   \    .   |    
#  |    \     . b      TODO: WHY?  Use a3 or b3, but not both
#  |     \     /|
#  |      \   / |
#  |       \ /  |
#  0--------a---1
#
# OR
#
#       E
#  3----------2
#  |         /|
#  |        / |D
#  |       /  |
# F|      /   |       
#  |     /   /b
#  |    /   / |
#  |   /   /  |C
#  |  /   /   |
#  | /   /    |
#  0----a-----1
#    A     B
#
# (It is straightforward to add more cases...)

def rule1100(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    rcats=[cats[(i+rotation)%4] for i in range(4)]
    A=rcats[0][0]
    B=rcats[0][1]
    C=rcats[1][0]
    D=rcats[1][1]
    E=rcats[2][0]
    F=rcats[3][0]
    na = edgenodes[rotation][0]
    nb = edgenodes[(rotation+1)%4][0]
    if D!=E and E==F and F!=A:
        return (
            newSkeleton.newElement(nodes=[na, n1, nb], parents=[element]),
            newSkeleton.newElement(nodes=[na, nb, n2, n0], parents=[element]),
            newSkeleton.newElement(nodes=[n0, n2, n3], parents=[element]))
    else:
        return (
            newSkeleton.newElement(nodes=[na, n1, nb], parents=[element]),
            newSkeleton.newElement(nodes=[nb, n2, n3], parents=[element]),
            newSkeleton.newElement(nodes=[na, nb, n3], parents=[element]),
            newSkeleton.newElement(nodes=[na, n3, n0], parents=[element]))
RefinementRule(liberalRuleSet, (1,1,0,0), rule1100)

#  3-----b-----2
#  |     |     |
#  |     |     |
#  |     |     |
#  |     |     |
#  |     |     |
#  |     |     |
#  |     |     |
#  |     |     |
#  0-----a-----1
#
# OR (various vays of splitting the two quads (x9))
#    E      D
#  3-----b-----2
#  |     |     |
#  |     |     |
# F|     |     |C
#  |     |     |
#  |     |     |
#  0-----a-----1
#    A      B
def rule1010(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    rcats=[cats[(i+rotation)%4] for i in range(4)]
    A=rcats[0][0]
    B=rcats[0][1]
    C=rcats[1][0]
    D=rcats[2][0]
    E=rcats[2][1]
    F=rcats[3][0]
    na = edgenodes[rotation][0]
    nb = edgenodes[(rotation+2)%4][0]
    newelements=()
    # Examine left quad
    if E==F and F!=A:
        newelements+=(
            newSkeleton.newElement(nodes=[n0, nb, n3], parents=[element]),
            newSkeleton.newElement(nodes=[n0, na, nb], parents=[element]))
    elif E!=F and F==A:
        newelements+=(
            newSkeleton.newElement(nodes=[n0, na, n3], parents=[element]),
            newSkeleton.newElement(nodes=[n3, na, nb], parents=[element]))
    else:
        newelements+=(
            newSkeleton.newElement(nodes=[n0, na, nb, n3], parents=[element]),)
    # Examine right quad
    if B==C and C!=D:
        newelements+=(
            newSkeleton.newElement(nodes=[na, n1, n2], parents=[element]),
            newSkeleton.newElement(nodes=[na, n2, nb], parents=[element]))
    elif B!=C and C==D:
        newelements+=(
            newSkeleton.newElement(nodes=[na, n1, nb], parents=[element]),
            newSkeleton.newElement(nodes=[n1, n2, nb], parents=[element]))
    else:
        newelements+=(
            newSkeleton.newElement(nodes=[na, n1, n2, nb], parents=[element]),)
    return newelements
RefinementRule(liberalRuleSet, (1,0,1,0), rule1010)

#    F     E
#  3----c-----2
#  |    |     |
#  |    |     |D
# G|    |     |
#  |    |_____|b 
#  |   /|d    |
#  |  / |     |
#  | /  |     |C
#  |/   |     |
#  0----a-----1
#    A     B
# OR
#
#  3----c-----2
#  |\   |     |
#  | \  |     |
#  |  \ |     |
#  |   \|_____|b 
#  |    |d    |
#  |    |     |
#  |    |     |
#  |    |     |
#  0----a-----1
#
# OR
#
#  3----c-----2
#  |\   |     |
#  | \  |     |
#  |  \ |     |
#  |   \|_____|b 
#  |   /|d    |
#  |  / |     |
#  | /  |     |
#  |/   |     |
#  0----a-----1
#

def rule1110(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    rcats=[cats[(i+rotation)%4] for i in range(4)]
    A=rcats[0][0]
    B=rcats[0][1]
    C=rcats[1][0]
    D=rcats[1][1]
    E=rcats[2][0]
    F=rcats[2][1]
    G=rcats[3][0]
    na = edgenodes[rotation][0]
    nb = edgenodes[(rotation+1)%4][0]
    nc = edgenodes[(rotation+2)%4][0]
    napt=na.position()
    ncpt=nc.position()
    nd = newSkeleton.newNode(0.5*(napt.x+ncpt.x), 0.5*(napt.y+ncpt.y))
    #Add the two right quads
    newelements=(
        newSkeleton.newElement(nodes=[na, n1, nb, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nd, nb, n2, nc], parents=[element]))
    #Choices for splitting the left quad
    if F==G and G!=A:
        newelements+=(
            newSkeleton.newElement(nodes=[n0, na, nd], parents=[element]),
            newSkeleton.newElement(nodes=[n0, nd, nc, n3], parents=[element]))
    elif F!=G and G==A:
        newelements+=(
            newSkeleton.newElement(nodes=[n3, nd, nc], parents=[element]),
            newSkeleton.newElement(nodes=[n3, n0, na, nd], parents=[element]))
    else:
        newelements+=(
            newSkeleton.newElement(nodes=[n0, na, nd], parents=[element]),
            newSkeleton.newElement(nodes=[nd, nc, n3], parents=[element]),
            newSkeleton.newElement(nodes=[n0, nd, n3], parents=[element]))
    return newelements
##    refine0 = ProvisionalRefinement(
##        newbies = [ProvisionalQuad(nodes=[na, n1, nb, nd], parents=[element]),
##                   ProvisionalQuad(nodes=[nd, nb, n2, nc], parents=[element]),
##                   ProvisionalQuad(nodes=[n0, nd, nc, n3], parents=[element]),
##                   ProvisionalTriangle(nodes=[n0, na, nd], parents=[element])],
##        internalNodes = [nd])
##    refine1 = ProvisionalRefinement(
##        newbies = [ProvisionalQuad(nodes=[na, n1, nb, nd], parents=[element]),
##                   ProvisionalQuad(nodes=[nd, nb, n2, nc], parents=[element]),
##                   ProvisionalQuad(nodes=[n0, na, nd, n3], parents=[element]),
##                   ProvisionalTriangle(nodes=[nd, nc, n3], parents=[element])],
##        internalNodes = [nd])    
##    refine2 = ProvisionalRefinement(
##        newbies = [ProvisionalQuad(nodes=[na, n1, nb, nd], parents=[element]),
##                   ProvisionalQuad(nodes=[nd, nb, n2, nc], parents=[element]),
##                   ProvisionalTriangle(nodes=[n0, na, nd], parents=[element]),
##                   ProvisionalTriangle(nodes=[n0, nd, n3], parents=[element]),
##                   ProvisionalTriangle(nodes=[nd, nc, n3], parents=[element])],
##        internalNodes = [nd])                      
##    return theBetter(newSkeleton, (refine0, refine1, refine2), alpha)
RefinementRule(liberalRuleSet, (1,1,1,0), rule1110)

#     F     E
#  3-----c-----2
#  |     |     |
# G|     |     |D
#  |     |     |
#  d_____e_____b
#  |     |     |
# H|     |     |C
#  |     |     |
#  |     |     |
#  0-----a-----1
#     A     B
#
# OR
#
#     F     E
#  3-----c-----2
#  |    / \    |
# G|   /   \   |D
#  |  /     \  |
#  d /       \ b
#  | \       / |
# H|  \     /  |C
#  |   \   /   |
#  |    \ /    |
#  0-----a-----1
#     A     B

def rule1111(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    rcats=[cats[(i+rotation)%4] for i in range(4)]
    A=rcats[0][0]
    B=rcats[0][1]
    C=rcats[1][0]
    D=rcats[1][1]
    E=rcats[2][0]
    F=rcats[2][1]
    G=rcats[3][0]
    H=rcats[3][1]
    #The rotation should not be necessary (also see refinemethod.py)
    na = edgenodes[rotation][0]
    nb = edgenodes[(rotation+1)%4][0]
    nc = edgenodes[(rotation+2)%4][0]
    nd = edgenodes[(rotation+3)%4][0]
    if (B==C and F==G and C==F) or (D==E and H==A and E==H):
        return (
            newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
            newSkeleton.newElement(nodes=[n0, na, nd], parents=[element]),
            newSkeleton.newElement(nodes=[na, n1, nb], parents=[element]),
            newSkeleton.newElement(nodes=[nb, n2, nc], parents=[element]),
            newSkeleton.newElement(nodes=[nd, nc, n3], parents=[element]))
    else:
        napt=na.position()
        ncpt=nc.position()
        ne = newSkeleton.newNode(0.5*(napt.x+ncpt.x), 0.5*(napt.y+ncpt.y))
        return (
            newSkeleton.newElement(nodes=[n0, na, ne, nd], parents=[element]),
            newSkeleton.newElement(nodes=[na, n1, nb, ne], parents=[element]),
            newSkeleton.newElement(nodes=[ne, nb, n2, nc], parents=[element]),
            newSkeleton.newElement(nodes=[nd, ne, nc, n3], parents=[element]))
RefinementRule(liberalRuleSet, (1,1,1,1), rule1111)


#################################################################################
# Cases with at least one edge being trisected.
# These don't use the edge categories (yet).

### Quads, cases 2XXX ###

#
#  3-----------------------2
#  |\                     /|
#  | \                   / |
#  |  \                 /  |
#  |   \               /   |
#  |    \             /    |
#  |     \           /     |
#  |      \         /      |
#  |       \       /       |
#  |        \     /        |
#  0---------a---b---------1
#

def rule2000(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    return (
        newSkeleton.newElement(nodes=[n0, na, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, n2, n3], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, n2], parents=[element]))
RefinementRule(liberalRuleSet, (2,0,0,0), rule2000)

#
#  3-----------------------2
#  |                    * /|
#  |                *    / |
#  |            *       /  |
#  |        *          /   |
#  |    *             /    |
#  c *               /     |
#  |   *            /      |
#  |     *         /       |
#  |       *      /        |
#  0---------a---b---------1
#

def rule2001(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+3)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nc], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, n2, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, n2], parents=[element]),
        newSkeleton.newElement(nodes=[nc, n2, n3], parents=[element]))
RefinementRule(liberalRuleSet, (2,0,0,1), rule2001)

#
#  3---------c-------------2
#  |         *             |
#  |                       |
#  |        * *            |
#  |                       |
#  |       *   *           |
#  |                       |
#  |      *     *          |
#  |                       |
#  |     *       *         |
#  0----a---------b--------1
#

def rule2010(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+2)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nc, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, n2, nc], parents=[element]))
RefinementRule(liberalRuleSet, (2,0,1,0), rule2010)

#
#  3---------c-------------2
#  |        **             |
#  |      *                |
#  |    *     *            |
#  |  *                    |
#  d*          *           |
#  |\                      |
#  | \          *          |
#  |  \                    |
#  |   \         *         |
#  0----a---------b--------1
#

def rule2011(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+2)%4][0]
    nd = edgenodes[(rotation+3)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nd], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, n2, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, nc, n3], parents=[element]))
RefinementRule(liberalRuleSet, (2,0,1,1), rule2011)

#
#  3-----------------------2
#  |\ *                    |
#  | \      *              |
#  |  \            *       |
#  |   \                 * c
#  |    \                 *|
#  |     \              *  |
#  |      \           *    |
#  |       \        *      |
#  |        \     *        |
#  0---------a---b---------1
#

def rule2100(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, n3], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nc, n2, n3], parents=[element]))
RefinementRule(liberalRuleSet, (2,1,0,0), rule2100)

#
#  3-----------------------2
#  |                       |
#  |                       |
#  |                       |
#  d-----------------------c
#  |*                     *|
#  |  *                 *  |
#  |    *             *    |
#  |      *         *      |
#  |        *     *        |
#  0---------a---b---------1
#

def rule2101(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+3)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nd], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nc, n2, n3, nd], parents=[element]))
RefinementRule(liberalRuleSet, (2,1,0,1), rule2101)

#
#  3--------------d--------2
#  |             / *       |
#  |            /    *     |
#  |           /       *   |
#  |          /          * |
#  |         /             c
#  |        /             /|
#  |       /             / |
#  |      /             /  |
#  |     /             /   |
#  0----a-------------b----1
#

def rule2110(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+2)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nd, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, nc, n2], parents=[element]))
RefinementRule(liberalRuleSet, (2,1,1,0), rule2110)

#
#  3----------d------------2
#  |       *     *         |
#  |    *            *     |
#  | *                  *  |
#  e-----------------------c
#  |*                     *|
#  |  *                 *  |
#  |    *             *    |
#  |      *         *      |
#  |        *     *        |
#  0---------a---b---------1
#

def rule2111(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+2)%4][0]
    ne = edgenodes[(rotation+3)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, ne], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, ne], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nc, n2, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nc, nd, ne], parents=[element]),
        newSkeleton.newElement(nodes=[n3, ne, nd], parents=[element]))
RefinementRule(liberalRuleSet, (2,1,1,1), rule2111)

### Quads, cases 22XX ###

#
#  3-----------------------2
#  |\ *                    |
#  | \     *               |
#  |  \        *           |
#  |   \            *      |
#  |    \                * d
#  |     \               * |
#  |      \           *    c
#  |       \       *     * |
#  |        \   *      *   |
#  0---------a--------b----1
#

def rule2200(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+1)%4][1]
    return (
        newSkeleton.newElement(nodes=[n0, na, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, n2, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nd, n3], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,0,0), rule2200)

#
#  3-----------------------2
#  |                       |
#  |                       |
#  |                       |
#  |                       |
#  e-----------------------d
#  |                     * |
#  |                  *    c
#  |               *     * |
#  |            *      *   |
#  0---------a--------b----1
#

def rule2201(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+1)%4][1]
    ne = edgenodes[(rotation+3)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nd, ne], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, n2, n3, ne], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,0,1), rule2201)

#
#  3---------e-------------2
#  |         |*            |
#  |         |   *         |
#  |         |      *      |
#  |         |         *   |
#  |         |            *d
#  |         |           * |
#  |         |        *    c
#  |         |     *     * |
#  |         |  *      *   |
#  0---------a--------b----1
#

def rule2210(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+1)%4][1]
    ne = edgenodes[(rotation+2)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, ne, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, n2, ne], parents=[element]),
        newSkeleton.newElement(nodes=[na, nd, ne], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,1,0), rule2210)

#
#  3---------e-------------2
#  |        *\*            |
#  |      *   \  *         |
#  |    *      \    *      |
#  |  *         \      *   |
#  f *           \        *d
#  | *            \        |
#  |   *           \       c
#  |     *          \    * |
#  |       *         \ *   |
#  0---------a--------b----1
#

def rule2211(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+1)%4][1]
    ne = edgenodes[(rotation+2)%4][0]
    nf = edgenodes[(rotation+3)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nf], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, ne, nf], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, n2, ne], parents=[element]),
        newSkeleton.newElement(nodes=[n3, nf, ne], parents=[element]),
        newSkeleton.newElement(nodes=[ne, nb, nc, nd], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,1,1), rule2211)

### Quads, cases 222X ###

#
#  3------f-------e--------2
#  |      |       |  *     |
#  |      |       |     *  |
#  |      |       |       *d
#  |      |       |        |
#  |      |       |        |
#  |      |       |       *c
#  |      |       |     *  |
#  |      |       |   *    |
#  |      |       | *      |
#  0------a-------b--------1
#

def rule2220(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+1)%4][1]
    ne = edgenodes[(rotation+2)%4][0]
    nf = edgenodes[(rotation+2)%4][1]
    return (
        newSkeleton.newElement(nodes=[n0, na, nf, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, ne, nf], parents=[element]),
        newSkeleton.newElement(nodes=[nb, nc, nd, ne], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, n2, ne], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,2,0), rule2220)

#
#  3------f-------e--------2
#  |     *|       |  *     |
#  |      |       |     *  |
#  |   *  |       |       *d
#  |      |       |        |
#  g *    |       |        |
#  |  *   |       |       *c
#  |   *  |       |     *  |
#  |    * |       |   *    |
#  |     *|       | *      |
#  0------a-------b--------1
#

def rule2221(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+1)%4][1]
    ne = edgenodes[(rotation+2)%4][0]
    nf = edgenodes[(rotation+2)%4][1]
    ng = edgenodes[(rotation+3)%4][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, ng], parents=[element]),
        newSkeleton.newElement(nodes=[ng, na, nf], parents=[element]),
        newSkeleton.newElement(nodes=[nf, n3, ng], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, ne, nf], parents=[element]),
        newSkeleton.newElement(nodes=[nb, nc, nd, ne], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, n2, ne], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,2,1), rule2221)

### Quads, cases 2X2X ###

#
#  3------d-------c--------2
#  |      |       |        |
#  |      |       |        |
#  |      |       |        |
#  |      |       |        |
#  |      |       |        |
#  |      |       |        |
#  |      |       |        |
#  |      |       |        |
#  |      |       |        |
#  0------a-------b--------1
#

def rule2020(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+2)%4][0]
    nd = edgenodes[(rotation+2)%4][1]
    return (
        newSkeleton.newElement(nodes=[n0, na, nd, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, n2, nc], parents=[element]))
RefinementRule(liberalRuleSet, (2,0,2,0), rule2020)

#
#  3------e-------d--------2
#  |      |       |*       |
#  |      |       |  *     |
#  |      |       |    *   |
#  |      |       |      * |
#  |      |       |        c
#  |      |       |      * |
#  |      |       |    *   |
#  |      |       |  *     |
#  |      |       |*       |
#  0------a-------b--------1
#

#The case 2021 should reduce to the ff. case.
def rule2120(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+2)%4][0]
    ne = edgenodes[(rotation+2)%4][1]
    return (
        newSkeleton.newElement(nodes=[n0, na, ne, n3], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nd, ne], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nc, n2, nd], parents=[element]))
RefinementRule(liberalRuleSet, (2,1,2,0), rule2120)

#
#  3------e-------d--------2
#  |               *       |
#  |   *             *     |
#  |                   *   |
#  |*                    * |
#  f---------------------- c
#  |*                    * |
#  |                   *   |
#  |   *             *     |
#  |               *       |
#  0------a-------b--------1
#

def rule2121(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+2)%4][0]
    ne = edgenodes[(rotation+2)%4][1]
    nf = edgenodes[(rotation+3)%4][0]
    return (
        newSkeleton.newElement(nodes=[nf, nc, nd, ne], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nf], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, nc, n2], parents=[element]),
        newSkeleton.newElement(nodes=[n0, na, nf], parents=[element]),
        newSkeleton.newElement(nodes=[nf, ne, n3], parents=[element]))
RefinementRule(liberalRuleSet, (2,1,2,1), rule2121)

### Quad, case 2222 ###


#
#  3------f-------e--------2
#  |    * |       |  *     |
#  |  *   |       |     *  |
#  g*     |       |       *d
#  |      |       |        |
#  h*     |       |        |
#  |      |       |       *c
#  |  *   |       |     *  |
#  |      |       |   *    |
#  |     *|       | *      |
#  0------a-------b--------1
#

def rule2222(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2, n3 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%4][0]
    nd = edgenodes[(rotation+1)%4][1]
    ne = edgenodes[(rotation+2)%4][0]
    nf = edgenodes[(rotation+2)%4][1]
    ng = edgenodes[(rotation+3)%4][0]
    nh = edgenodes[(rotation+3)%4][1]
    return (
        newSkeleton.newElement(nodes=[na, nf, ng, nh], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, ne, nf], parents=[element]),
        newSkeleton.newElement(nodes=[nb, nc, nd, ne], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nd, n2, ne], parents=[element]),
        newSkeleton.newElement(nodes=[n0, na, nh], parents=[element]),
        newSkeleton.newElement(nodes=[ng, nf, n3], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,2,2), rule2222)

### Triangles, cases 2XX ###

#                
#                2
#               /\
#              /  \
#             /    \
#            / *  * \
#           /        \
#          /          \
#         /   *    *   \
#        /              \ 
#       /                \
#      /_____*______*_____\
#      0     a       b     1
#

def rule200(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    return (
        newSkeleton.newElement(nodes=[n0, na, n2], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, n2], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, n2], parents=[element]))
RefinementRule(liberalRuleSet, (2,0,0), rule200)

#                
#                2
#               /\
#              /  \
#             /    \
#            c      \
#           /|\      \
#          / | \      \
#         /  |  \      \
#        /   |   \      \ 
#       /    |    \      \
#      /_____|_____\______\
#      0     a       b     1
#

def rule201(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+2)%3][0]
    return (
        newSkeleton.newElement(nodes=[n0, na, nc], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc], parents=[element]),
        newSkeleton.newElement(nodes=[nb, n1, n2, nc], parents=[element]))
RefinementRule(liberalRuleSet, (2,0,1), rule201)

#                
#                2
#               /\
#              /  \
#             /    \
#            /      \
#           /        c
#          /        /|\
#         /        / | \
#        /        /  |  \ 
#       /        /   |   \
#      /________/____|____\
#      0       a     b     1
#

def rule210(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    return (
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc], parents=[element]),
        newSkeleton.newElement(nodes=[n0, na, nc, n2], parents=[element]))
RefinementRule(liberalRuleSet, (2,1,0), rule210)

#                
#                2
#               /\
#              /  \
#             /    \
#            /      \
#           d--------c
#          /|        |\
#         / |        | \
#        /  |        |  \ 
#       /   |        |   \
#      /____|________|____\
#      0    a        b     1
#

def rule211(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    nd = edgenodes[(rotation+2)%3][0]
    return (
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[n0, na, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nc, n2, nd], parents=[element]))
RefinementRule(liberalRuleSet, (2,1,1), rule211)

### Triangles, cases 22X ###

#                
#                2
#               /\
#              /  \
#             /    \
#            /      \
#           /        d
#          /        / \
#         /        /   \
#        /        /    /c 
#       /        /    /  \
#      /________/____/____\
#      0       a     b     1
#

def rule220(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    nd = edgenodes[(rotation+1)%3][1]
    return (
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[n0, na, nd, n2], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,0), rule220)

#                
#                2
#               /\
#              /  \
#             /    \
#            /      \
#           e------- d
#          /*       / \
#         /        /   \
#        /   *    /    /c 
#       /        /    /  \
#      /______*_/____/____\
#      0       a     b     1
#

def rule221(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    nd = edgenodes[(rotation+1)%3][1]
    ne = edgenodes[(rotation+2)%3][0]
    return (
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nd], parents=[element]),
        newSkeleton.newElement(nodes=[n0, na, ne], parents=[element]),
        newSkeleton.newElement(nodes=[ne, na, nd], parents=[element]),
        newSkeleton.newElement(nodes=[nd, n2, ne], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,1), rule221)

### Triangle, case 222 ###

#                
#                2
#               /\
#              /  \
#             /    \
#            /      \
#           e--------d
#          /          \
#         /            \
#        f--------------c 
#       / \           /  \
#      /___\_________/____\
#      0    a        b     1
#

def rule222(element, rotation, cats, edgenodes, newSkeleton):
    n0, n1, n2 = baseNodes(element, rotation)
    na = edgenodes[rotation][0]
    nb = edgenodes[rotation][1]
    nc = edgenodes[(rotation+1)%3][0]
    nd = edgenodes[(rotation+1)%3][1]
    ne = edgenodes[(rotation+2)%3][0]
    nf = edgenodes[(rotation+2)%3][1]
    return (
        newSkeleton.newElement(nodes=[nb, n1, nc], parents=[element]),
        newSkeleton.newElement(nodes=[na, nb, nc, nf], parents=[element]),
        newSkeleton.newElement(nodes=[n0, na, nf], parents=[element]),
        newSkeleton.newElement(nodes=[ne, nd, n2], parents=[element]),
        newSkeleton.newElement(nodes=[nf, nc, nd, ne], parents=[element]))
RefinementRule(liberalRuleSet, (2,2,2), rule222)
