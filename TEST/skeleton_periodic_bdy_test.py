# -*- python -*-

# This software was produced by NIST, an agency of the U.S. government,
# and by statute is not subject to copyright in the United States.
# Recipients of this software assume all responsibilities associated
# with its operation, modification and maintenance. However, to
# facilitate maintenance we ask that before distributing modified
# versions of this software, you first contact the authors at
# oof_manager@nist.gov. 

# Test suite for skeleton boundary construction, modification, and
# deletion.  Currently only creates boundaries from selections, not
# groups.  This test should follow the more basic skeleton_test tests.

import unittest, os
from . import memorycheck
from .UTILS.file_utils import reference_file

from .skeleton_bdy_test import nodesFromEdgeBdy

class Skeleton_Boundary(unittest.TestCase):
    def setUp(self):
        global skeletoncontext
        from ooflib.engine import skeletoncontext
        OOF.Microstructure.Create_From_ImageFile(
            filename=reference_file("ms_data","small.ppm"),
            microstructure_name="skeltest",
            height=20.0, width=20.0)
        OOF.Image.AutoGroup(image="skeltest:small.ppm")
        global gfxmanager
        from ooflib.common.IO import gfxmanager
        OOF.Skeleton.New(
            name="bdytest",
            microstructure="skeltest", 
            x_elements=8, y_elements=8,
            skeleton_geometry=TriSkeleton(arrangement="middling",
                                          left_right_periodicity=True,
                                          top_bottom_periodicity=True))

        # Need a graphics window so we can do the direct selection.
        OOF.Windows.Graphics.New()
        self.sk_context = skeletoncontext.skeletonContexts[
            "skeltest:bdytest"]

    def tearDown(self):
        OOF.Graphics_1.File.Close()

    # Check that the default boundaries exist and are the right size.
    # As with most tests, this could do more, i.e. ensure edges are
    # exterior, check that indices are as expected, etc.
    @memorycheck.check("skeltest")
    def Defaults(self):
        default_edges = ["top", "bottom", "left", "right"]
        self.assertEqual(len(self.sk_context.edgeboundaries), 4)
        for e in self.sk_context.edgeboundaries.keys():
            self.assertTrue(e in default_edges)
            default_edges.remove(e)
            
        default_points = ["topleft", "topright", "bottomleft", "bottomright"]
        self.assertEqual(len(self.sk_context.pointboundaries), 4)
        for p in self.sk_context.pointboundaries.keys():
            self.assertTrue(p in default_points)
            default_points.remove(p)

        for e in self.sk_context.edgeboundaries.values():
            self.assertEqual(e.current_size(), 8)

        for p in self.sk_context.pointboundaries.values():
            self.assertEqual(p.current_size(), 1)


    @memorycheck.check("skeltest")
    def Construct_Edge_from_Elements(self):
        OOF.Graphics_1.Toolbox.Select_Element.Rectangle(
            skeleton="skeltest:bdytest",
            points=[Point(11,3), Point(19,-.5)],
            shift=0, ctrl=0)
        OOF.Graphics_1.Toolbox.Select_Element.Rectangle(
            skeleton="skeltest:bdytest",
            points=[Point(11,17), Point(19,20.5)],
            shift=0, ctrl=1)
        OOF.Skeleton.Boundary.Construct(
            skeleton="skeltest:bdytest", name="test",
            constructor=EdgeFromElements(group=selection,
                                         direction="Clockwise"))
        self.assertTrue("test" in self.sk_context.edgeboundaries.keys())
        test_bdy = self.sk_context.edgeboundaries["test"]
        self.assertEqual(test_bdy.current_size(), 8)

    @memorycheck.check("skeltest")
    def Construct_Edge_from_Segments(self):
        OOF.Graphics_1.Toolbox.Select_Segment.Rectangle(
            skeleton="skeltest:bdytest",
            points=[Point(11,3), Point(19,-.5)],
            shift=0, ctrl=0)
        OOF.Graphics_1.Toolbox.Select_Segment.Rectangle(
            skeleton="skeltest:bdytest",
            points=[Point(11,17), Point(19,20.5)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(13.5,20)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(16.5,20)],
            shift=0, ctrl=1) 
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(13.5,0)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(16.5,0)],
            shift=0, ctrl=1) 
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(15,19)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(15,1)],
            shift=0, ctrl=1)      
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(16.25,1.25)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(13.75,1.25)],
            shift=0, ctrl=1)      
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(16.25,18.75)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(13.75,18.75)],
            shift=0, ctrl=1)       
        OOF.Skeleton.Boundary.Construct(
            skeleton="skeltest:bdytest", name="test",
            constructor=EdgeFromSegments(group=selection,
                                         direction="Clockwise"))
        self.assertTrue("test" in self.sk_context.edgeboundaries.keys())
        test_bdy = self.sk_context.edgeboundaries["test"]
        self.assertEqual(test_bdy.current_size(), 8)

    @memorycheck.check("skeltest")
    def Construct_Edge_from_Nodes(self):
        OOF.Graphics_1.Toolbox.Select_Node.Rectangle(
            skeleton="skeltest:bdytest",
            points=[Point(11,3), Point(19,-.5)],
            shift=0, ctrl=0)
        OOF.Graphics_1.Toolbox.Select_Node.Rectangle(
            skeleton="skeltest:bdytest",
            points=[Point(11,17), Point(19,20.5)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Node.Single_Node(
            skeleton="skeltest:bdytest",
            points=[Point(15,0)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Node.Single_Node(
            skeleton="skeltest:bdytest",
            points=[Point(15,20)],
            shift=0, ctrl=1)
        OOF.Skeleton.Boundary.Construct(
            skeleton="skeltest:bdytest", name="test",
            constructor=EdgeFromNodes(group=selection,
                                      direction="Clockwise"))
        self.assertTrue("test" in self.sk_context.edgeboundaries.keys())
        test_bdy = self.sk_context.edgeboundaries["test"]
        self.assertEqual(test_bdy.current_size(), 8)

    @memorycheck.check("skeltest")
    def Construct_Edge_from_Nodes2(self):
        OOF.Graphics_1.Toolbox.Select_Node.Single_Node(
            skeleton='skeltest:bdytest',
            points=[Point(17.418666666666667,9.86133333333333)],
            shift=False, ctrl=False)
        OOF.Graphics_1.Toolbox.Select_Node.Single_Node(
            skeleton='skeltest:bdytest',
            points=[Point(19.914666666666665,9.930666666666665)],
            shift=True, ctrl=False)
        OOF.Graphics_1.Toolbox.Select_Node.Single_Node(
            skeleton='skeltest:bdytest',
            points=[Point(2.5033333333333347,12.357333333333331)],
            shift=True, ctrl=False)
        OOF.Graphics_1.Toolbox.Select_Node.Single_Node(
            skeleton='skeltest:bdytest',
            points=[Point(4.652666666666668,12.495999999999999)],
            shift=True, ctrl=False)
        OOF.Skeleton.Boundary.Construct(
            skeleton='skeltest:bdytest',
            name='test',
            constructor=EdgeFromNodes(group=selection,
                                      direction='Left to right'))
        self.assertEqual(nodesFromEdgeBdy(self.sk_context, "test"),
                         [(43, 44), (36, 46), (46, 47)])
        # Select the periodic partner of one of the nodes selected
        # above.  The bdy should not change.
        OOF.Graphics_1.Toolbox.Select_Node.Single_Node(
            skeleton='skeltest:bdytest',
            points=[Point(0.2586666666666664,9.861333333333333)],
            shift=True, ctrl=False)
        OOF.Skeleton.Boundary.Construct(
            skeleton='skeltest:bdytest',
            name='test2',
            constructor=EdgeFromNodes(group=selection,
                                      direction='Left to right'))
        self.assertEqual(nodesFromEdgeBdy(self.sk_context, "test2"),
                         [(43, 44), (36, 46), (46, 47)])

    @memorycheck.check("skeltest")
    def Winding_Test(self):
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(1.25,8.75)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(3.75,6.25)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(6.25,3.75)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(8.75,1.25)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(11.25,18.75)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(13.75,16.25)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(16.25,13.75)],
            shift=0, ctrl=1)
        OOF.Graphics_1.Toolbox.Select_Segment.Single_Segment(
            skeleton="skeltest:bdytest",
            points=[Point(18.75,11.25)],
            shift=0, ctrl=1)
        OOF.Skeleton.Boundary.Construct(
            skeleton="skeltest:bdytest", name="test",
            constructor=EdgeFromSegments(group=selection,
                                      direction="Left to right"))
        self.assertTrue("test" in self.sk_context.edgeboundaries.keys())
        test_bdy = self.sk_context.edgeboundaries["test"]
        self.assertEqual(test_bdy.current_size(), 8)

        
test_set = [
    Skeleton_Boundary("Defaults"),
    Skeleton_Boundary("Construct_Edge_from_Elements"),
    Skeleton_Boundary("Construct_Edge_from_Segments"),
    Skeleton_Boundary("Construct_Edge_from_Nodes"),
    Skeleton_Boundary("Construct_Edge_from_Nodes2"),
    Skeleton_Boundary("Winding_Test")
]

