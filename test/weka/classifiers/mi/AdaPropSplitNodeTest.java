package weka.classifiers.mi;

import org.junit.Test;

import java.util.BitSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/** Author: Siva Manoharan, 1117707 */
public class AdaPropSplitNodeTest extends AdaPropTestBase
{
    // TODO complete rewrite

    /** Check that the node is equal to the expected values */
    private static void assertNodeEquals(SplitNode actNode, int expSplitAttrIndex, int expSplitPt, SplitNode expLeft, SplitNode expRight)
    {
        assertEquals("splitAttrIndex", expSplitAttrIndex, actNode.splitAttrIndex);
        assertEquals("splitPoint", expSplitPt, actNode.splitPoint, AdaPropOptionsTest.TOLERANCE);
        assertEquals("left", expLeft, actNode.left);
        assertEquals("right", expRight, actNode.right);
    }

    /** Check that the node is a leaf */
    private static void assertNodeIsALeaf(SplitNode node)
    {
        assertNodeEquals(node, -1, 0, null, null);
    }

    // try expanding a leaf node, check that it fails
    private static void assertNodeIsNotExpanded(final int maxDepth, final int minOcc, final int instCount)
    {
        TreeBuildingParams params = new TreeBuildingParams((1 << maxDepth), minOcc,
                null, instCount, null, new CountBasedPropositionalisationStrategy(), null);
        SplitNode node = new SplitNode(1, 2, 0);
        BreadthFirstSearchStrategy bfs = new BreadthFirstSearchStrategy();
        assertFalse(bfs.isExpandable(node, params, new BitSet(instCount)));
    }

//    @Test
//    public void ShouldBeALeafWhenMaxDepthIsNotPositive()
//    {
//        assertNodeIsNotExpanded(0, 0, 999);
//        assertNodeIsNotExpanded(-1, 0, 999);
//        assertNodeIsNotExpanded(-190, 0, 999);
//    }

    @Test
    public void ShouldBeALeafWhenMinCapacityIsNotMet()
    {
        assertNodeIsNotExpanded(999, 5, 4);
        assertNodeIsNotExpanded(999, 5, 0);
        assertNodeIsNotExpanded(999, 2, 1);
    }
}
