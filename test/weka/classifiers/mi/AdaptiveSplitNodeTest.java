package weka.classifiers.mi;

import org.junit.Test;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.BitSet;

import static org.junit.Assert.assertEquals;

/** Author: Siva Manoharan, 1117707 */
public class AdaptiveSplitNodeTest extends AdaptiveSplitTest
{
    /** Check that the node is equal to the expected values */
    private static void assertNodeEquals(SplitNode actNode, int expSplitAttrIndex, int expSplitPt, SplitNode expLeft, SplitNode expRight)
    {
        assertEquals("splitAttrIndex", expSplitAttrIndex, actNode.splitAttrIndex);
        assertEquals("splitPoint", expSplitPt, actNode.splitPoint, AdaptiveSplitTest.TOLERANCE);
        assertEquals("left", expLeft, actNode.left);
        assertEquals("right", expRight, actNode.right);
    }

    /** Check that the node is a leaf */
    private static void assertNodeIsALeaf(SplitNode node)
    {
        assertNodeEquals(node, -1, 0, null, null);
    }

    /** Convert a string (format "attr1,attr2,...") to an instance */
    private static Instance convertToInstance(int numAttrInclClass, String str)
    {
        double[] attVals = new double[numAttrInclClass];
        String[] sVals = str.split(",");
        for (int i=0; i<numAttrInclClass; i++)
        {
            attVals[i] = Double.parseDouble(sVals[i]);
        }
        return new DenseInstance(1, attVals);
    }

    /** Convert an array of string-represented instances into a dataset of weka instances */
    private static Instances convertToDataset(int numAttrInclClass, String ... instances)
    {
        ArrayList<Attribute> attInfo = new ArrayList<Attribute>(numAttrInclClass);
        for (int i=0; i<numAttrInclClass-1; i++)
        {
            attInfo.add(new Attribute("attr"+i));
        }

        attInfo.add(createClassAttribute());

        Instances dataset = new Instances("test", attInfo, instances.length);
        for (String instStr : instances)
        {
            Instance inst = convertToInstance(numAttrInclClass, instStr);
            inst.setDataset(dataset);
            dataset.add(inst);
        }
        return dataset;
    }

    // TODO - remove?
    private static Instances convertToMIDataset(String[] ... bags)
    {
        ArrayList<Attribute> attInfo = new ArrayList<Attribute>(3);
        ArrayList<String> bagIDs = new ArrayList<String>(bags.length);
        for (int i=0; i<bags.length; i++)
        {
            bagIDs.add("bag"+i);
        }

        attInfo.add(new Attribute("bag-id", bagIDs));
        attInfo.add(new Attribute("bag"));
        attInfo.add(new Attribute("class"));

        return null;
    }

    // try expanding a leaf node, check that it fails
    private static void assertNodeIsNotExpanded(final int maxDepth, final int minOcc, final int instCount)
    {
        try
        {
            SplitNode.TreeBuildingParams p = new SplitNode.TreeBuildingParams(maxDepth, minOcc, null, instCount, null, null);
            SplitNode node = SplitNode.newLeafNode(0, 0);
            node.expand(p, new BitSet(instCount), null, 4);
            assertNodeIsALeaf(node);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void ShouldBeALeafWhenMaxDepthIsNotPositive()
    {
        assertNodeIsNotExpanded(0, 0, 999);
        assertNodeIsNotExpanded(-1, 0, 999);
        assertNodeIsNotExpanded(-190, 0, 999);
    }

    @Test
    public void ShouldBeALeafWhenMinCapacityIsNotMet()
    {
        assertNodeIsNotExpanded(999, 5, 4);
        assertNodeIsNotExpanded(999, 5, 0);
        assertNodeIsNotExpanded(999, 2, 1);
    }

    /** Create a new leaf node */
    private static SplitNode createLeafNode(int propIndex)
    {
        return new SplitNode(propIndex, -1, 0, null, null, 1);
    }

    @Test
    public void TestPropositionalisationOfOneLevelSplitNode()
    {
        // expect: one instance per attribute.
        // the split is on attr2 <= 26. Thus we expect (4,0), (2,2), (0,4) as the prop values (retaining class vals).
        final int splitPoint = 26;
        String[] expStrs = new String[] {"4,4,0,0", "4,2,2,0", "4,0,4,1"};

        final RootSplitNode root = RootSplitNode.toRootNode(new SplitNode(0, 1, splitPoint, createLeafNode(1), createLeafNode(2), 3));
        root.setNodeCount(3);
        assertDatasetEquals(convertToDataset(4, expStrs), SplitNode.propositionaliseDataset(miData, root));
    }

    @Test
    public void TestPropositionalisationOfMultiLevelSplitTree()
    {
        // Test a tree with 3 levels as follows: (attrs are 0-based)
        /*  [0]:=<a3,31>
         *              -- <=: [1]:=<a1,15>
         *                                      -- <=: [3]:<>
         *                                      --  >: [4]:=<a0,-9>
         *                                                              -- <=: [7]:<>
         *                                                              --  >: [8]:<>
         *              --  >: [2]:=<a2,50>
         *                                      -- <=: [5]:<>
         *                                      --  >: [6]:<>
         */
        SplitNode splitNode4 = new SplitNode(4, 0, -9, createLeafNode(7), createLeafNode(8), 3);
        SplitNode splitNode2 = new SplitNode(2, 2, 50, createLeafNode(5), createLeafNode(6), 3);
        SplitNode splitNode1 = new SplitNode(1, 1, 15, createLeafNode(3), splitNode4, 5);
        RootSplitNode root = RootSplitNode.toRootNode(new SplitNode(0, 3, 31, splitNode1, splitNode2, 9));
        root.setNodeCount(9);

        String[] expStrs = new String[] {
                "4, 4,0, 3,1, 0,0, 0,1,  0",
                "4, 2,2, 0,2, 2,0, 0,2,  0",
                "4, 0,4, 0,0, 2,2, 0,0,  1"};
        assertDatasetEquals(convertToDataset(10, expStrs), SplitNode.propositionaliseDataset(miData, root));
    }

}
