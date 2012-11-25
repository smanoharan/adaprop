package weka.classifiers.mi.adaprop;

import org.junit.BeforeClass;
import org.junit.Test;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Tests the two search strategies: Breadth First and Best First
 */
public class SearchStrategyTest extends TestBase
{
    private static Instances simpleMIdata;
    private static Instances complexMIdata;

    @BeforeClass
    public static void initSimpleMIData() throws Exception
    {
        simpleMIdata = new Instances(new BufferedReader(new FileReader("test/weka/classifiers/mi/test-mi.arff")));
        simpleMIdata.setClassIndex(2);
        complexMIdata = new Instances(new BufferedReader(new FileReader("test/weka/classifiers/mi/test-2-mi.arff")));
        complexMIdata.setClassIndex(2);
    }

    private static void assertAttributeListEquals(List<Attribute> act, String ... exp)
    {
        List<String> actNames = new ArrayList<String>(act.size());
        for (Attribute attr : act)
        {
            actNames.add(attr.name());
        }
        assertListEquals("Attribute List", Arrays.asList(exp), actNames);
    }

    private RootSplitNode buildSimpleTreeWith(int maxNodeCount, SearchStrategy strategy) throws Exception
    {
        final int instCount = 25;
        final OneR classifier = new OneR();
        classifier.setMinBucketSize(1);

        final int numAttr = 3;
        TreeBuildingParams params = new TreeBuildingParams(
                maxNodeCount, 1, simpleMIdata, instCount,
                new MeanSplitStrategy(numAttr), new CountBasedPropositionalisationStrategy(),
                classifier);

        // build root
        return strategy.buildTree(params, instCount, simpleMIdata);
    }

    private RootSplitNode buildComplexTreeWith(int maxNodeCount, SearchStrategy strategy) throws Exception
    {
        final int instCount = 15;
        final OneR classifier = new OneR();
        classifier.setMinBucketSize(1);

        final int numAttr = 2;
        TreeBuildingParams params = new TreeBuildingParams(
                maxNodeCount, 1, complexMIdata, instCount,
                new MeanSplitStrategy(numAttr), new CountBasedPropositionalisationStrategy(),
                classifier);

        // build root
        return strategy.buildTree(params, instCount, complexMIdata);
    }

    private void shouldBeASingleNodeWhenMaxTreeSizeIsOne(SearchStrategy strategy) throws Exception
    {
        final int maxNodeCount = 1;
        RootSplitNode root = buildSimpleTreeWith(maxNodeCount, strategy);

        // should have 1 node
        assertEquals("NodeCount", maxNodeCount, root.getNodeCount());

        // should have 3 attributes (total, count-left, count-right)
        shouldHaveNAttr(3, root);

        // we expect a split on attr2 (index=1) @ 1.3
        assertNodeEquals("root", root, 1, 1.3, 1, 2);
    }

    private static void shouldHaveNAttr(int numAttr, RootSplitNode root)
    {
        assertEquals("PropCount", numAttr, root.getNumPropAttr());
        String[] expAttrs = new String[numAttr];
        for (int attrIndex=0; attrIndex<numAttr; attrIndex++)
        {
            expAttrs[attrIndex] = "region " + attrIndex;
        }
        assertAttributeListEquals(root.getAttrInfo(), expAttrs);
    }

    private static SplitNode deepCopySubtree(SplitNode node)
    {
        SplitNode left = (node.left == null ? null : deepCopySubtree(node.left));
        SplitNode right = (node.right == null ? null : deepCopySubtree(node.right));
        return new SplitNode(node.propLeftIndex, node.propRightIndex, node.splitAttrIndex, node.splitPoint,
                left, right, node.curDepth);
    }

    private static RootSplitNode deepCopyTree(RootSplitNode root, final PropositionalisationStrategy propStrategy)
    {
        SplitNode left = (root.left == null ? null : deepCopySubtree(root.left));
        SplitNode right = (root.right == null ? null : deepCopySubtree(root.right));

        return new RootSplitNode(root.propLeftIndex, root.propRightIndex, root.splitAttrIndex,
                root.splitPoint, left, right, root.curDepth, propStrategy);
    }

    private static double findErrorOnTrainingSet(final TreeBuildingParams params, final Instances trainingData,
                                                 final RootSplitNode root) throws Exception
    {
        final Instances propositionalisedTrainingData = SplitNode.propositionaliseDataset(trainingData, root, params.propStrategy);
        params.classifier.buildClassifier(propositionalisedTrainingData);
        Evaluation evaluation = new Evaluation(propositionalisedTrainingData);
        evaluation.evaluateModel(params.classifier, propositionalisedTrainingData);
        return evaluation.incorrect();
    }

    private static void splitShouldBeOptimal(SplitNode node, RootSplitNode root, TreeBuildingParams params,
                                             Instances trainingData, BitSet nodeIgnore) throws Exception
    {
        List<CompPair<Integer, Double>> splits =
                params.splitStrategy.generateSplitPoints(trainingData, nodeIgnore);
        int curBestSplitIndex = node.splitAttrIndex;
        double curBestSplitVal = node.splitPoint;
        double expLeastErr = findErrorOnTrainingSet(params, trainingData, root);

        for (CompPair<Integer, Double> split : splits)
        {
            // set split to cur-split, eval, check error is not less than least error
            node.splitAttrIndex = split.key;
            node.splitPoint = split.value;
            double curErr = findErrorOnTrainingSet(params, trainingData, root);
            assertTrue(split + " should not be a better split than original", curErr <= expLeastErr);
        }

        // reset node
        node.splitAttrIndex = curBestSplitIndex;
        node.splitPoint = curBestSplitVal;

    }

    private static void assertTreeEquals(String msg, SplitNode act, SplitNode exp)
    {
        // check that the current nodes have the same split ; same indices:
        assertNodeEquals(msg, act, exp);

        // check that both left-children are null or equal
        final String leftMsg = msg + ".left";
        final String rightMsg = msg + ".right";

        if (exp.left == null) { assertNull(leftMsg, act.left); }
        else
        {
            assertNotNull(leftMsg, act.left);
            assertTreeEquals(leftMsg, act.left, exp.left);
        }

        if (exp.right == null) { assertNull(rightMsg, act.right); }
        else
        {
            assertNotNull(rightMsg, act.right);
            assertTreeEquals(rightMsg, act.right, exp.right);
        }
    }

    // TODO : should be subtree of

    /** Check that the node is equal to the expected node */
    private static void assertNodeEquals(String msg, SplitNode actNode, SplitNode expNode)
    {
        assertNodeEquals(msg, actNode, expNode.splitAttrIndex, expNode.splitPoint, expNode.propLeftIndex,
                expNode.propRightIndex);
    }

    /** Check that the node is equal to the expected values */
    private static void assertNodeEquals(String msg, SplitNode actNode, int expSplitAttrIndex, double expSplitPt,
                                         int expPropLeft, int expPropRight)
    {
        assertEquals(msg + "splitAttrIndex", expSplitAttrIndex, actNode.splitAttrIndex);
        assertEquals(msg + "splitPoint", expSplitPt, actNode.splitPoint, TOLERANCE);
        assertEquals(msg + "left-index", expPropLeft, actNode.propLeftIndex);
        assertEquals(msg + "right-index", expPropRight, actNode.propRightIndex);
    }

    private void assertNullOrLeaf(String msg, SplitNode node)
    {
        assertTrue(msg + "node: " + node, node == null || node.splitAttrIndex < 0);
    }

    private void assertNotNullNorLeaf(String msg, SplitNode node)
    {
        assertTrue(msg + "node: " + node, node != null && node.splitAttrIndex >= 0);
    }

    @Test
    public void shouldBeASingleNodeTreeWhenMaxTreeSizeIsOneForBreadthFirst() throws Exception
    {
        shouldBeASingleNodeWhenMaxTreeSizeIsOne(new BreadthFirstSearchStrategy());
    }

    @Test
    public void shouldBeASingleNodeTreeWhenMaxTreeSizeIsOneForBestFirst() throws Exception
    {
        shouldBeASingleNodeWhenMaxTreeSizeIsOne(new BestFirstSearchStrategy());
    }

    @Test
    public void shouldBeA2NodeTreeForBreadthFirst() throws Exception
    {
        final int maxNodeCount = 2;
        SearchStrategy strategy = new BreadthFirstSearchStrategy();
        RootSplitNode root = buildSimpleTreeWith(maxNodeCount, strategy);

        // should have 2 nodes
        assertEquals("NodeCount", maxNodeCount, root.getNodeCount());

        // root should be as per before
        assertNodeEquals("root", root, 1, 1.3, 1, 2);

        // the next node should be on the left (not right)
        assertNotNullNorLeaf("root.left", root.left);
        assertNullOrLeaf("root.right", root.right);

        // expect left to split on attr1 (index=0) @ 0.64 (all splits are equal here though)
        assertNodeEquals("root.left", root.left, 0, 0.638462, 3, 4);
    }

    @Test
    public void shouldBeA3NodeTreeForBreadthFirst() throws Exception
    {
        final int maxNodeCount = 3;
        SearchStrategy strategy = new BreadthFirstSearchStrategy();
        RootSplitNode root = buildSimpleTreeWith(maxNodeCount, strategy);

        // should have 3 nodes
        assertEquals("NodeCount", maxNodeCount, root.getNodeCount());

        // root should be as per before
        assertNodeEquals("root", root, 1, 1.3, 1, 2);

        // the next two nodes should be also present
        assertNotNullNorLeaf("root.left", root.left);
        assertNotNullNorLeaf("root.right", root.right);

        // expect left to split on attr1 (index=0) @ 0.638 (all splits are equal here though)
        assertNodeEquals("root.left", root.left, 0, 0.638462, 3, 4);
        // expect right to split on attr1 @ 0.642 (all splits are equal here though)
        assertNodeEquals("root.right", root.right, 0, 0.6416667, 5, 6);
    }


    // Breadth first, complex tree
    @Test
    public void testComplexDatasetForBreadthFirstWith1Node() throws Exception
    {
        final int maxNodeCount = 1;
        SearchStrategy strategy = new BreadthFirstSearchStrategy();
        RootSplitNode root = buildComplexTreeWith(maxNodeCount, strategy);

        // should have 1 node
        assertEquals("NodeCount", maxNodeCount, root.getNodeCount());

        // root should have two leaf nodes:
        assertNodeEquals("root", root, 0, 0.28, 1, 2);
        assertNullOrLeaf("root.left", root.left);
        assertNullOrLeaf("root.right", root.right);
    }

    @Test
    public void testComplexDatasetForBreadthFirstWith2Nodes() throws Exception
    {
        final int maxNodeCount = 2;
        SearchStrategy strategy = new BreadthFirstSearchStrategy();
        RootSplitNode root = buildComplexTreeWith(maxNodeCount, strategy);

        // should have 2 nodes
        assertEquals("NodeCount", maxNodeCount, root.getNodeCount());

        // root should have left-non-leaf and right-leaf nodes
        assertNodeEquals("root", root, 0, 0.28, 1, 2);
        assertNullOrLeaf("root.right", root.right);
        assertNotNullNorLeaf("root.left", root.left);
        assertNodeEquals("root.left", root.left, 1, 0.25, 3, 4);
        assertNullOrLeaf("root.left.left", root.left.left);
        assertNullOrLeaf("root.left.right", root.left.right);
    }

    @Test
    public void testComplexDatasetForBreadthFirstWith3Nodes() throws Exception
    {
        final int maxNodeCount = 3;
        SearchStrategy strategy = new BreadthFirstSearchStrategy();
        RootSplitNode root = buildComplexTreeWith(maxNodeCount, strategy);

        // should have 3 nodes
        assertEquals("NodeCount", maxNodeCount, root.getNodeCount());

        // root should have two internal nodes as children
        assertNodeEquals("root", root, 0, 0.28, 1, 2);
        assertNotNullNorLeaf("root.left", root.left);
        assertNotNullNorLeaf("root.right", root.right);

        // root.left - should have two leaf nodes
        assertNodeEquals("root.left", root.left, 1, 0.25, 3, 4);
        assertNullOrLeaf("root.left.left", root.left.left);
        assertNullOrLeaf("root.left.right", root.left.right);

        // root.right - should have two leaf nodes
        assertNodeEquals("root.right", root.right, 0, 0.428571, 5, 6);
        assertNullOrLeaf("root.right.left", root.right.left);
        assertNullOrLeaf("root.right.right", root.right.right);
    }
}
