package weka.classifiers.mi.adaprop;

import org.junit.BeforeClass;
import org.junit.Test;
import weka.classifiers.rules.OneR;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
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
        assertNodeEquals(root, 1, 1.3, 1, 2);
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

//    public void shouldBeASingleNodeTreeWhenMaxTreeSizeIsOne() throws Exception
//    {
//        SearchStrategy searchStrategy = new BreadthFirstSearchStrategy();
//        searchStrategy = new BestFirstSearchStrategy();
//
//
//
//        // find the expected split point:
//        double minErr = Double.MAX_VALUE;
//        CompPair<Integer, Double> bestSplit = null;
//        List<CompPair<Integer, Double>> splits =
//                params.splitStrategy.generateSplitPoints(simpleMIdata, new BitSet(instCount));
//        for (CompPair<Integer, Double> split : splits)
//        {
//            // train & eval
//            RootSplitNode newRoot = new RootSplitNode(1,2,split.key, split.value, null, null, 0, params.propStrategy);
//            newRoot.setNodeCount(1);
//            Instances propositionalisedTrainingData = SplitNode.propositionaliseDataset(
//                    simpleMIdata, newRoot, params.propStrategy);
//            System.out.println(propositionalisedTrainingData);
//            params.classifier.buildClassifier(propositionalisedTrainingData);
//            Evaluation evaluation = new Evaluation(propositionalisedTrainingData);
//            evaluation.evaluateModel(params.classifier, propositionalisedTrainingData);
//            double curErr = evaluation.incorrect();
//            System.out.println(split + " " + curErr);
//            if (curErr < minErr)
//            {
//
//                minErr = curErr;
//                bestSplit = split;
//            }
//        }
//
//        // check those are equal
//        assertNotNull(bestSplit);
//
//    }

    // TODO complete rewrite

    /** Check that the node is equal to the expected values */
    private static void assertNodeEquals(SplitNode actNode, int expSplitAttrIndex, double expSplitPt,
                                         int expPropLeft, int expPropRight)
    {
        assertEquals("splitAttrIndex", expSplitAttrIndex, actNode.splitAttrIndex);
        assertEquals("splitPoint", expSplitPt, actNode.splitPoint, TOLERANCE);
        assertEquals("left", expPropLeft, actNode.propLeftIndex);
        assertEquals("right", expPropRight, actNode.propRightIndex);
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
        assertNodeEquals(root, 1, 1.3, 1, 2);

        // the next node should be on the left (not right)
        assertNotNullNorLeaf("root.left", root.left);
        assertNullOrLeaf("root.right", root.right);

        // expect left to split on attr1 (index=0) @ 0.64 (all splits are equal here though)
        assertNodeEquals(root.left, 0, 0.638462, 3, 4);
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
        assertNodeEquals(root, 1, 1.3, 1, 2);

        // the next two nodes should be also present
        assertNotNullNorLeaf("root.left", root.left);
        assertNotNullNorLeaf("root.right", root.right);

        // expect left to split on attr1 (index=0) @ 0.638 (all splits are equal here though)
        assertNodeEquals(root.left, 0, 0.638462, 3, 4);
        // expect right to split on attr1 @ 0.642 (all splits are equal here though)
        assertNodeEquals(root.right, 0, 0.6416667, 5, 6);
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
        assertNodeEquals(root, 0, 0.28, 1, 2);
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
        assertNodeEquals(root, 0, 0.28, 1, 2);
        assertNullOrLeaf("root.right", root.right);
        assertNotNullNorLeaf("root.left", root.left);
        assertNodeEquals(root.left, 1, 0.25, 3, 4);
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
        assertNodeEquals(root, 0, 0.28, 1, 2);
        assertNotNullNorLeaf("root.left", root.left);
        assertNotNullNorLeaf("root.right", root.right);

        // root.left - should have two leaf nodes
        assertNodeEquals(root.left, 1, 0.25, 3, 4);
        assertNullOrLeaf("root.left.left", root.left.left);
        assertNullOrLeaf("root.left.right", root.left.right);

        // root.right - should have two leaf nodes
        assertNodeEquals(root.right, 0, 0.428571, 5, 6);
        assertNullOrLeaf("root.right.left", root.right.left);
        assertNullOrLeaf("root.right.right", root.right.right);
    }
}
