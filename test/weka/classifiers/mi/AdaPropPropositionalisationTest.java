package weka.classifiers.mi;

import org.junit.Test;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Tests that AdaProp propositionalises the MI data-set
 *  correctly given a tree of partitioning hyperplanes.
 *
 * All tests are performed on the example data-set
 *  built up in {@link AdaPropTestBase}.
 *
 * @author Siva Manoharan, 1117707
 */
public class AdaPropPropositionalisationTest extends AdaPropTestBase
{
    // NOTE: the example data-set from the superclass (AdaPropTestBase) is used below.

    /**
     * Test propositionalisation of the example data-set for a single node tree
     *  (where the split is at "a1 <= 26").
     */
    @Test
    public void testPropositionalisationOfOneNodeSplitTreeAtAttributeOne()
    {
        // the split is on a1 <= 26.
        final int splitAttrIndex = 1;
        final double splitPoint = 26;

        // expected propositionalised counts are: (4,0), (2,2), (0,4).
        // the first attribute is always 4 (the total number of instances) and
        // the last attribute is the class index (copied from the bags).
        String[] exp = new String[] {"4,4,0,0", "4,2,2,0", "4,0,4,1"};

        assertPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp);
    }

    /**
     * Test propositionalisation of the example data-set for a single node tree
     *  (where the split is at "a2 <= 44").
     */
    @Test
    public void testPropositionalisationOfOneNodeSplitTreeAtAttributeTwo() throws Exception
    {
        // split: a2 <= 49
        final int splitAttrIndex = 2;
        final double splitPoint = 44;

        // expected propositionalised counts are: (4,0), (4,0), (1,3).
        // the first attribute is always 4 (the total number of instances) and
        // the last attribute is the class index (copied from the bags).
        String[] exp = new String[] {"4,4,0,0", "4,4,0,0", "4,1,3,1"};

        assertPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp);
    }

    /**
     * Test propositionalisation of the example data-set for a single node tree
     *  (where the split is at "a3 <= 11").
     */
    @Test
    public void testPropositionalisationOfOneNodeSplitTreeAtAttributeThree() throws Exception
    {
        // split: a3 <= 11
        final int splitAttrIndex = 3;
        final double splitPoint = 11;

        // expected propositionalised counts are: (2,2), (0,4), (0,4).
        // the first attribute is always 4 (the total number of instances) and
        // the last attribute is the class index (copied from the bags).
        String[] exp = new String[] {"4,2,2,0", "4,0,4,0", "4,0,4,1"};

        assertPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp);
    }

    /**
     * Test propositionalisation of the example data-set for a tree
     *  with multiple nodes (depth of the tree is 3).
     */
    @Test
    public void testPropositionalisationOfMultiNodeSplitTree()
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
        SplitNode splitNode4 = new SplitNode(7, 8, 0, -9, null, null, 3);
        SplitNode splitNode2 = new SplitNode(5, 6, 2, 50, null, null, 3);
        SplitNode splitNode1 = new SplitNode(3, 4, 1, 15, null, splitNode4, 5);
        RootSplitNode root = RootSplitNode.toRootNode(new SplitNode(1, 2, 3, 31, splitNode1, splitNode2, 9),
                new CountBasedPropositionalisationStrategy());
        root.setNodeCount(4);

        String[] exp = new String[] { // the following values were hand-computed:
                "4, 4,0, 3,1, 0,0, 0,1,  0",
                "4, 2,2, 0,2, 2,0, 0,2,  0",
                "4, 0,4, 0,0, 2,2, 0,0,  1"};

        final Instances expInstances = convertToDataset(10, exp);
        final Instances actInstances = SplitNode.propositionaliseDataset(miData, root, new CountBasedPropositionalisationStrategy());
        assertDatasetEquals(expInstances, actInstances);
    }

    /**
     * Test propositionalisation of the example data-set for a single node tree,
     *      by trying out all the attributes and all relevant split points.
     *
     * In other words, exhaustively test over all (relevant) single node split trees.
     */
    @Test
    public void testPropositionalisationOfSingleNodeSplitTreeOverAllAttributes() throws Exception
    {
        final int numInst = NUM_INST_PER_BAG*NUM_BAGS;

        // for each attribute
        for (int attrIndex = 0; attrIndex < NUM_ATTR; attrIndex++)
        {
            for (int instToLeft = 0; instToLeft <= numInst; instToLeft++)
            {
                double splitPt = instToLeft*NUM_ATTR + attrIndex - 0.5;
                Instances expected = expectedPropositionalisedBagFor(instToLeft);
                Instances actual = actualPropositionalisedBagFor(attrIndex, splitPt, new CountBasedPropositionalisationStrategy());

                assertDatasetEquals(expected, actual);
            }
        }
    }

    // <editor-fold desc="===Helper Methods===">
    /**
     * Convert a string (format "attr1,attr2,...") to an instance.
     *
     * @param numAttrInclClass Number of attributes, including the class attribute.
     * @param str The string representing the instance, in csv form.
     * @return The generated instance.
     */
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

    /**
     * Convert an array of string-represented instances into a dataset of weka instances.
     *
     * @param numAttrInclClass Number of attributes, including the class attribute.
     * @param instances The array of strings, each representing an instance in csv form.
     * @return The generated set of instances.
     */
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

    /**
     * Compute the actual propositionalised bag when the tree has just one node (the root).
     *
     *
     * @param attrIndex The attribute to split on (for the root node).
     * @param split The value to split along the specified attribute.
     * @param propStrategy
     * @return The propositionalised set of instances.
     */
    private static Instances actualPropositionalisedBagFor(final int attrIndex, final double split,
                                                           final PropositionalisationStrategy propStrategy) throws Exception
    {
        final Instances result = new Instances(propHeader, NUM_BAGS);
        final RootSplitNode root = createRootSplit(attrIndex, split);

        // for each bag
        for (int bagIndex = 0; bagIndex < NUM_BAGS; bagIndex++)
        {
            final Instance bag = miData.get(bagIndex);

            // find actual value:
            result.add(SplitNode.propositionaliseBag(bag, root, result, propStrategy));
        }

        return result;
    }

    /**
     * Compute the expected propositionalised bag for
     *  a single level split.
     * This is done assuming that the data-set is sorted
     *  (which is true for the example data-set),
     *  therefore the result can be found by just placing
     *  the first N instances into one bucket and the
     *  remanining into another.
     *
     * @param instToLeft The expected number of instances to the left of the split point.
     * @return The expected set of instances.
     */
    protected static Instances expectedPropositionalisedBagFor(
            final int instToLeft) throws Exception
    {
        final Instances result = new Instances(propHeader, NUM_BAGS);
        int instRemainingToLeft = instToLeft;

        // for each bag
        for (int bagIndex = 0; bagIndex < NUM_BAGS; bagIndex++)
        {
            final Instance bag = miData.get(bagIndex);

            // find expected
            int countLt = 0;
            int countGeq = 0;
            if (instRemainingToLeft == 0)
            {
                countGeq = NUM_INST_PER_BAG;
            }
            else if (instRemainingToLeft < NUM_INST_PER_BAG)
            {
                countLt = instRemainingToLeft;
                countGeq = NUM_INST_PER_BAG - instRemainingToLeft;
                instRemainingToLeft = 0;
            }
            else
            {
                instRemainingToLeft -= NUM_INST_PER_BAG;
                countLt = NUM_INST_PER_BAG;
            }

            final double[] attValues = {countLt + countGeq, countLt, countGeq, bag.classValue()};
            Instance expInst = new DenseInstance(1.0, attValues);
            expInst.setDataset(result);
            result.add(expInst);
        }

        return result;
    }

    /**
     * Check if the propositionalisation of the example data-set
     *  using a single node split tree (splitting at the specified
     *  attribute and split point) is as specified in exp.
     *
     * @param splitAttrIndex The attribute to split on.
     * @param splitPoint The value to split on.
     * @param exp The expected propositionalised dataset, as an array of csv-formatted strings.
     */
    private static void assertPropositionalisationOfOneNodeSplitTreeIs(final int splitAttrIndex,
                                                                       final double splitPoint, final String[] exp)
    {
        // build the single-node split tree:
        final RootSplitNode root = RootSplitNode.toRootNode(
                new SplitNode(1, 2, splitAttrIndex, splitPoint, null, null, 3),
                new CountBasedPropositionalisationStrategy());
        root.setNodeCount(1);

        // when propositionalised using one split, there are always 4 attributes:
        //      count(all) ; count(left-of-split) ; count(right-of-split) ; class-index
        final int numAttrInclClass = 4;
        final Instances actual = SplitNode.propositionaliseDataset(miData, root, new CountBasedPropositionalisationStrategy());
        final Instances expected = convertToDataset(numAttrInclClass, exp);
        assertDatasetEquals(expected, actual);
    }
    // </editor-fold>
}
