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
    // <editor-fold desc="===Count Based Propositionalisation===">

    /**
     * Test propositionalisation of the example data-set for a single node tree
     *  (where the split is at "a1 <= 26").
     */
    @Test
    public void testCountBasedPropositionalisationOfOneNodeSplitTreeAtAttributeOne()
    {
        // the split is on a1 <= 26.
        final int splitAttrIndex = 1;
        final double splitPoint = 26;

        // expected propositionalised counts are: (4,0), (2,2), (0,4).
        // the first attribute is always 4 (the total number of instances) and
        // the last attribute is the class index (copied from the bags).
        String[] exp = new String[] {"4,4,0,0", "4,2,2,0", "4,0,4,1"};


        assertCountBasedPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp);
    }

    /**
     * Test propositionalisation of the example data-set for a single node tree
     *  (where the split is at "a2 <= 44").
     */
    @Test
    public void testCountBasedPropositionalisationOfOneNodeSplitTreeAtAttributeTwo() throws Exception
    {
        // split: a2 <= 49
        final int splitAttrIndex = 2;
        final double splitPoint = 44;

        // expected propositionalised counts are: (4,0), (4,0), (1,3).
        // the first attribute is always 4 (the total number of instances) and
        // the last attribute is the class index (copied from the bags).
        String[] exp = new String[] {"4,4,0,0", "4,4,0,0", "4,1,3,1"};

        assertCountBasedPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp);
    }

    /**
     * Test propositionalisation of the example data-set for a single node tree
     *  (where the split is at "a3 <= 11").
     */
    @Test
    public void testCountBasedPropositionalisationOfOneNodeSplitTreeAtAttributeThree() throws Exception
    {
        // split: a3 <= 11
        final int splitAttrIndex = 3;
        final double splitPoint = 11;

        // expected propositionalised counts are: (2,2), (0,4), (0,4).
        // the first attribute is always 4 (the total number of instances) and
        // the last attribute is the class index (copied from the bags).
        String[] exp = new String[] {"4,2,2,0", "4,0,4,0", "4,0,4,1"};

        assertCountBasedPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp);
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
        final PropositionalisationStrategy propStrategy = new CountBasedPropositionalisationStrategy();
        RootSplitNode root = RootSplitNode.toRootNode(new SplitNode(1, 2, 3, 31, splitNode1, splitNode2, 9),
                propStrategy);
        root.setNodeCount(4);

        String[] exp = new String[] { // the following values were hand-computed:
                "4, 4,0, 3,1, 0,0, 0,1,  0",
                "4, 2,2, 0,2, 2,0, 0,2,  0",
                "4, 0,4, 0,0, 2,2, 0,0,  1"};

        final Instances expInstances = convertToDataset(10, exp);
        final Instances actInstances = SplitNode.propositionaliseDataset(miData, root, propStrategy);
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

    // </editor-fold>

    // <editor-fold desc="===Summary Stat Based Propositionalisation===">
    private static final String BAG1_SUMMARY_STRING =
        "4,30,0,15,7.5," + // attr 1
        "4,34,1,16,8.5," + // attr 2
        "4,38,2,17,9.5," + // attr 3
        "4,42,3,18,10.5," + // attr 4
        "4,46,4,19,11.5,"; // attr 5
    private static final String BAG2_SUMMARY_STRING =
        "4,110,20,35,27.5," + // attr 1
        "4,114,21,36,28.5," + // attr 2
        "4,118,22,37,29.5," + // attr 3
        "4,122,23,38,30.5," + // attr 4
        "4,126,24,39,31.5,"; // attr 5
    private static final String BAG3_SUMMARY_STRING =
        "4,190,40,55,47.5," + // attr 1
        "4,194,41,56,48.5," + // attr 2
        "4,198,42,57,49.5," + // attr 3
        "4,202,43,58,50.5," + // attr 4
        "4,206,44,59,51.5,"; // attr 5
    private static final String EMPTY_REGION_SUMMARY_STRING =
        "0,0,0,0,0," + // attr 1
        "0,0,0,0,0," + // attr 2
        "0,0,0,0,0," + // attr 3
        "0,0,0,0,0," + // attr 4
        "0,0,0,0,0,"; // attr 5
    /**
     * Test propositionalisation of the example data-set for a single node tree
     *  (where the split is at "a1 <= 26").
     */
    @Test
    public void testSummaryBasedPropositionalisationOfOneNodeSplitTreeAtAttributeOne()
    {
        // the split is on a1 <= 26.
        final int splitAttrIndex = 1;
        final double splitPoint = 26;

        String[] exp = new String[] {
                // bag 1
                BAG1_SUMMARY_STRING + // region 1 (all incl)
                BAG1_SUMMARY_STRING + // region 2 (all incl)
                EMPTY_REGION_SUMMARY_STRING + // region 3 (nothing incl)
                "0", //  class

                // bag 2
                BAG2_SUMMARY_STRING + // region 1 (all incl)
                //  region 2 (half incl)
                    "2,45,20,25,22.5," + // attr 1
                    "2,47,21,26,23.5," + // attr 2
                    "2,49,22,27,24.5," + // attr 3
                    "2,51,23,28,25.5," + // attr 4
                    "2,53,24,29,26.5," + // attr 5
                //  region 3 (half incl)
                    "2,65,30,35,32.5," + // attr 1
                    "2,67,31,36,33.5," + // attr 2
                    "2,69,32,37,34.5," + // attr 3
                    "2,71,33,38,35.5," + // attr 4
                    "2,73,34,39,36.5," + // attr 5
                "0", // class

                // bag 3
                BAG3_SUMMARY_STRING + // region 1 (all incl)
                EMPTY_REGION_SUMMARY_STRING + // region 2 (nothing incl)
                BAG3_SUMMARY_STRING + // region 3 (all incl)
                "1", // class
        };

        assertSummaryStatBasedPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp);
    }

    /**
     * Test propositionalisation of the example data-set for a single node tree
     *  (where the split is at "a2 <= 44").
     */
    @Test
    public void testSummaryBasedPropositionalisationOfOneNodeSplitTreeAtAttributeTwo() throws Exception
    {
        // split: a2 <= 49
        final int splitAttrIndex = 2;
        final double splitPoint = 44;

        String[] exp = new String[] {
                // bag 1
                BAG1_SUMMARY_STRING + // region 1 (all incl)
                BAG1_SUMMARY_STRING + // region 2 (all incl)
                EMPTY_REGION_SUMMARY_STRING + // region 3 (nothing incl)
                "0", //  class

                // bag 2
                BAG2_SUMMARY_STRING + // region 1 (all incl)
                BAG2_SUMMARY_STRING + // region 2 (all incl)
                EMPTY_REGION_SUMMARY_STRING + // region 3 (nothing incl)
                "0", //  class

                // bag 3
                BAG3_SUMMARY_STRING + // region 1 (all incl)
                //  region 2 (half incl)
                    "1,40,40,40,40," + // attr 1
                    "1,41,41,41,41," + // attr 2
                    "1,42,42,42,42," + // attr 3
                    "1,43,43,43,43," + // attr 4
                    "1,44,44,44,44," + // attr 5
                //  region 3 (half incl)
                    "3,150,45,55,50," + // attr 1
                    "3,153,46,56,51," + // attr 2
                    "3,156,47,57,52," + // attr 3
                    "3,159,48,58,53," + // attr 4
                    "3,162,49,59,54," + // attr 5
                "1", // class
        };

        assertSummaryStatBasedPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp);
    }

    /**
     * Test propositionalisation of the example data-set for a single node tree
     *  (where the split is at "a3 <= 11").
     */
    @Test
    public void testSummaryBasedPropositionalisationOfOneNodeSplitTreeAtAttributeThree() throws Exception
    {
        // split: a3 <= 11
        final int splitAttrIndex = 3;
        final double splitPoint = 11;

        String[] exp = new String[] {
                // bag 1
                BAG1_SUMMARY_STRING + // region 1 (all incl)
                //  region 2 (half incl)
                    "2,5,0,5,2.5," + // attr 1
                    "2,7,1,6,3.5," + // attr 2
                    "2,9,2,7,4.5," + // attr 3
                    "2,11,3,8,5.5," + // attr 4
                    "2,13,4,9,6.5," + // attr 5
                //  region 3 (half incl)
                    "2,25,10,15,12.5," + // attr 1
                    "2,27,11,16,13.5," + // attr 2
                    "2,29,12,17,14.5," + // attr 3
                    "2,31,13,18,15.5," + // attr 4
                    "2,33,14,19,16.5," + // attr 5
                "0", // class

                // bag 2
                BAG2_SUMMARY_STRING +           // region 1 (all incl)
                EMPTY_REGION_SUMMARY_STRING +   // region 2 (nothing incl)
                BAG2_SUMMARY_STRING +           // region 3 (all incl)
                "0", //  class

                // bag 3
                BAG3_SUMMARY_STRING +         // region 1 (all incl)
                EMPTY_REGION_SUMMARY_STRING + // region 2 (nothing incl)
                BAG3_SUMMARY_STRING +         // region 3 (all incl)
                "1", //  class
        };

        assertSummaryStatBasedPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp);
    }

    /**
     * Test propositionalisation of the example data-set for a tree
     *  with multiple nodes (depth of the tree is 3).
     */
    @Test
    public void testSummaryBasedPropositionalisationOfMultiNodeSplitTree()
    {
        final PropositionalisationStrategy propStrategy = new SummaryStatsBasedPropositionalisationStrategy(NUM_ATTR);
        final int nodeCount = 4;
        final int numRegions = nodeCount*2 + 1;
        final int numSummaryStats = SummaryStatsBasedPropositionalisationStrategy.SummaryStatCalculator.NUM_ATTR;
        final int propOffset = NUM_ATTR * numSummaryStats;

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

        SplitNode splitNode4 = new SplitNode(7*propOffset, 8*propOffset, 0, -9, null, null, 3);
        SplitNode splitNode2 = new SplitNode(5*propOffset, 6*propOffset, 2, 50, null, null, 3);
        SplitNode splitNode1 = new SplitNode(3*propOffset, 4*propOffset, 1, 15, null, splitNode4, 5);
        RootSplitNode root = RootSplitNode.toRootNode(new SplitNode(propOffset, 2*propOffset, 3, 31, splitNode1, splitNode2, 9),
                propStrategy);
        root.setNodeCount(nodeCount);

        String[] exp = new String[] { // the following values were hand-computed:
                // bag 1
                BAG1_SUMMARY_STRING + // region 0
                BAG1_SUMMARY_STRING + // region 1
                EMPTY_REGION_SUMMARY_STRING + // region 2
                //  region 3 (x3)
                    "3,15,0,10,5," + // attr 1
                    "3,18,1,11,6," + // attr 2
                    "3,21,2,12,7," + // attr 3
                    "3,24,3,13,8," + // attr 4
                    "3,27,4,14,9," + // attr 5
                //  region 4 (x1)
                    "1,15,15,15,15," + // attr 1
                    "1,16,16,16,16," + // attr 2
                    "1,17,17,17,17," + // attr 3
                    "1,18,18,18,18," + // attr 4
                    "1,19,19,19,19," + // attr 5
                EMPTY_REGION_SUMMARY_STRING + // region 5
                EMPTY_REGION_SUMMARY_STRING + // region 6
                EMPTY_REGION_SUMMARY_STRING + // region 7
                //  region 8 (x1)
                    "1,15,15,15,15," + // attr 1
                    "1,16,16,16,16," + // attr 2
                    "1,17,17,17,17," + // attr 3
                    "1,18,18,18,18," + // attr 4
                    "1,19,19,19,19," + // attr 5
                "0", // class

                // bag 2
                BAG2_SUMMARY_STRING + // region 0
                //  region 1 (x2)
                    "2,45,20,25,22.5," + // attr 1
                    "2,47,21,26,23.5," + // attr 2
                    "2,49,22,27,24.5," + // attr 3
                    "2,51,23,28,25.5," + // attr 4
                    "2,53,24,29,26.5," + // attr 5
                //  region 2 (x2)
                    "2,65,30,35,32.5," + // attr 1
                    "2,67,31,36,33.5," + // attr 2
                    "2,69,32,37,34.5," + // attr 3
                    "2,71,33,38,35.5," + // attr 4
                    "2,73,34,39,36.5," + // attr 5
                EMPTY_REGION_SUMMARY_STRING + // region 3
                //  region 4 (x2)
                    "2,45,20,25,22.5," + // attr 1
                    "2,47,21,26,23.5," + // attr 2
                    "2,49,22,27,24.5," + // attr 3
                    "2,51,23,28,25.5," + // attr 4
                    "2,53,24,29,26.5," + // attr 5
                //  region 5 (x2)
                    "2,65,30,35,32.5," + // attr 1
                    "2,67,31,36,33.5," + // attr 2
                    "2,69,32,37,34.5," + // attr 3
                    "2,71,33,38,35.5," + // attr 4
                    "2,73,34,39,36.5," + // attr 5
                EMPTY_REGION_SUMMARY_STRING + // region 6
                EMPTY_REGION_SUMMARY_STRING + // region 7
                //  region 4 (x2)
                    "2,45,20,25,22.5," + // attr 1
                    "2,47,21,26,23.5," + // attr 2
                    "2,49,22,27,24.5," + // attr 3
                    "2,51,23,28,25.5," + // attr 4
                    "2,53,24,29,26.5," + // attr 5
                "0", // class

                // bag 3
                BAG3_SUMMARY_STRING + // region 0
                EMPTY_REGION_SUMMARY_STRING + // region 1
                BAG3_SUMMARY_STRING + // region 2
                EMPTY_REGION_SUMMARY_STRING + // region 3
                EMPTY_REGION_SUMMARY_STRING + // region 4
                //  region 5 (x2)
                    "2,85,40,45,42.5," + // attr 1
                    "2,87,41,46,43.5," + // attr 2
                    "2,89,42,47,44.5," + // attr 3
                    "2,91,43,48,45.5," + // attr 4
                    "2,93,44,49,46.5," + // attr 5
                //  region 6 (x2)
                    "2,105,50,55,52.5," + // attr 1
                    "2,107,51,56,53.5," + // attr 2
                    "2,109,52,57,54.5," + // attr 3
                    "2,111,53,58,55.5," + // attr 4
                    "2,113,54,59,56.5," + // attr 5
                EMPTY_REGION_SUMMARY_STRING + // region 7
                EMPTY_REGION_SUMMARY_STRING + // region 8
                        "1" // class
        };

        // expected attributes: (9=nodeCount*2+1) (* numberOfRegions *) x 5 (* num-summary-stats *) x NUM_ATTR
        //                      +1 for class
        final int expNumAttr = (numRegions * numSummaryStats * NUM_ATTR) + 1;

        final Instances expInstances = convertToDataset(expNumAttr, exp);
        final Instances actInstances = SplitNode.propositionaliseDataset(miData, root, propStrategy);
        assertDatasetEquals(expInstances, actInstances);
    }

    // </editor-fold>

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
     * @param numAttrInclClass The number of attributes in the propositionalised form, including class.
     * @param splitAttrIndex The attribute to split on.
     * @param splitPoint The value to split on.
     * @param exp The expected propositionalised dataset, as an array of csv-formatted strings.
     */
    private static void assertPropositionalisationOfOneNodeSplitTreeIs(
            final int splitAttrIndex, final double splitPoint, final String[] exp,
            final PropositionalisationStrategy propStrategy, final int numAttrInclClass)
    {
        // build the single-node split tree:
        final int attrPerRegion = propStrategy.getNumPropAttrPerRegion();
        final RootSplitNode root = RootSplitNode.toRootNode(
                new SplitNode(attrPerRegion, 2*attrPerRegion, splitAttrIndex, splitPoint, null, null, 3),
                propStrategy);
        root.setNodeCount(1);

        final Instances actual = SplitNode.propositionaliseDataset(miData, root, propStrategy);
        final Instances expected = convertToDataset(numAttrInclClass, exp);
        assertDatasetEquals(expected, actual);
    }

    /**
     * Check if the count-based propositionalisation of the example data-set
     *  using a single node split tree (splitting at the specified
     *  attribute and split point) is as specified in exp.
     *
     * @param splitAttrIndex The attribute to split on.
     * @param splitPoint The value to split on.
     * @param exp The expected propositionalised dataset, as an array of csv-formatted strings.
     */
    private static void assertCountBasedPropositionalisationOfOneNodeSplitTreeIs(
            final int splitAttrIndex, final double splitPoint, final String[] exp)
    {
        final CountBasedPropositionalisationStrategy propStrategy = new CountBasedPropositionalisationStrategy();
        final int numAttrInclClass = 4; // always 4 attributes for 1 split
        assertPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp, propStrategy, numAttrInclClass);
    }

    /**
     * Check if the summary-stat-based propositionalisation of the example data-set
     *  using a single node split tree (splitting at the specified
     *  attribute and split point) is as specified in exp.
     *
     * @param splitAttrIndex The attribute to split on.
     * @param splitPoint The value to split on.
     * @param exp The expected propositionalised dataset, as an array of csv-formatted strings.
     */
    private static void assertSummaryStatBasedPropositionalisationOfOneNodeSplitTreeIs(
            final int splitAttrIndex, final double splitPoint, final String[] exp)
    {
        final PropositionalisationStrategy propStrategy = new SummaryStatsBasedPropositionalisationStrategy(NUM_ATTR);

        // total number of attributes in propositionalised dataset:
        //      number of attributes of each instance in the mi dataset times number of summary stats * numRegions
        //      + 1 for class.
        final int numSummaryStats = SummaryStatsBasedPropositionalisationStrategy.SummaryStatCalculator.NUM_ATTR;
        final int numRegions = 3; // for 1 split, always 3 regions
        final int numAttrInclClass = (numSummaryStats * NUM_ATTR * numRegions) + 1;

        assertPropositionalisationOfOneNodeSplitTreeIs(splitAttrIndex, splitPoint, exp, propStrategy, numAttrInclClass);
    }

    // </editor-fold>
}
