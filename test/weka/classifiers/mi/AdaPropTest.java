package weka.classifiers.mi;

import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import weka.classifiers.Classifier;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.*;

import java.util.*;

import static org.junit.Assert.*;

/** Author: Siva Manoharan, 1117707 */
public class AdaPropTest
{
    public static final int NUM_ATTR = 5; // number of attr in si-dataset
    public static final int NUM_BAGS = 3; // number of mi bags
    public static final int NUM_INST_PER_BAG = 4;
    public static final int REL_INDEX = AdaProp.REL_INDEX;

    /** Instance, under test */
    private AdaProp adaProp;

    /** Contains the mi bags for testing */
    protected static Instances miData;

    /** Header for the single-instance relation */
    protected static Instances siHeader;

    /** Header for the resultant propositionalised dataset */
    protected static Instances propHeader;

    /** For comparing doubles */
    static final double TOLERANCE = 0.000001;

    /* TODO Tests:
    *  - GetCapabilities
    *  - MultiInstanceGetCapabilities
    *  - ListOptions
    *  - SetOptions
    *  - GetOptions
    *
    *  - main? (e.g. with an artificial dataset)
    *  - buildClassifier?
    *  - distributionForInstance
    */

    @BeforeClass
    /** Setup the instance headers */
    public static void setupClass() throws Exception
    {
        setupSingleInstanceHeader();
        setupMultiInstanceData();
        setupPropositionalisedHeader();
    }

    /** Setup the dataset for the propositionalised instances */
    private static void setupPropositionalisedHeader()
    {
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
        attInfo.add(new Attribute("total"));
        attInfo.add(new Attribute("less-than"));
        attInfo.add(new Attribute("greater-than"));
        attInfo.add((Attribute) miData.classAttribute().copy());
        propHeader = new Instances("prop-header", attInfo, 0);
        propHeader.setClassIndex(3);
    }

    /** Initialise the siHeader */
    private static void setupSingleInstanceHeader()
    {
        // si-header has 5 numeric attributes + 1 numeric class.
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();

        for(int i=0; i< NUM_ATTR; i++)
        {
            attInfo.add(new Attribute("attr" + i));
        }

        attInfo.add(new Attribute("class"));

        siHeader = new Instances("si-header", attInfo, 0);
        siHeader.setClassIndex(NUM_ATTR);
    }

    // initialise numInst new single-instances and return the set.
    private static Instances newRelation(final int startVal, final int numInst)
    {
        // initialise relation as a copy of siHeader
        Instances relation = new Instances(siHeader, numInst);

        int val = startVal;
        for (int instIndex = 0; instIndex < numInst; instIndex++)
        {
            Instance inst = new DenseInstance(NUM_ATTR);

            for (int attrIndex = 0; attrIndex < NUM_ATTR; attrIndex++)
            {
                // increase value in a predictable pattern
                inst.setValue(attrIndex, val++);
            }

            // add inst to dataset
            inst.setDataset(relation);
            relation.add(inst);
        }

        return relation;
    }

    /** Create the attribute corresponding to the attribute field */
    private static Attribute createRelationAttribute()
    {
        Attribute attr = new Attribute("bag", siHeader);
        for (int bagIndex = 0; bagIndex < NUM_BAGS; bagIndex++)
        {
            final int startVal = bagIndex*NUM_ATTR*NUM_INST_PER_BAG;
            attr.addRelation(newRelation(startVal, NUM_INST_PER_BAG));
        }
        return attr;
    }

    protected static Attribute createClassAttribute()
    {
        final List<String> attributeValues = new ArrayList<String>(2);
        attributeValues.add("class-0");
        attributeValues.add("class-1");
        return new Attribute("class", attributeValues);
    }

    /** Initialise the miData, assuming siHeader has been init'd */
    private static void setupMultiInstanceData()
    {
        // this has 3 attr: bag-id (numeric), bag (relational), class (numeric)
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
        attInfo.add(new Attribute("bag-id"));
        attInfo.add(createRelationAttribute());
        attInfo.add(createClassAttribute());

        miData = new Instances("mi-header", attInfo, 0);
        miData.setClassIndex(2);

        // populate mi-header
        for (int bagIndex = 0; bagIndex < NUM_BAGS; bagIndex++)
        {
            Instance bag = new DenseInstance(3);
            bag.setValue(0, bagIndex);
            bag.setValue(1, bagIndex);
            int bagClass = bagIndex < 2 ? 0 : 1;
            bag.setValue(2, bagClass);

            // add bag to dataset
            bag.setDataset(miData);
            miData.add(bag);
        }
    }

    @Before
    /** Create a new instance of AdaProp */
    public void setUp() throws Exception
    {
        this.adaProp = new AdaProp();
    }

    // check that the string is not null and is not empty.
    private static void assertNotNullOrEmpty(String toTest)
    {
        assertNotNull(toTest);
        assertTrue(toTest.length() > 0);
    }

    @Test
    public void testGlobalInfoIsNotNullOrEmpty()
    {
        assertNotNullOrEmpty(adaProp.globalInfo());
    }

    @Test
    public void testToStringIsNotNullOrEmpty() throws Exception
    {
        assertNotNullOrEmpty(adaProp.toString());
    }

    // ==================================================================================
    //  Tests for option handling
    // ==================================================================================

    private static void assertOptionEquals(
            final Option actual, final String expDesc,
            final int expNumArgs, final String expSynopsis)
    {
        assertEquals(actual.name() + " Desc: ",     expDesc,        actual.description());
        assertEquals(actual.name() + " NumArgs: ",  expNumArgs,     actual.numArguments());
        assertEquals(actual.name() + " Synopsis: ", expSynopsis,    actual.synopsis());
    }

    // find the option corresponding to the name in the enumeration
    private static Option findOption(final Enumeration opts, final String key)
    {
        while (opts.hasMoreElements())
        {
            Option opt = (Option) opts.nextElement();
            if (opt.name().equals(key))
            {
                return opt;
            }
        }
        return null;
    }

    @Test
    public void testSplitPointOptionsAreListed() // in .listOptions();
    {
        // split point is chosen by '-s'. Try to find it:
        Option opt = findOption(adaProp.listOptions(), "S");
        if (opt == null)
        {
            Assert.fail("Option -S (split point) not found");
        }
        else
        {
            assertOptionEquals(opt,
                "\tSplit point criterion: 1=mean (default), 2=median, 3=discretized, 4=range",
                1, "-S <num>");
        }
    }

    private static void assertOptionValueEquals(
            String[] options, String key, String expVal)
    {
        // find the key
        for (int i=0; i<options.length; i++)
        {
            if (options[i].equals(key))
            {
                assertEquals("Value for option " + key, expVal, options[i+1]);
                return;
            }
        }
        Assert.fail("Option " + key + " not found in getOptions");
    }

    @Test
    public void testGetAndSetSplitOptions() throws Exception
    {
        // by default: split point should be set to 1;
        assertOptionValueEquals(adaProp.getOptions(), "-S", "1");

        // try setting it to 2 (median) & use get to verify
        adaProp.setOptions(new String[]{"-S", "2"});
        assertOptionValueEquals(adaProp.getOptions(), "-S", "2");

        // try setting it to 3 (discretize) & use get to verify
        adaProp.setOptions(new String[]{"-S", "3"});
        assertOptionValueEquals(adaProp.getOptions(), "-S", "3");

        // try setting it to 4 (range) & use get to verify
        adaProp.setOptions(new String[]{"-S", "4"});
        assertOptionValueEquals(adaProp.getOptions(), "-S", "4");
    }

    // ==================================================================================

    @Test
    public void testFindMeanViaInstance() throws Exception
    {
        double[] splits = new double[NUM_ATTR];
        final int numInst = NUM_INST_PER_BAG * NUM_BAGS;
        for (int i = 0; i < NUM_ATTR; i++)
        {
            splits[i] = i + (NUM_ATTR*(numInst-1)/2.0);
        }
        assertSplitPtListEquals(new MeanSplitStrategy(NUM_ATTR), arrayToPairList(splits), "mean");
    }

    @Test
    public void testFindMeanViaStatic() throws Exception
    {
        // actual values for mean:
        //      there are 12 instances, 5 attributes
        //      the values are the natural numbers in sequence.
        //      for example, inst1 = {0,1,2,3,4} ; inst2 = {5,6,7,8,9} etc.
        //      so the mean(attr-i)=(\sum_{j=0}^{num_inst-1} (i+j*num_attr))/12
        // in fact, sum(attr-i)
        //      = num_inst*i + num_attr*(\sum_{j=0}^{num_inst-1}(j))
        //      = num_inst*i + num_attr*(1+2+3+...+num_inst-1)
        //      = num_inst*i + num_attr*((num_inst * num_inst-1) / 2)
        // thus, mean(attr-i) = i + num_attr*(num_inst-1)/2

        // for each attribute:
        final int numInst = NUM_INST_PER_BAG * NUM_BAGS;
        for (int attrIndex = 0; attrIndex < NUM_ATTR; attrIndex++)
        {
            final double expectedMean = attrIndex + (NUM_ATTR*(numInst-1)/2.0);
            final String msg = "Mean for attribute " + attrIndex;

            final double actual = MeanSplitStrategy.findMean(miData, attrIndex, new BitSet(numInst));
            assertEquals(msg, expectedMean, actual, TOLERANCE);
        }
    }

    @Test
    public void testFindMedianViaInstance() throws Exception
    {
        double[] splits = new double[NUM_ATTR];
        final int numInst = NUM_INST_PER_BAG * NUM_BAGS;
        for (int i = 0; i < NUM_ATTR; i++)
        {
            splits[i] = i + (NUM_ATTR*(numInst-1)/2.0);
        }
        assertSplitPtListEquals(new MedianSplitStrategy(NUM_ATTR), arrayToPairList(splits), "median");
    }

    @Test
    public void testFindMedianViaStatic() throws Exception
    {
        // there are 12 instances, with values in increasing order
        // median of attribute i is the average of the 6th and 7th bags
        // e.g. for attr=0: 0, 5, ..., 25, 30, ...
        //      for attr=1: 1, 6, ..., 26, 31, ...
        final int numInst = NUM_INST_PER_BAG * NUM_BAGS;
        for (int attrIndex = 0; attrIndex < NUM_ATTR; attrIndex++)
        {
            final double expectedMedian = 27.5 + attrIndex;
            final String msg = "Median for attribute " + attrIndex;
            final double actual = MedianSplitStrategy.findMedian(miData, attrIndex, new BitSet(numInst));
            assertEquals(msg, expectedMedian, actual, TOLERANCE);
        }
    }

    @Test
    public void testFindDiscretizedViaInstance() throws Exception
    {
        double[] splits = new double[NUM_ATTR];
        for (int i = 0; i < NUM_ATTR; i++)
        {
            splits[i] = i + 37.5;
        }
        assertSplitPtListEquals(new DiscretizedSplitStrategy(NUM_ATTR), arrayToPairList(splits), "discretized");
    }

    @Test
    public void testFindDiscretizeSplitPoints() throws Exception
    {
        // there are 12 instances, with values in increasing order
        // discretized split points are when class changes (8-9)
        // e.g. for attr=0: 0, ..., 35, 40, ...
        //      for attr=0: 1, ..., 36, 41, ...
        final int numInst = NUM_INST_PER_BAG * NUM_BAGS;
        for (int attrIndex = 0; attrIndex < NUM_ATTR; attrIndex++)
        {
            final List<Double> exp = Arrays.asList(37.5+attrIndex);
            final String msg = "Split points for attribute " + attrIndex;

            final ArrayList<Double> act = DiscretizedSplitStrategy.findDiscretizedSplits(miData, attrIndex, new BitSet(numInst));
            assertListEquals(msg, exp, act);
        }
    }

    @Test
    public void testFindRangeViaInstance() throws Exception
    {
        double[] splits = new double[NUM_ATTR];
        final int numInst = NUM_INST_PER_BAG * NUM_BAGS;
        for (int i = 0; i < NUM_ATTR; i++)
        {
            splits[i] = i + (NUM_ATTR*(numInst-1)/2.0);
        }
        assertSplitPtListEquals(new RangeSplitStrategy(NUM_ATTR), arrayToPairList(splits), "range");
    }

    @Test
    public void testFindRange() throws Exception
    {
        // actual values for mean:
        //      there are 12 instances, 5 attributes
        //      the values are the natural numbers in sequence.
        //      for example, inst1 = {0,1,2,3,4} ; inst2 = {5,6,7,8,9} ; .. ; inst12 = {55,56,57,58,59} ;
        //      so the midpt is {27.5, 28.5, ... }

        // for each attribute:
        final int numInst = NUM_INST_PER_BAG * NUM_BAGS;
        for (int attrIndex = 0; attrIndex < NUM_ATTR; attrIndex++)
        {
            final double expected = attrIndex + (NUM_ATTR*(numInst-1)/2.0);
            final String msg = "Range-MidPt for attribute " + attrIndex;
            final double actualViaStatic = RangeSplitStrategy.findMidpt(miData, attrIndex, new BitSet(numInst));
            assertEquals(msg, expected, actualViaStatic, TOLERANCE);
        }
    }

    // can be used when there is a unique split point for each attr index
    private static List<Pair<Integer, Double>> arrayToPairList(double ... splitPts)
    {
        List<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>(splitPts.length);
        for (int i=0; i<splitPts.length; i++)
        {
            list.add(new Pair<Integer, Double>(i, splitPts[i]));
        }
        return list;
    }

    // Test the splitting when invoked via instance methods:
    private static void assertSplitPtListEquals(SplitStrategy strategy, List<Pair<Integer, Double>> exp, String msg)
    {
        List<Pair<Integer, Double>> act = strategy.generateSplitPoints(miData, new BitSet(NUM_INST_PER_BAG*NUM_BAGS));
        assertPairListEquals(msg, exp, act);
    }

    private static void assertListEquals(String msg, List<Double> exp, List<Double> act)
    {
        // check sizes are equal
        assertEquals(msg + " size", exp.size(), act.size());

        // check each elem
        for (int i=0;i<exp.size();i++)
        {
            assertEquals(msg + " index " + i, exp.get(i), act.get(i), TOLERANCE);
        }
    }

    private static void assertPairListEquals(String msg, List<Pair<Integer, Double>> exp, List<Pair<Integer, Double>> act)
    {
        // check sizes are equal
        assertEquals(msg + " size", exp.size(), act.size());

        // check each elem
        for (int i=0;i<exp.size();i++)
        {
            String msgSuffix = exp.get(i).toString() + " != " + act.get(i).toString();
            assertEquals(msg + " (key) index " + i + msgSuffix, exp.get(i).key, act.get(i).key, TOLERANCE);
            assertEquals(msg + " (val) index " + i + msgSuffix, exp.get(i).value, act.get(i).value, TOLERANCE);
        }
    }

    private static RootSplitNode createRootSplit(final int attrIndex, final double splitPt)
    {
        SplitNode leftLeaf = SplitNode.newLeafNode(1,1);
        SplitNode rightLeaf = SplitNode.newLeafNode(2,1);
        RootSplitNode root = new RootSplitNode(0, attrIndex, splitPt, leftLeaf, rightLeaf, 0);
        root.setNodeCount(3);
        return root;
    }

    /** Test evaluation of with the specified classifier gives the correct value */
    private void evalSplitWithClassifier(Classifier classifier, double exp)
    {
        try
        {
            // init the m_classifier
            adaProp.setClassifier(classifier);

            // find actual split:
            final int attrIndex = 2;
            final BitSet ignore = new BitSet(NUM_BAGS * NUM_INST_PER_BAG);
            final double splitPt = new MeanSplitStrategy(NUM_ATTR).findCenter(miData, attrIndex, ignore);
            RootSplitNode root = createRootSplit(attrIndex, splitPt);

            final double act = SplitNode.evaluateCurSplit(miData, classifier, root);
            assertEquals(classifier.getClass().getName(), exp, act, TOLERANCE);
        }
        catch (Exception e) { throw new RuntimeException(e); }
    }

    @Test
    public void testEvalSplitWithZeroR() throws Exception
    {
        evalSplitWithClassifier(new ZeroR(), 1);
    }

    @Test
    public void testEvalSplitWithOneR() throws Exception
    {
        evalSplitWithClassifier(new OneR(), 1);
    }

    @Test
    public void testEvalSplitWithJ48() throws Exception
    {
        J48 j48 = new J48();
        j48.setMinNumObj(1);
        evalSplitWithClassifier(j48, 0);
    }

    /**
     * Assuming that exp and act have the same format,
     *  check that they have the same values for all attributes.
     *
     * @param exp Expected
     * @param act Actual
     */
    private static void assertInstanceEquals(String msg, Instance exp, Instance act)
    {
        // check number of attributes is equal
        int numAttr = act.numAttributes();
        assertEquals("Number of attributes", exp.numAttributes(), numAttr);

        // for each attribute
        for (int attrIndex = 0; attrIndex < numAttr; attrIndex++)
        {
            String msgAttr = ", attribute: " + attrIndex;
            assertEquals(msg + msgAttr, exp.value(attrIndex), act.value(attrIndex), TOLERANCE);
        }
    }

    /**
     * Assuming that exp and act have the same format,
     *  check that their instances have the same values
     *
     * @param exp Expected
     * @param act Actual
     */
    protected static void assertDatasetEquals(Instances exp, Instances act)
    {
        // check num of instances is the same
        int actNumInst = act.numInstances();
        assertEquals("Number of instances \n\n" + exp.toString() + "\n\n" + act.toString() + "\n\n",
                exp.numInstances(), actNumInst);

        // check each instance is equal
        for (int instIndex = 0; instIndex < actNumInst; instIndex++)
        {
            final Instance expInst = exp.get(instIndex);
            final Instance actInst = act.get(instIndex);
            final String msg = "Instance: [" + instIndex + "] -- " + expInst.toString() + " -- " + actInst.toString();

            assertInstanceEquals(msg, expInst, actInst);
        }
    }

    private static Instances actualPropositionalisedBagFor(
            final int attrIndex, final double split) throws Exception
    {
        final Instances result = new Instances(propHeader, NUM_BAGS);
        final RootSplitNode root = createRootSplit(attrIndex, split);

        // for each bag
        for (int bagIndex = 0; bagIndex < NUM_BAGS; bagIndex++)
        {
            final Instance bag = miData.get(bagIndex);

            // find actual value:
            result.add(SplitNode.propositionaliseBag(bag, root, result));
        }

        return result;
    }

    // check one case of propositionaliseBag
    private static Instances expectedPropositionalisedBagFor(
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


    @Test /** Check just one hard-coded case for propositionaliseBag */
    public void testPropositionaliseBagForASingleCase() throws Exception
    {
        // split on the 3rd attribute,
        //  which takes values: {2, 7, 12, 17, 22, 27, ... 57}
        // use split pt 25; so first 5 instances are to the left.

        final int attrIndex = 2;
        final double splitPt = 25;
        final int instToLeft = 5;

        Instances expected = expectedPropositionalisedBagFor(instToLeft);
        Instances actual = actualPropositionalisedBagFor(attrIndex, splitPt);

        assertDatasetEquals(expected, actual);
    }

    @Test /** Check all cases for propositionaliseBag */
    public void testPropositionaliseBagForAllAttributes() throws Exception
    {
        final int numInst = NUM_INST_PER_BAG*NUM_BAGS;

        // for each attribute
        for (int attrIndex = 0; attrIndex < NUM_ATTR; attrIndex++)
        {
            for (int instToLeft = 0; instToLeft <= numInst; instToLeft++)
            {
                double splitPt = instToLeft*NUM_ATTR + attrIndex - 0.5;
                Instances expected = expectedPropositionalisedBagFor(instToLeft);
                Instances actual = actualPropositionalisedBagFor(attrIndex, splitPt);

                assertDatasetEquals(expected, actual);
            }
        }
    }
}

