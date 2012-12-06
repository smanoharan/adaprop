package weka.classifiers.mi;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import weka.classifiers.mi.adaprop.EvaluationStrategy;
import weka.classifiers.mi.adaprop.PropositionalisationStrategy;
import weka.classifiers.mi.adaprop.SearchStrategy;
import weka.classifiers.mi.adaprop.SplitStrategy;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Tag;

import java.util.*;

import static org.junit.Assert.*;

/**
 * Tests the option handling and other global methods
 *  of AdaProp.
 *
 *  @author Siva Manoharan
 */
public class AdaPropOptionsTest
{
    /** Instance, under test */
    protected AdaProp adaProp;

    @Before
    /** Create a new instance of AdaProp */
    public void setUp() throws Exception
    {
        this.adaProp = new AdaProp();
    }

    /* TODO Tests:
    *  - GetCapabilities
    *  - MultiInstanceGetCapabilities
    *
    *  - main? (e.g. with an artificial dataset)
    *  - buildClassifier
    *  - distributionForInstance
    */

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

    // <editor-fold desc="===Helper Functions===">

    // check that the string is not null and is not empty.
    protected static void assertNotNullOrEmpty(String toTest)
    {
        assertNotNull(toTest);
        assertTrue(toTest.length() > 0);
    }

    protected static void assertOptionEquals(
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

    private static void assertFlagStatusIs(String[] options, String key, boolean expected)
    {
        final List<String> optionList = Arrays.asList(options);
        final String msg = "Flag " + key + " in " + optionList + "?\n\t";
        assertEquals(msg, expected, optionList.contains(key));
    }

    private static void assertFlagIsSet(String[] options, String key)
    {
        assertFlagStatusIs(options, key, true);
    }

    private static void assertFlagIsNotSet(String[] options, String key)
    {
        assertFlagStatusIs(options, key, false);
    }

    private static void assertSelectedTagIs(int exp, SelectedTag tag, String key)
    {
        assertEquals("Value for option " + key, exp, tag.getSelectedTag().getID());
    }

    private void assertOptionIsListed(String key, String desc)
    {
        Option opt = findOption(adaProp.listOptions(), key);
        if (opt == null)
        {
            Assert.fail("Option -" + key + " not found");
        }
        else
        {
            assertOptionEquals(opt, "\t" + desc, 1, "-" + key + " <num>");
        }
    }

    private void assertFlagIsListed(String key, String desc)
    {
        Option opt = findOption(adaProp.listOptions(), key);
        if (opt == null)
        {
            Assert.fail("Option -" + key + " not found");
        }
        else
        {
            assertOptionEquals(opt, "\t" + desc, 0, "[-" + key + "]");
        }
    }

    private static void assertHasAllStrategies(String msg, ArrayList<String> act, String ... exp)
    {
        // check size is equal:
        assertEquals(msg + " size", exp.length, act.size());

        // convert exp to list
        List<String> expList = Arrays.asList(exp);

        // subset equality
        final String fullMsg = msg + "\n\tact: " + act.toString() + "\n\texp: " + expList.toString() + "\n\t";
        assertTrue(fullMsg + "act not subsetof exp", expList.containsAll(act));
        assertTrue(fullMsg + "exp not subsetof act", act.containsAll(expList));
    }
    // </editor-fold>

    // <editor-fold desc="===Split Strategy===">
    @Test
    public void testSplitPointOptionsAreListed() // in .listOptions();
    {
        assertOptionIsListed("split", "Split point criterion: 1=mean (default), 2=median, 3=discretized, 4=range");
    }

    @Test
    public void testGetAndSetSplitOptions() throws Exception
    {
        final String key = "-split";

        // by default: split strategy should be set to 1;
        int val = 1;
        assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
        assertSelectedTagIs(val, adaProp.getSplitStrategy(), key);

        // try setting it to all possible values & use get to verify
        for (val = 4; val >= 1; val--)
        {
            adaProp.setOptions(new String[]{key, Integer.toString(val)});
            assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
            assertSelectedTagIs(val, adaProp.getSplitStrategy(), key);
        }
    }

    @Test
    public void testSplitStrategyObjects()
    {
        ArrayList<String> strategies = new ArrayList<String>();
        for (Tag t : SplitStrategy.STRATEGIES)
        {
            strategies.add(SplitStrategy.getStrategy(t.getID(), 3).getClass().getSimpleName());
        }
        assertHasAllStrategies("Split Strategy", strategies,
                "MeanSplitStrategy",
                "MedianSplitStrategy",
                "DiscretizedSplitStrategy",
                "RangeSplitStrategy");
    }

    @Test
    public void shouldThrowExceptionForInvalidSplitStrategyCode()
    {
        try
        {
            SplitStrategy.getStrategy(999, 3);
            fail("Expected IllegalArgumentException");
        }
        catch (IllegalArgumentException iae)
        {
            assertEquals("Unknown split strategy code: 999", iae.getMessage());
        }
    }
    // </editor-fold>

    // <editor-fold desc="===Search Strategy===">
    @Test
    public void testSearchStrategyOptionsAreListed() // in .listOptions();
    {
        assertOptionIsListed("search", "Search strategy: 1=breadth-first (default), 2=best-first");
    }

    @Test
    public void testGetAndSetSearchOptions() throws Exception
    {
        final String key = "-search";

        // by default: search strategy should be set to 1;
        int val = 1;
        assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
        assertSelectedTagIs(val, adaProp.getSearchStrategy(), key);

        // try setting it to all possible values & use get to verify
        for (val = 2; val >= 1; val--)
        {
            adaProp.setOptions(new String[]{key, Integer.toString(val)});
            assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
            assertSelectedTagIs(val, adaProp.getSearchStrategy(), key);
        }
    }

    @Test
    public void testSearchStrategyObjects()
    {
        ArrayList<String> strategies = new ArrayList<String>();
        for (Tag t : SearchStrategy.STRATEGIES)
        {
            strategies.add(SearchStrategy.getStrategy(t.getID()).getClass().getSimpleName());
        }
        assertHasAllStrategies("Search Strategy", strategies,
                "BestFirstSearchStrategy",
                "BreadthFirstSearchStrategy");
    }

    @Test
    public void shouldThrowExceptionForInvalidSearchStrategyCode()
    {
        try
        {
            SearchStrategy.getStrategy(999);
            fail("Expected IllegalArgumentException");
        }
        catch (IllegalArgumentException iae)
        {
            assertEquals("Unknown search strategy code: 999", iae.getMessage());
        }
    }
    // </editor-fold>

    // <editor-fold desc="===Propositionalisation Strategy===">
    @Test
    public void testPropStrategyOptionsAreListed() // in .listOptions();
    {
        assertOptionIsListed("prop", "Propositionalisation strategy: 1=count-only (default), 2=all-summary-stats");
    }

    @Test
    public void testGetAndSetPropOptions() throws Exception
    {
        final String key = "-prop";

        // by default: prop strategy should be set to 1;
        int val = 1;
        assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
        assertSelectedTagIs(val, adaProp.getPropositionalisationStrategy(), key);

        // try setting it to all possible values & use get to verify
        for (val = 2; val >= 1; val--)
        {
            adaProp.setOptions(new String[]{key, Integer.toString(val)});
            assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
            assertSelectedTagIs(val, adaProp.getPropositionalisationStrategy(), key);
        }
    }

    @Test
    public void testPropStrategyObjects()
    {
        ArrayList<String> strategies = new ArrayList<String>();
        for (Tag t : PropositionalisationStrategy.STRATEGIES)
        {
            strategies.add(PropositionalisationStrategy.getStrategy(t.getID(), 3).getClass().getSimpleName());
        }
        assertHasAllStrategies("Prop Strategy", strategies,
                "CountBasedPropositionalisationStrategy",
                "SummaryStatsBasedPropositionalisationStrategy");
    }

    @Test
    public void shouldThrowExceptionForInvalidPropStrategyCode()
    {
        try
        {
            PropositionalisationStrategy.getStrategy(999, 3);
            fail("Expected IllegalArgumentException");
        }
        catch (IllegalArgumentException iae)
        {
            assertEquals("Unknown propositionalisation strategy code: 999", iae.getMessage());
        }
    }
    // </editor-fold>

    // <editor-fold desc="===Evaluation Strategy===">
    @Test
    public void testEvalStrategyOptionsAreListed() // in .listOptions();
    {
        assertOptionIsListed("eval", "Split Evaluation strategy: " +
                "1=mis-classification error (default), " +
                "2=cross-validated mis-classification error, " +
                "3=root mean squared error, " +
                "4=cross-validated root mean squared error, " +
                "5=gain ratio, " +
                "6=cross-validated gain ratio");
    }

    @Test
    public void testGetAndSetEvalOptions() throws Exception
    {
        final String key = "-eval";

        // by default: prop strategy should be set to 1;
        int val = 1;
        assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
        assertSelectedTagIs(val, adaProp.getPropositionalisationStrategy(), key);

        // try setting it to all possible values & use get to verify
        for (val = 6; val >= 1; val--)
        {
            adaProp.setOptions(new String[]{key, Integer.toString(val)});
            assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
            assertSelectedTagIs(val, adaProp.getEvalStrategy(), key);
        }
    }

    @Test
    public void testEvalStrategyObjects()
    {
        ArrayList<String> strategies = new ArrayList<String>();
        for (Tag t : EvaluationStrategy.STRATEGIES)
        {
            strategies.add(EvaluationStrategy.getStrategy(t.getID(), new Random()).getClass().getSimpleName());
        }
        assertHasAllStrategies("Eval Strategy", strategies,
                "MisClassificationErrorEvaluationStrategy",
                "MisClassificationCrossValidatedErrorEvaluationStrategy",
                "RMSEEvaluationStrategy",
                "RMSECrossValidatedErrorEvaluationStrategy",
                "InfoGainEvaluationStrategy",
                "InfoGainCrossValidatedErrorEvaluationStrategy");
    }

    @Test
    public void shouldThrowExceptionForInvalidEvalStrategyCode()
    {
        try
        {
            EvaluationStrategy.getStrategy(999, new Random());
            fail("Expected IllegalArgumentException");
        }
        catch (IllegalArgumentException iae)
        {
            assertEquals("Unknown evaluation strategy code: 999", iae.getMessage());
        }
    }
    // </editor-fold>

    // <editor-fold desc="===Max Tree Size===">

    @Test
    public void testMaxTreeSizeOptionsAreListed() // in .listOptions();
    {
        assertOptionIsListed("maxTreeSize", "Maximum size (number of nodes) of the tree. Default=8.");
    }

    @Test
    public void testGetAndSetMaxTreeSizeOptions() throws Exception
    {
        final String key = "-maxTreeSize";
        final String message = "Value for " + key;

        // by default: max tree size should be set to 8:
        int val = 8;
        assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
        assertEquals(message, val, adaProp.getMaxTreeSize());

        // try setting it to some possible values & use get to verify
        for (val = 16; val >= 1; val--)
        {
            adaProp.setOptions(new String[]{key, Integer.toString(val)});
            assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
            assertEquals(message, val, adaProp.getMaxTreeSize());
        }
    }
    // </editor-fold>

    // <editor-fold desc="===Min Occupancy===">
    @Test
    public void testMinOccupancyOptionsAreListed() // in .listOptions();
    {
        assertOptionIsListed("minOcc", "Minimum occupancy of each node of the tree. Default=5.");
    }

    @Test
    public void testGetAndSetMinOccupancyOptions() throws Exception
    {
        final String key = "-minOcc";
        final String message = "Value for " + key;

        // by default: min occupancy should be set to 5:
        int val = 5;
        assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
        assertEquals(message, val, adaProp.getMinOccupancy());

        // try setting it to some possible values & use get to verify
        for (val = 16; val >= 1; val--)
        {
            adaProp.setOptions(new String[]{key, Integer.toString(val)});
            assertOptionValueEquals(adaProp.getOptions(), key, Integer.toString(val));
            assertEquals(message, val, adaProp.getMinOccupancy());
        }
    }
    // </editor-fold>

    // <editor-fold desc="===Parameter Selection===">
    @Test
    public void testDoParameterSelectionOptionsAreListed()
    {
        assertFlagIsListed("paramSel", "Perform Cross-validated Tree Size Parameter Selection. " + "Default=False.");
    }

    @Test
    public void testGetAndSetDoParameterSelectionOptions() throws Exception
    {
        final String key = "-paramSel";

        // by default: no parameter selection:
        assertFlagIsNotSet(adaProp.getOptions(), key);
        assertFalse("Flag " + key + "should not be set", adaProp.getDoCVParameterSelection());

        // try setting it to some possible values & use get to verify
        adaProp.setOptions(new String[]{key});
        assertFlagIsSet(adaProp.getOptions(), key);
        assertTrue("Flag " + key + "should be set", adaProp.getDoCVParameterSelection());

        adaProp.setOptions(new String[]{});
        assertFlagIsNotSet(adaProp.getOptions(), key);
        assertFalse("Flag " + key + "should not be set", adaProp.getDoCVParameterSelection());

    }
    // </editor-fold>

    // TODO remove or move below tests to another file:

    /** Test evaluation of with the specified classifier gives the correct value */
//    private void evalSplitWithClassifier(Classifier classifier, double exp)
//    {
//        try
//        {
//            // init the m_classifier
//            adaProp.setClassifier(classifier);
//
//            // find actual split:
//            final int attrIndex = 2;
//            final BitSet ignore = new BitSet(NUM_BAGS * NUM_INST_PER_BAG);
//            final double splitPt = new MeanSplitStrategy(NUM_ATTR).findCenter(miData, attrIndex, ignore);
//            RootSplitNode root = createRootSplit(attrIndex, splitPt);
//
//            final double act = SplitNode.evaluateCurSplit(miData, classifier, root, new CountBasedPropositionalisationStrategy());
//            assertEquals(classifier.getClass().getName(), exp, act, TOLERANCE);
//        }
//        catch (Exception e) { throw new RuntimeException(e); }
//    }

//    @Test
//    public void testEvalSplitWithZeroR() throws Exception
//    {
//        evalSplitWithClassifier(new ZeroR(), 1);
//    }
//
//    @Test
//    public void testEvalSplitWithOneR() throws Exception
//    {
//        evalSplitWithClassifier(new OneR(), 1);
//    }
//
//    @Test
//    public void testEvalSplitWithJ48() throws Exception
//    {
//        J48 j48 = new J48();
//        j48.setMinNumObj(1);
//        evalSplitWithClassifier(j48, 0);
//    }
}

