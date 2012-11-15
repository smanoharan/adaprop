package weka.classifiers.mi;

import org.junit.Assert;
import org.junit.Test;
import weka.classifiers.Classifier;
import weka.core.Option;
import weka.core.SelectedTag;

import java.util.BitSet;
import java.util.Enumeration;

import static org.junit.Assert.assertEquals;

/**
 * Tests the option handling and other global methods
 *  of AdaProp.
 *
 *  @author Siva Manoharan
 */
public class AdaPropOptionsTest extends AdaPropTestBase
{
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

    private static void assertSelectedTagIs(int exp, SelectedTag tag, String key)
    {
        assertEquals("Value for option " + key, exp, tag.getSelectedTag().getID());
    }
    // </editor-fold>

    // <editor-fold desc="===Split Strategy===">
    @Test
    public void testSplitPointOptionsAreListed() // in .listOptions();
    {
        Option opt = findOption(adaProp.listOptions(), "split");
        if (opt == null)
        {
            Assert.fail("Option -split (split point) not found");
        }
        else
        {
            assertOptionEquals(opt,
                    "\tSplit point criterion: 1=mean (default), 2=median, 3=discretized, 4=range",
                    1, "-split <num>");
        }
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

    // </editor-fold>

    // <editor-fold desc="===Search Strategy===">

    @Test
    public void testSearchStrategyOptionsAreListed() // in .listOptions();
    {
        Option opt = findOption(adaProp.listOptions(), "search");
        if (opt == null)
        {
            Assert.fail("Option -search (search strategy) not found");
        }
        else
        {
            assertOptionEquals(opt,
                    "\tSearch strategy: 1=breadth-first (default), 2=best-first",
                    1, "-search <num>");
        }
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

    // </editor-fold>

    // <editor-fold desc="===Propositionalisation Strategy===">

    @Test
    public void testPropStrategyOptionsAreListed() // in .listOptions();
    {
        Option opt = findOption(adaProp.listOptions(), "prop");
        if (opt == null)
        {
            Assert.fail("Option -prop (propositionalisation strategy) not found");
        }
        else
        {
            assertOptionEquals(opt,
                    "\tPropositionalisation strategy: 1=count-only (default), 2=all-summary-stats",
                    1, "-prop <num>");
        }
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

    // </editor-fold>

    // <editor-fold desc="===Max Tree Size===">

    @Test
    public void testMaxTreeSizeOptionsAreListed() // in .listOptions();
    {
        Option opt = findOption(adaProp.listOptions(), "maxTreeSize");
        if (opt == null)
        {
            Assert.fail("Option -maxTreeSize not found");
        }
        else
        {
            assertOptionEquals(opt,
                    "\tMaximum size (number of nodes) of the tree. Default=8.",
                    1, "-maxTreeSize <num>");
        }
    }

    @Test
    public void testGetAndSetMaxTreeSizeOptions() throws Exception
    {
        final String key = "-maxTreeSize";
        final String message = "Value for -" + key;

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
        Option opt = findOption(adaProp.listOptions(), "minOcc");
        if (opt == null)
        {
            Assert.fail("Option -minOcc not found");
        }
        else
        {
            assertOptionEquals(opt,
                    "\tMinimum occupancy of each node of the tree. Default=5.",
                    1, "-minOcc <num>");
        }
    }

    @Test
    public void testGetAndSetMinOccupancyOptions() throws Exception
    {
        final String key = "-minOcc";
        final String message = "Value for -" + key;

        // by default: max tree size should be set to 5:
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


    // TODO remove or move below tests to another file:

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

            final double act = SplitNode.evaluateCurSplit(miData, classifier, root, new CountBasedPropositionalisationStrategy());
            assertEquals(classifier.getClass().getName(), exp, act, TOLERANCE);
        }
        catch (Exception e) { throw new RuntimeException(e); }
    }

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

