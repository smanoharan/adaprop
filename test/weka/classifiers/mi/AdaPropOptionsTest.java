package weka.classifiers.mi;

import org.junit.Assert;
import org.junit.Test;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Option;

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
    // TODO currently incomplete.

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
        assertOptionValueEquals(adaProp.getOptions(), "-split", "1");

        // try setting it to 2 (median) & use get to verify
        adaProp.setOptions(new String[]{"-split", "2"});
        assertOptionValueEquals(adaProp.getOptions(), "-split", "2");

        // try setting it to 3 (discretize) & use get to verify
        adaProp.setOptions(new String[]{"-split", "3"});
        assertOptionValueEquals(adaProp.getOptions(), "-split", "3");

        // try setting it to 4 (range) & use get to verify
        adaProp.setOptions(new String[]{"-split", "4"});
        assertOptionValueEquals(adaProp.getOptions(), "-split", "4");
    }

    @Test
    public void testGetAndSetSearchOptions() throws Exception
    {
        // by default: Search strategy should be set to 1;
        assertOptionValueEquals(adaProp.getOptions(), "-search", "1");

        // try setting it to 2 (best-first) & use get to verify
        adaProp.setOptions(new String[]{"-search", "2"});
        assertOptionValueEquals(adaProp.getOptions(), "-search", "2");

        // try setting it back to 1 (breadth-first) & use get to verify
        adaProp.setOptions(new String[]{"-search", "1"});
        assertOptionValueEquals(adaProp.getOptions(), "-search", "1");
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

            final double act = SplitNode.evaluateCurSplit(miData, classifier, root, new CountBasedPropositionalisationStrategy());
            assertEquals(classifier.getClass().getName(), exp, act, TOLERANCE);
        }
        catch (Exception e) { throw new RuntimeException(e); }
    }

    @Test
    public void testEvalSplitWithZeroR() throws Exception
    {
        // TODO evalSplitWithClassifier(new ZeroR(), 1);
    }

    @Test
    public void testEvalSplitWithOneR() throws Exception
    {
        // TODO evalSplitWithClassifier(new OneR(), 1);
    }

    @Test
    public void testEvalSplitWithJ48() throws Exception
    {
        J48 j48 = new J48();
        j48.setMinNumObj(1);
        // TODO evalSplitWithClassifier(j48, 0);
    }
}

