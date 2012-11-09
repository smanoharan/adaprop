package weka.classifiers.mi;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Test the split strategies. TODO needs to be more extensive.
 */
public class AdaPropSplitStrategyTest extends AdaPropTestBase
{
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
            final List<Double> exp = Arrays.asList(37.5 + attrIndex);
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

    // Test the splitting when invoked via instance methods:
    private static void assertSplitPtListEquals(SplitStrategy strategy, List<CompPair<Integer, Double>> exp, String msg)
    {
        List<CompPair<Integer, Double>> act = strategy.generateSplitPoints(miData, new BitSet(NUM_INST_PER_BAG*NUM_BAGS));
        assertPairListEquals(msg, exp, act);
    }


    // can be used when there is a unique split point for each attr index
    private static List<CompPair<Integer, Double>> arrayToPairList(double ... splitPts)
    {
        List<CompPair<Integer, Double>> list = new ArrayList<CompPair<Integer, Double>>(splitPts.length);
        for (int i=0; i<splitPts.length; i++)
        {
            list.add(new CompPair<Integer, Double>(i, splitPts[i]));
        }
        return list;
    }
}
