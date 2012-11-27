package weka.classifiers.mi.adaprop;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Tag;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;

/**
 * A strategy for generating candidate splits
 */
public abstract class SplitStrategy implements Serializable
{
    /**
     * Generate all candidate splits using the current split strategy
     * @param trainingData The training data (as bags)
     * @param ignore The bitSet of instances to ignore.
     * @return A list of candidate splits
     */
    public abstract List<CompPair<Integer, Double>> generateSplitPoints(final Instances trainingData,
                                                                 final BitSet ignore);

    public boolean canExpand(Instances dataset, BitSet ignoreMask)
    {
        return true;
    }

    // <editor-fold desc="===Option Handling===">
    private static final int SPLIT_MEAN = 1;
    private static final int SPLIT_MEDIAN = 2;
    private static final int SPLIT_DISCRETIZED = 3;
    private static final int SPLIT_RANGE = 4;
    public static final int DEFAULT_STRATEGY = SPLIT_MEAN;
    public static final String DESCRIPTION =
            "Split point criterion: 1=mean (default), 2=median, 3=discretized, 4=range";

    public static final Tag[] STRATEGIES =
    {
        new Tag(SPLIT_MEAN, "Split by the mean value of an attribute"),
        new Tag(SPLIT_MEDIAN, "Split by the median value of an attribute"),
        new Tag(SPLIT_DISCRETIZED, "Split by any value of an attribute where class value changes"),
        new Tag(SPLIT_RANGE, "Split by the midpoint of the range of the values of an attribute")
    };

    /**
     * Get the strategy object corresponding to the specified
     *  strategy ID
     *
     * @param strategyID The ID representing the strategy
     * @param numAttr The number of attributes each instance (of each bag in the MI dataset)
     * @return The strategy object corresponding to the strategyID
     */
    public static SplitStrategy getStrategy(final int strategyID, final int numAttr)
    {
        switch (strategyID)
        {
            case SPLIT_MEAN:
                return new MeanSplitStrategy(numAttr);
            case SPLIT_MEDIAN:
                return new MedianSplitStrategy(numAttr);
            case SPLIT_DISCRETIZED:
                return new DiscretizedSplitStrategy(numAttr);
            case SPLIT_RANGE:
                return new RangeSplitStrategy(numAttr);
            default:
                throw new IllegalArgumentException(
                        "Unknown split strategy code: " + strategyID);
        }
    }
    // </editor-fold>
}

abstract class CenterSplitStrategy extends SplitStrategy
{
    private final int numAttr;

    /** @param numAttr Number of attributes in the single-instance dataset. */
    protected CenterSplitStrategy(final int numAttr)
    {
        this.numAttr = numAttr;
    }

    /**
     * Find the center of the instances in trainingData along the attrIndex axis.
     * @param trainingData The bags of training instances.
     * @param attrIndex The attribute to find the center for.
     * @param ignore bitset of instances to ignore.
     * @return the center value of the instances along the attribute.
     */
    abstract double findCenter(Instances trainingData, int attrIndex, BitSet ignore);

    /** @inheritDoc */
    @Override
    public List<CompPair<Integer, Double>> generateSplitPoints(
            final Instances trainingData, final BitSet ignore)
    {
        List<CompPair<Integer, Double>> splits = new ArrayList<CompPair<Integer, Double>>(numAttr);

        for(int attr=0; attr<numAttr; attr++)
        {
            splits.add(new CompPair<Integer, Double>(attr, findCenter(trainingData, attr, ignore)));
        }

        return splits;
    }
}

/** Each candidate split is a mean of an attribute */
class MeanSplitStrategy extends CenterSplitStrategy
{
    /** @param numAttr Number of attributes in the single-instance dataset. */
    protected MeanSplitStrategy(final int numAttr)
    {
        super(numAttr);
    }

    /**
     * Find the mean of all instances in trainingData for the attribute at instIndex=attrIndex.
     * Assumes that the attribute is numeric. <== TODO may cause problems
     *
     * @param trainingData The dataset of mi-bags
     * @param attrIndex The instIndex of the attribute to find the mean for
     * @return The mean for the attribute over all instances in all bags
     */
    static double findMean(Instances trainingData, int attrIndex, BitSet ignore)
    {
        double sum = 0;
        int count = 0;
        int index = 0;

        // check in each bag
        for (Instance bag : trainingData)
        {
            // consider each instance in each bag
            for (Instance inst : bag.relationalValue(SplitNode.REL_INDEX))
            {
                if (!ignore.get(index++))
                {
                    sum += inst.value(attrIndex);
                    count++;
                }
            }
        }

        return sum / count;
    }

    /** @inheritDoc */
    @Override
    double findCenter(Instances trainingData, int attrIndex, BitSet ignore)
    {
        return findMean(trainingData, attrIndex, ignore);
    }
}

/** Each candidate split is the median of an attribute */
class MedianSplitStrategy extends CenterSplitStrategy
{
    /** @param numAttr Number of attributes in the single-instance dataset. */
    protected MedianSplitStrategy(final int numAttr)
    {
        super(numAttr);
    }

    /**
     * Find the median of all instances in trainingData for the attribute at instIndex=attrIndex.
     * Assumes that the attribute is numeric. <== TODO may cause problems
     *
     * @param trainingData The dataset of mi-bags
     * @param attrIndex The instIndex of the attribute to find the mean for
     * @param ignore bitset of instances to ignore
     * @return The mean for the attribute over all instances in all bags
     */
    static double findMedian(final Instances trainingData, final int attrIndex, BitSet ignore)
    {
        // for now:
        //  copy all values into a collection then sort
        List<Double> vals = new ArrayList<Double>();
        int index = 0;
        for (Instance bag : trainingData)
        {
            for (Instance inst : bag.relationalValue(SplitNode.REL_INDEX))
            {
                if (!ignore.get(index++))
                {
                    vals.add(inst.value(attrIndex));
                }
            }
        }

        Collections.sort(vals);

        final int count = vals.size();
        final boolean isEven = (count & 1) == 0;
        final int midIndex = count / 2;

        // if there is an even number of values, take the avg of the two middle elems.
        return isEven ? 0.5*(vals.get(midIndex) + vals.get(midIndex-1)) : vals.get(midIndex);
    }

    /** @inheritDoc */
    @Override
    double findCenter(Instances trainingData, int attrIndex, BitSet ignore)
    {
        return findMedian(trainingData, attrIndex, ignore);
    }
}

/** Each candidate is the midpt of the range of each attribute */
class RangeSplitStrategy extends CenterSplitStrategy
{
    /** @param numAttr Number of attributes in the single-instance dataset. */
    protected RangeSplitStrategy(final int numAttr)
    {
        super(numAttr);
    }

    /**
     * Find the midpoint of the range of all instances in trainingData for the attribute at instIndex=attrIndex.
     * Assumes that the attribute is numeric. <== TODO may cause problems
     *
     * @param trainingData The dataset of mi-bags
     * @param attrIndex The instIndex of the attribute to find the mean for
     * @param ignore bitset of instances to ignore
     * @return The mean for the attribute over all instances in all bags
     */
    static double findMidpt(final Instances trainingData, final int attrIndex, BitSet ignore)
    {
        double min = Double.MAX_VALUE;
        double max = -Double.MIN_VALUE;

        //  copy all values into a collection then sort
        int index = 0;
        for (Instance bag : trainingData)
        {
            for (Instance inst : bag.relationalValue(SplitNode.REL_INDEX))
            {
                if (!ignore.get(index++))
                {
                    double iVal = inst.value(attrIndex);
                    if (iVal < min) { min = iVal; }
                    if (iVal > max) { max = iVal; }
                }
            }
        }

        // return the midpoint of the range
        return ((max - min) / 2) + min;
    }

    /** @inheritDoc */
    @Override
    double findCenter(Instances trainingData, int attrIndex, BitSet ignore)
    {
        return findMidpt(trainingData, attrIndex, ignore);
    }
}

/** Each split point is a class-boundary across an attribute */
class DiscretizedSplitStrategy extends SplitStrategy
{
    private final int numAttr;

    /** @param numAttr Number of attributes in the single-instance dataset. */
    DiscretizedSplitStrategy(final int numAttr)
    {
        this.numAttr = numAttr;
    }

    /**
     * Find the points where the class changes when the single-instance
     *  dataset is sorted by the specified attribute.
     * @param trainingData The training data bags
     * @param attrIndex The attribute to order by
     * @param ignore the bitset of instances to ignore.
     * @return The points representing the class boundaries
     */
    static ArrayList<Double> findDiscretizedSplits(
            final Instances trainingData, final int attrIndex, final BitSet ignore)
    {
        List<CompPair<Double,Double>> vals = new ArrayList<CompPair<Double,Double>>();
        int index = 0;
        for (Instance bag : trainingData)
        {
            for (Instance inst : bag.relationalValue(SplitNode.REL_INDEX))
            {
                if (!ignore.get(index++))
                {
                    vals.add(new CompPair<Double,Double>(inst.value(attrIndex), bag.classValue()));
                }
            }
        }

        Collections.sort(vals);

        // iterate through the list, finding class-boundaries
        ArrayList<Double> splits = new ArrayList<Double>();
        CompPair<Double, Double> last = vals.get(0);
        final int size = vals.size();
        for(int i=1; i<size; i++)
        {
            CompPair<Double, Double> cur = vals.get(i);
            if (!last.value.equals(cur.value))
            {
                // this is a class boundary
                final double split = (last.key + cur.key) / 2.0;
                splits.add(split);
            }
            last = cur;
        }
        return splits;
    }

    /** @inheritDoc */
    @Override
    public List<CompPair<Integer, Double>> generateSplitPoints(
            final Instances trainingData, final BitSet ignore)
    {
        List<CompPair<Integer, Double>> splits = new ArrayList<CompPair<Integer, Double>>(numAttr);

        for(int attr=0; attr<numAttr; attr++)
        {
            for (double split : findDiscretizedSplits(trainingData, attr, ignore))
            {
                splits.add(new CompPair<Integer, Double>(attr, split));
            }
        }

        return splits;
    }

    /** @inheritDoc */
    @Override
    public boolean canExpand(final Instances dataset, final BitSet ignoreMask)
    {
        // check if bag is pure:
        boolean hasClass0 = false;
        boolean hasClass1 = false;
        for (Instance inst : dataset)
        {
            if (inst.classValue() < 0.5)
            {
                // this inst is class 0
                hasClass0 = true;
                if (hasClass1)
                {
                    return true;
                }
            }
            else
            {
                hasClass1 = true;
                if (hasClass0)
                {
                    return true;
                }
            }
        }

        return false;
    }
}