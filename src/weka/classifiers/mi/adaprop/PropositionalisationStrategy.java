package weka.classifiers.mi.adaprop;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Tag;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;

/**
 * Represents a method for propositionalisation of a set of instances.
 * This strategy is used to convert a set of instances
 * (i.e. those which fall into a region)
 * into a vector of attributes.
 */
public abstract class PropositionalisationStrategy implements Serializable
{
    /**
     * Propositionalise the bag and place the resultant vector in the
     *  result array, starting at the specified index.
     *
     * @param bag The bag of instances to propositionalise
     * @param result The resultant array to place the results into.
     * @param resultStartIndex The starting location (inclusive) to place the result.
     */
    public abstract void propositionalise(Instances bag, BitSet ignore, double[] result, int resultStartIndex);

    /**
     * @return the number of attributes per region in the propositionalised data-set
     */
    public abstract int getNumPropAttrPerRegion();

    /**
     * @return the attributes of the propositionalised data-set
     */
    public abstract ArrayList<Attribute> getPropAttributes(final int numRegions);

    // <editor-fold desc="===Option Handling===">
    private static final int PROP_COUNT = 1;
    private static final int PROP_SUMMARY = 2;
    public static final int DEFAULT_STRATEGY = PROP_COUNT;
    public static final String DESCRIPTION =
            "Propositionalisation strategy: 1=count-only (default), 2=all-summary-stats";

    public static final Tag[] STRATEGIES =
    {
        new Tag(PROP_COUNT, "Using counts of each region only"),
        new Tag(PROP_SUMMARY, "Using summary statistics of each region")
    };

    /**
     * Get the strategy object corresponding to the specified
     *  strategy ID
     *
     * @param strategyID The ID representing the strategy
     * @param numAttr The number of attributes each instance (of each bag in the MI dataset)
     * @return The strategy object corresponding to the strategyID
     */
    public static PropositionalisationStrategy getStrategy(final int strategyID, final int numAttr)
    {
        switch (strategyID)
        {
            case PROP_COUNT:
                return new CountBasedPropositionalisationStrategy();
            case PROP_SUMMARY:
                return new SummaryStatsBasedPropositionalisationStrategy(numAttr);
            default:
                throw new IllegalArgumentException(
                        "Unknown propositionalisation strategy code: " +
                        strategyID);
        }
    }
    // </editor-fold>
}

class CountBasedPropositionalisationStrategy extends PropositionalisationStrategy
{
    @Override /** @inheritDoc */
    public void propositionalise(final Instances bag, final BitSet ignore, final double[] result,
                                 final int resultStartIndex)
    {
        // just place the count into the result
        result[resultStartIndex] = bag.size() - ignore.cardinality();
    }

    @Override /** @inheritDoc */
    public int getNumPropAttrPerRegion()
    {
        return 1; // only 1 per region
    }

    @Override /** @inheritDoc */
    public ArrayList<Attribute> getPropAttributes(final int numRegions)
    {
        ArrayList<Attribute> attrInfo = new ArrayList<Attribute>(numRegions+1);
        for (int i=0; i<numRegions; i++)
        {
            attrInfo.add(new Attribute("region " + i)); // TODO better names for attr?
        }
        return attrInfo;
    }
}

class SummaryStatsBasedPropositionalisationStrategy extends PropositionalisationStrategy
{
    private final int numAttr;

    SummaryStatsBasedPropositionalisationStrategy(final int numAttr)
    {
        this.numAttr = numAttr;
    }

    @Override /** @inheritDoc */
    public void propositionalise(final Instances bag, final BitSet ignore, final double[] result,
                                 final int resultStartIndex)
    {
        // compute each of the necessary summary stats, for each attribute.
        for (int attrIndex=0; attrIndex<numAttr; attrIndex++)
        {
            SummaryStatCalculator sumStat = new SummaryStatCalculator(attrIndex);
            int instIndex = 0;
            for (Instance inst : bag)
            {
                if (!ignore.get(instIndex++))
                {
                    sumStat.addInstance(inst);
                }
            }
            sumStat.storeResults(result, resultStartIndex + SummaryStatCalculator.NUM_ATTR * attrIndex);
        }
    }

    @Override /** @inheritDoc */
    public int getNumPropAttrPerRegion()
    {
        return this.numAttr * SummaryStatCalculator.NUM_ATTR;
    }

    @Override /** @inheritDoc */
    public ArrayList<Attribute> getPropAttributes(final int numRegions)
    {
        final int propNumAttr = SummaryStatCalculator.NUM_ATTR;
        ArrayList<Attribute> attrInfo = new ArrayList<Attribute>(numAttr*numRegions*propNumAttr + 1);

        // for each region
        for (int region=0; region<numRegions; region++)
        {
            // for each attribute
            for (int attr=0; attr<numAttr; attr++)
            {
                // for each summary stat
                for (int sumStat=0; sumStat<propNumAttr; sumStat++)
                {
                    final String attrName = SummaryStatCalculator.SUMMARY_STATS[sumStat] +
                            "-of-attr-" + attr + " region " + region;
                    attrInfo.add(new Attribute(attrName)); // TODO better names for attr?
                }
            }
        }
        return attrInfo;
    }

    /**
     * Computes the summary statistics for one attribute
     */
    static class SummaryStatCalculator
    {
        static final String[] SUMMARY_STATS = { "count", "sum", "min", "max", "avg"};
        static final int NUM_ATTR = SUMMARY_STATS.length;

        private double min;
        private double max;
        private double sum;
        private double count;

        private final int attrIndex;

        public SummaryStatCalculator(final int attrIndex)
        {
            this.attrIndex = attrIndex;

            // set to default values
            this.min = Double.MAX_VALUE;
            this.max = - Double.MAX_VALUE;
            this.count = 0;
            this.sum = 0;
        }

        public void addInstance(final Instance instance)
        {
            final double attrVal = instance.value(attrIndex);

            // update summary stats
            this.count++;
            this.sum += attrVal;
            if (attrVal < this.min) {
                this.min = attrVal;
            }
            if (attrVal > this.max) {
                this.max = attrVal;
            }
        }

        public void storeResults(final double[] result, final int resultStartIndex)
        {
            // store the summary stats
            result[resultStartIndex    ] = this.count;
            result[resultStartIndex + 1] = this.sum;
            result[resultStartIndex + 2] = (this.count == 0) ? 0.0 : this.min;
            result[resultStartIndex + 3] = (this.count == 0) ? 0.0 : this.max;
            final double avg = (this.count == 0) ? 0.0 : this.sum / this.count;
            result[resultStartIndex + 4] = avg;
        }
    }
}
