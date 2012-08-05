package weka.classifiers.mi;

import weka.classifiers.SingleClassifierEnhancer;
import weka.core.*;

import java.util.*;

/**
 * An adaptive propositionalization algorithm. Uses the base learner to decide
 *  on the best attribute to split on. For now, just a 1-level tree.
 *  TODO update
 *
 * Author: Siva Manoharan
 */
public class AdaptiveSplit extends SingleClassifierEnhancer
        implements MultiInstanceCapabilitiesHandler, OptionHandler
{
    /** for serialization */
    static final long serialVersionUID = -131449935521003121L;

    /** The index of the relational attribute in the bag instance */
    public static final int REL_INDEX = 1;

    // ==================================================================================
    // For Options:
    // ==================================================================================

    // Split point
    private static final int SPLIT_MEAN = 1;
    private static final int SPLIT_MEDIAN = 2;
    private static final int SPLIT_DISCRETIZED = 3;
    private static final int DEFAULT_SPLIT_STRATEGY = SPLIT_MEAN;

    public static final Tag [] SPLIT_STRATEGIES =
    {
        new Tag(SPLIT_MEAN, "Split by the mean value of an attribute"),
        new Tag(SPLIT_MEDIAN, "Split by the median value of an attribute"),
        new Tag(SPLIT_DISCRETIZED, "Split by any value of an attribute where class value changes")
    };

    /** The instance-space splitting strategy to use */
    protected int m_SplitStrategy = DEFAULT_SPLIT_STRATEGY;

    /**
     * Gets the current instance-space splitting strategy
     * @return the current splitting strategy
     */
    public SelectedTag getSplitStrategy()
    {
        return new SelectedTag(this.m_SplitStrategy, SPLIT_STRATEGIES);
    }

    /**
     * Sets the instance-space splitting selection strategy.
     * @param newStrategy splitting selection strategy.
     */
    public void setSplitStrategy(final SelectedTag newStrategy)
    {
        if (newStrategy.getTags() == SPLIT_STRATEGIES)
        {
            this.m_SplitStrategy = newStrategy.getSelectedTag().getID();
        }
        else throw new RuntimeException(
                "Unknown tag (not a splitting strategy tag): " + newStrategy);
    }

    // ==================================================================================

    /** The best attribute to split on */
    protected int m_BestAttrToSplitOn;

    /** The split point for that attribute (assume NUMERIC) */
    protected double m_AttrSplitPoint;

    /** Contains the bags as propositionalised instances */
    protected Instances m_propositionalisedDataset;

    /** Allow running from cmd prompt. */
    public static void main(String[] args)
    {
        runClassifier(new AdaptiveSplit(), args);
    }

    /** @return a String describing this classifier. */
    public String globalInfo()
    {
        return "An adaptive propositionalization algorithm."; // TODO add more
    }

    @Override // TODO Copy over Javadocs
    public Capabilities getCapabilities()
    {
        // TODO Check these
        Capabilities result = super.getCapabilities();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.disable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.ONLY_MULTIINSTANCE);

        // class
        result.disableAllClasses();
        result.disableAllClassDependencies();
        result.enable(Capabilities.Capability.BINARY_CLASS);

        // Only multi instance data
        result.enable(Capabilities.Capability.ONLY_MULTIINSTANCE);

        return result;
    }

    // TODO Copy over Javadocs
    public Capabilities getMultiInstanceCapabilities()
    {
        // TODO check these
        Capabilities result = super.getCapabilities();

        // class
        result.disableAllClasses();
        result.enable(Capabilities.Capability.NO_CLASS);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    /** @inheritDoc */
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public Enumeration listOptions()
    {
        Vector result = new Vector();

        // split point choice
        result.addElement(new Option(
            "\tSplit point criterion: 1=mean (default), 2=median, 3=discretized",
            "S", 1, "-S <num>"));

        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements())
        {
            result.addElement(enu.nextElement());
        }

        return result.elements();
    }

    /**
     * Lists the options for this classifier.
     <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -S <num>
     *  Split point criterion: 1=mean (default), 2=median, 3=discretized</pre>
     *
     <!-- options-end -->
     */
    @Override
    public void setOptions(String[] options) throws Exception
    {
//            String valString = Utils.getOption('M', options);
//            Type newValue = isValid(valString) ? parse(valString) : defaultValue;
//            setProperty(newValue);
//            OR
//            setB(Utils.getFlag('B', options));

        // split strategy
        String splitStrategyStr = Utils.getOption('S', options);
        this.setSplitStrategy(new SelectedTag(Integer.parseInt(splitStrategyStr), SPLIT_STRATEGIES));
        super.setOptions(options);
    }

    /** @inheritDoc */
    @SuppressWarnings({ "unchecked", "rawtypes" })
    public String[] getOptions()
    {
        Vector result = new Vector();

        // split option
        result.add("-S");
        result.add("" + m_SplitStrategy);

        String[] options = super.getOptions();
        for (int i = 0; i < options.length; i++)
            result.add(options[i]);

        return (String[]) result.toArray(new String[result.size()]);
    }


    /**
     * @return A string representation of this model.
     */
    @Override
    public String toString()
    {
        // TODO
        return "TODO";
    }

    @Override /** @inheritDoc */
    public double[] distributionForInstance(Instance newBag) throws Exception
    {
        // propositionalise the bag using the attribute and it's split point.
        final Instance propositionalisedBag = propositionaliseBag(
                newBag.relationalValue(REL_INDEX),
                m_BestAttrToSplitOn,
                m_AttrSplitPoint,
                newBag.classValue(),
                m_propositionalisedDataset);

        // use the base classifier for prediction.
        return m_Classifier.distributionForInstance(propositionalisedBag);
    }

    @Override /** @inheritDoc */
    public void buildClassifier(Instances trainingDataBags) throws Exception
    {
        if (m_Classifier == null)
        {
            throw new Exception("A base classifier has not been specified!");
        }

        // can classifier handle the data?
        getCapabilities().testWithFail(trainingDataBags);

        // remove instances with missing class (make a copy first)
        Instances trainingBags = new Instances(trainingDataBags);
        trainingBags.deleteWithMissingClass();

        // TODO create a single-instance table? (unnecessary?)

        int numSingleInstAttr = trainingBags.instance(0).relationalValue(1).numAttributes();
        double minErr = Double.MAX_VALUE;

        // iterate through the attributes, find the one with the least error.
        for(int attr = 0; attr < numSingleInstAttr; attr++)
        {
            double err = evaluateSplittingDimension(trainingBags, attr);
            if (err < minErr)
            {
                minErr = err;
                m_BestAttrToSplitOn = attr;
            }
        }

        // TODO repeat recursively?
        // TODO retrain m_classifier with the best attribute.
        // TODO can we avoid re-training classifier with best-attr-to-split-on
    }

    /**
     * Find the mean of all instances in trainingData for the attribute at index=attrIndex.
     * Assumes that the attribute is numeric. <== TODO may cause problems
     *
     * @param trainingData The dataset of mi-bags
     * @param attrIndex The index of the attribute to find the mean for
     * @return The mean for the attribute over all instances in all bags
     */
    static double findMean(Instances trainingData, int attrIndex)
    {
        double sum = 0;
        int count = 0;

        // check in each bag
        for (Instance bag : trainingData)
        {
            // consider each instance in each bag
            for (Instance inst : bag.relationalValue(REL_INDEX))
            {
                sum += inst.value(attrIndex);
                count++;
            }
        }

        return sum / count;
    }

    /**
     * Find the median of all instances in trainingData for the attribute at index=attrIndex.
     * Assumes that the attribute is numeric. <== TODO may cause problems
     *
     * @param trainingData The dataset of mi-bags
     * @param attrIndex The index of the attribute to find the mean for
     * @return The mean for the attribute over all instances in all bags
     */
    static double findMedian(final Instances trainingData, final int attrIndex)
    {
        // for now:
        //  copy all values into a collection then sort
        List<Double> vals = new ArrayList<Double>();
        for (Instance bag : trainingData)
        {
            for (Instance inst : bag.relationalValue(REL_INDEX))
            {
                vals.add(inst.value(attrIndex));
            }
        }

        Collections.sort(vals);

        final int count = vals.size();
        final boolean isEven = (count & 1) == 0;
        final int midIndex = count / 2;

        // if there is an even number of values, take the avg of the two middle elems.
        return isEven ? 0.5*(vals.get(midIndex) + vals.get(midIndex-1)) : vals.get(midIndex);
    }

    // TODO javadocs
    double evaluateSplittingDimension(Instances trainingData,
                                              int attrIndex)
    {
        // TODO : support NOM { branch split } , missing { how? }
        // TODO cache vals for efficiency

        // setup attr
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
        attInfo.add(new Attribute("less-than"));
        attInfo.add(new Attribute("greater-than"));
        attInfo.add((Attribute) trainingData.classAttribute().copy()); // class

        // create propositionalised dataset
        final int numBags = trainingData.numInstances();
        m_propositionalisedDataset = new Instances("prop", attInfo, numBags);

        // TODO update this or make a const
        m_propositionalisedDataset.setClassIndex(2);

        // find the split point
        // TODO need to convert this?
        double splitPoint = findMean(trainingData, attrIndex);

        for (Instance bag : trainingData)
        {
            // propositionalise the bag and add it to the set
            // TODO efficiency concerns
            Instance propositionalisedBag = propositionaliseBag(
                    bag.relationalValue(REL_INDEX),
                    attrIndex,
                    splitPoint,
                    bag.classValue(),
                    m_propositionalisedDataset);

            m_propositionalisedDataset.add(propositionalisedBag);
        }

        // eval on propositionalised dataset
        // TODO, not sure if the following works...
        // TODO efficiency reasons.. is it better to compute non-cv error rate?
        try
        {
            m_Classifier.buildClassifier(m_propositionalisedDataset);

            // count num errors
            int numErr = 0;
            for (Instance inst : m_propositionalisedDataset)
            {
                if (m_Classifier.classifyInstance(inst) != inst.classValue())
                {
                    numErr++;
                }
            }

            return ((double) numErr); // TODO no need to divide by numInst?
        }
        catch (Exception e)
        {
            // TODO what to do?
            throw new RuntimeException(e);
        }
    }

    /**
     *  Use this at test time.
     *  It is possible to be more efficient at train time.
     */
    static Instance propositionaliseBag(Instances bagInstances,
                    int attrIndex, double splitPoint, double classVal,
                    Instances propositionalisedDataset)
    {
        // TODO support NOM splitting attr, missing vals

        // count the number of instances with less than and geq value for the
        //  split attribute.
        int countLessThan = 0;
        int countGeq = 0;
        for(Instance inst : bagInstances)
        {
            if (inst.value(attrIndex) < splitPoint)
            {
                countLessThan++;
            }
            else
            {
                countGeq++;
            }
        }

        final double[] attValues = {countLessThan, countGeq, classVal};
        Instance i = new DenseInstance(1.0, attValues);
        i.setDataset(propositionalisedDataset);

        return i;
    }
}
