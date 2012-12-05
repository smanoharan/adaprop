package weka.classifiers.mi;

import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.mi.adaprop.*;
import weka.classifiers.trees.RandomForest;
import weka.core.*;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
 * An adaptive propositionalization algorithm. Uses the base learner to decide
 *  on the best attribute to split on. For now, just a 1-level tree.
 *  TODO update
 *
 * @author Siva Manoharan
 */
public class AdaProp extends SingleClassifierEnhancer
        implements MultiInstanceCapabilitiesHandler, OptionHandler
{
    /**
     * For serialization:
     *  format: 1[dd][mm][yyyy]00..0[digit revision number]L
     */
    public static final long serialVersionUID = 1041220120000022L;

    /** The tree of splits */
    protected RootSplitNode splitTreeRoot;

    /** Contains the bags as propositionalised instances */
    protected Instances propositionalisedDataset;

    /** The current propositionalisation strategy (as an object) */
    protected PropositionalisationStrategy propStrategy;

    //<editor-fold defaultstate="collapsed" desc="===Option Handling===">
    private static final int DEFAULT_MAX_TREE_SIZE = 8;
    private static final int DEFAULT_MIN_OCCUPANCY = 5;

    // keys for command line options: (e.g. when using "AdaProp -prop 1 -maxTreeSize 4" etc)
    public static final String SPLIT_KEY = "split";
    public static final String SEARCH_KEY = "search";
    public static final String PROP_KEY = "prop";
    public static final String EVAL_KEY = "eval";
    public static final String MAX_TREE_KEY = "maxTreeSize";
    public static final String MIN_OCC_KEY = "minOcc";
    public static final String MAX_TREE_DESCRIPTION =
            "Maximum size (number of nodes) of the tree. Default=8.";
    public static final String MIN_OCC_DESCRIPTION =
            "Minimum occupancy of each node of the tree. Default=5.";

    /** The id of the instance-space splitting strategy to use */
    protected int m_SplitStrategy = SplitStrategy.DEFAULT_STRATEGY;

    /** The id of the tree building search strategy to use */
    protected int m_SearchStrategy = SearchStrategy.DEFAULT_STRATEGY;

    /** The id of the propositionalisation strategy to use */
    protected int m_PropositionalisationStrategy = PropositionalisationStrategy.DEFAULT_STRATEGY;

    /** The id of the evaluation strategy to use */
    protected int m_EvalStrategy = EvaluationStrategy.DEFAULT_STRATEGY;

    /** The effective maximum depth of the tree of splits (0 for unlimited) */
    protected int m_MaxTreeSize = DEFAULT_MAX_TREE_SIZE;

    /** The minimum occupancy of each leaf node in the tree */
    protected int m_MinOccupancy = DEFAULT_MIN_OCCUPANCY;

    /** For randomization (when performing CV) */
    protected Random m_Random = new Random(1);

    /**
     * Helper function to select the correct ID for the
     *  selected tag.
     *
     * @param strategy The selected tag (corresponding to the selected strategy)
     * @param tags The array of all strategy tags
     * @return The ID corresponding to the selected strategy
     */
    private static int getID(SelectedTag strategy, Tag[] tags)
    {
        if (strategy.getTags() == tags) {
            return strategy.getSelectedTag().getID();
        } else {
            throw new IllegalArgumentException("Unknown tag: " + strategy);
        }
    }

    /**
     * Gets the current instance-space splitting strategy
     * @return the current splitting strategy
     */
    public SelectedTag getSplitStrategy()
    {
        return new SelectedTag(m_SplitStrategy, SplitStrategy.STRATEGIES);
    }

    /**
     * Sets the instance-space splitting selection strategy.
     * @param newStrategy splitting selection strategy.
     */
    public void setSplitStrategy(final SelectedTag newStrategy)
    {
        this.m_SplitStrategy = getID(newStrategy, SplitStrategy.STRATEGIES);
    }

    /**
     * Gets the current tree building search strategy
     * @return the current search strategy
     */
    public SelectedTag getSearchStrategy()
    {
        return new SelectedTag(this.m_SearchStrategy, SearchStrategy.STRATEGIES);
    }

    /**
     * Sets the tree building search strategy
     * @param newStrategy the new search strategy
     */
    public void setSearchStrategy(final SelectedTag newStrategy)
    {
        this.m_SearchStrategy = getID(newStrategy, SearchStrategy.STRATEGIES);
    }

    /**
     * Gets the propositionalisation strategy
     * @return the current propositionalisation strategy
     */
    public SelectedTag getPropositionalisationStrategy()
    {
        return new SelectedTag(this.m_PropositionalisationStrategy,
                PropositionalisationStrategy.STRATEGIES);
    }

    /**
     * Sets the propositionalisation strategy
     * @param newStrategy the new propositionalisation strategy
     */
    public void setPropositionalisationStrategy(final SelectedTag newStrategy)
    {
        this.m_PropositionalisationStrategy =
                getID(newStrategy, PropositionalisationStrategy.STRATEGIES);
    }

    /**
     * Gets the current tree building search strategy
     * @return the current search strategy
     */
    public SelectedTag getEvalStrategy()
    {
        return new SelectedTag(this.m_EvalStrategy, EvaluationStrategy.STRATEGIES);
    }

    /**
     * Sets the tree building search strategy
     * @param newStrategy the new search strategy
     */
    public void setEvalStrategy(final SelectedTag newStrategy)
    {
        this.m_EvalStrategy = getID(newStrategy, EvaluationStrategy.STRATEGIES);
    }

    /**
     * Gets the max tree size
     * @return the max tree size
     */
    public int getMaxTreeSize()
    {
        return m_MaxTreeSize;
    }

    /**
     * Sets the max tree size
     * @param maxTreeSize The maximum tree size
     */
    public void setMaxTreeSize(int maxTreeSize)
    {
        m_MaxTreeSize = maxTreeSize;
    }

    /**
     * Gets the min occupancy for each leaf node
     * @return the min occupancy
     */
    public int getMinOccupancy()
    {
        return m_MinOccupancy;
    }

    /**
     * Sets the min occupancy for each leaf node
     * @param minOccupancy The min occupancy
     */
    public void setMinOccupancy(int minOccupancy)
    {
        m_MinOccupancy = minOccupancy;
    }

    @Override /** @inheritDoc */
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

    /** @inheritDoc */
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

    /** Helper function for creating options with exactly 1 argument. */
    private static Option toUnaryOption(String desc, String key)
    {
        return new Option("\t" + desc, key, 1, "-"+ key +" <num>");
    }

    /** @inheritDoc */
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public Enumeration listOptions()
    {
        Vector result = new Vector();

        // add each option:
        result.addElement(toUnaryOption(SplitStrategy.DESCRIPTION, SPLIT_KEY));
        result.addElement(toUnaryOption(SearchStrategy.DESCRIPTION, SEARCH_KEY));
        result.addElement(toUnaryOption(PropositionalisationStrategy.DESCRIPTION, PROP_KEY));
        result.addElement(toUnaryOption(EvaluationStrategy.DESCRIPTION, EVAL_KEY));
        result.addElement(toUnaryOption(MAX_TREE_DESCRIPTION, MAX_TREE_KEY));
        result.addElement(toUnaryOption(MIN_OCC_DESCRIPTION, MIN_OCC_KEY));

        // copy each of the superclass' options
        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements())
        {
            result.addElement(enu.nextElement());
        }

        return result.elements();
    }

    /**
     * Check if the value for the specified key can be found in the options array,
     *      and if so, return the Tag corresponding to the specified value.
     * Otherwise, return the Tag corresponding to the default value.
     *
     * @param key The key to look for in the options array.
     * @param options The options array.
     * @param defaultID The default value.
     * @param tags The set of all tags for this option (i.e. the set of all strategy tags).
     * @return The Tag corresponding to the value or defaultID if key is not found.
     */
    private static SelectedTag parseTag(final String key, final String[] options,
                                        final int defaultID, final Tag[] tags) throws Exception
    {
        final String value = Utils.getOption(key, options);
        final int splitID = value.isEmpty() ? defaultID : Integer.parseInt(value);
        return new SelectedTag(splitID, tags);
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
        setSplitStrategy(parseTag(SPLIT_KEY, options,
                SplitStrategy.DEFAULT_STRATEGY, SplitStrategy.STRATEGIES));

        setSearchStrategy(parseTag(SEARCH_KEY, options,
                SearchStrategy.DEFAULT_STRATEGY, SearchStrategy.STRATEGIES));

        setPropositionalisationStrategy(parseTag(PROP_KEY, options,
                PropositionalisationStrategy.DEFAULT_STRATEGY,
                PropositionalisationStrategy.STRATEGIES));

        setEvalStrategy(parseTag(EVAL_KEY, options,
                EvaluationStrategy.DEFAULT_STRATEGY, EvaluationStrategy.STRATEGIES));

        final String maxDepthStr = Utils.getOption(MAX_TREE_KEY, options);
        this.setMaxTreeSize(maxDepthStr.isEmpty() ? DEFAULT_MAX_TREE_SIZE : Integer.parseInt(maxDepthStr));

        final String minOccStr = Utils.getOption(MIN_OCC_KEY, options);
        this.setMinOccupancy(minOccStr.isEmpty() ? DEFAULT_MIN_OCCUPANCY : Integer.parseInt(minOccStr));

        super.setOptions(options);
    }

    /** @inheritDoc */
    @SuppressWarnings({ "unchecked", "rawtypes" })
    public String[] getOptions()
    {
        Vector result = new Vector();

        result.add("-" + SPLIT_KEY);
        result.add("" + m_SplitStrategy);
        result.add("-" + SEARCH_KEY);
        result.add("" + m_SearchStrategy);
        result.add("-" + PROP_KEY);
        result.add("" + m_PropositionalisationStrategy);
        result.add("-" + EVAL_KEY);
        result.add("" + m_EvalStrategy);
        result.add("-" + MAX_TREE_KEY);
        result.add("" + m_MaxTreeSize);
        result.add("-" + MIN_OCC_KEY);
        result.add("" + m_MinOccupancy);

        result.addAll(Arrays.asList(super.getOptions()));
        return (String[]) result.toArray(new String[result.size()]);
    }
    //</editor-fold>

    /** Allow running from CLI. */
    public static void main(String[] args)
    {
        runClassifier(new AdaProp(), args);
    }

    public AdaProp()
    {
        super.m_Classifier = new RandomForest();
    }

    @Override
    protected String defaultClassifierString()
    {
        return "weka.classifiers.trees.RandomForest";
    }

    /** @return a String describing this classifier. */
    public String globalInfo()
    {
        return "An adaptive propositionalization algorithm."; // TODO add more
    }

    /** @return A string representation of this model. */
    @Override
    public String toString()
    {
        return "Tree of splits: \n---------------\n\n" +
                (splitTreeRoot == null ? "not-yet-created." : splitTreeRoot.toString()) + "\n\n" +
                (m_Classifier == null ? "no classifier model." : m_Classifier.toString());
    }

    @Override /** @inheritDoc */
    public double[] distributionForInstance(Instance newBag) throws Exception
    {
        // propositionalise the bag
        Instance propositionalisedTrainingData = SplitNode.propositionaliseBag(
                newBag, splitTreeRoot, propositionalisedDataset, propStrategy);

        // use the base classifier for prediction.
        return m_Classifier.distributionForInstance(propositionalisedTrainingData);
    }

    @Override /** @inheritDoc */
    public void buildClassifier(Instances trainingDataBags) throws Exception
    {
        if (m_Classifier == null)
        {
            throw new Exception("A base classifier has not been specified.");
        }

        // can classifier handle the data?
        getCapabilities().testWithFail(trainingDataBags);

        // remove instances with missing class (make a copy first)
        Instances trainingBags = new Instances(trainingDataBags);
        trainingBags.deleteWithMissingClass();

        // TODO : what if the dataset is empty?
        final int numAttr = trainingBags.instance(0).relationalValue(1).numAttributes();

        // convert the strategy IDs to strategy objects:
        SplitStrategy splitStrategy = SplitStrategy.getStrategy(m_SplitStrategy, numAttr);
        SearchStrategy searchStrategy = SearchStrategy.getStrategy(m_SearchStrategy);
        propStrategy = PropositionalisationStrategy.getStrategy(m_PropositionalisationStrategy, numAttr);
        EvaluationStrategy evalStrategy = EvaluationStrategy.getStrategy(m_EvalStrategy, m_Random);

        // create the tree of splits:
        splitTreeRoot = SplitNode.buildTree(trainingBags, splitStrategy, m_MaxTreeSize,
                m_MinOccupancy, m_Classifier, searchStrategy, propStrategy, evalStrategy);

        // retrain m_classifier with the best attribute:
        Instances propositionalisedTrainingData = SplitNode.propositionaliseDataset(
                trainingBags, splitTreeRoot, propStrategy);
        m_Classifier.buildClassifier(propositionalisedTrainingData);
        propositionalisedDataset = new Instances(propositionalisedTrainingData, 0);
    }
}
