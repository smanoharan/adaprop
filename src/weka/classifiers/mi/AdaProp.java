package weka.classifiers.mi;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.SingleClassifierEnhancer;
import weka.core.*;

import java.io.Serializable;
import java.util.*;
import java.util.Queue;

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
    static final long serialVersionUID = 1091120120000015L;

    /** The tree of splits */
    protected RootSplitNode splitTreeRoot;

    /** Contains the bags as propositionalised instances */
    protected Instances m_propositionalisedDataset;

    /** The propositionalisation strategy to use */
    protected PropositionalisationStrategy propStrategy = new CountBasedPropositionalisationStrategy();

    /** The instIndex of the relational attribute in the bag instance */
    public static final int REL_INDEX = 1;

    //<editor-fold defaultstate="collapsed" desc="===Option Handling===">
    private static final int SPLIT_MEAN = 1;
    private static final int SPLIT_MEDIAN = 2;
    private static final int SPLIT_DISCRETIZED = 3;
    private static final int SPLIT_RANGE = 4;
    private static final int SEARCH_BREADTH_FIRST = 1;
    private static final int SEARCH_BEST_FIRST = 2;
    private static final int DEFAULT_SPLIT_STRATEGY = SPLIT_MEAN;
    private static final int DEFAULT_SEARCH_STRATEGY = SEARCH_BREADTH_FIRST;
    private static final int DEFAULT_MAX_DEPTH = 3;
    private static final int DEFAULT_MIN_OCCUPANCY = 5;

    public static final Tag [] SPLIT_STRATEGIES =
    {
        new Tag(SPLIT_MEAN, "Split by the mean value of an attribute"),
        new Tag(SPLIT_MEDIAN, "Split by the median value of an attribute"),
        new Tag(SPLIT_DISCRETIZED, "Split by any value of an attribute where class value changes"),
        new Tag(SPLIT_RANGE, "Split by the midpoint of the range of the values of an attribute")
    };

    public static final Tag [] SEARCH_STRATEGIES =
    {
        new Tag(SEARCH_BREADTH_FIRST, "Build the tree using breadth first search"),
        new Tag(SEARCH_BEST_FIRST, "Build the tree using best first search")
    };

    /** The id of the instance-space splitting strategy to use */
    protected int m_SplitStrategy = DEFAULT_SPLIT_STRATEGY;

    /** The effective maximum depth of the tree of splits (0 for unlimited) */
    protected int m_MaxDepth = DEFAULT_MAX_DEPTH;

    /** The minimum occupancy of each leaf node in the tree */
    protected int m_MinOccupancy = DEFAULT_MIN_OCCUPANCY;

    /** The id of the tree building search strategy to use */
    protected int m_SearchStrategy = DEFAULT_SEARCH_STRATEGY;

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

    /**
     * Gets the current tree building search strategy
     * @return the current search strategy
     */
    public SelectedTag getSearchStrategy()
    {
        return new SelectedTag(this.m_SearchStrategy, SEARCH_STRATEGIES);
    }

    /**
     * Sets the tree building search strategy
     * @param newStrategy the new search strategy
     */
    public void setSearchStrategy(final SelectedTag newStrategy)
    {
        if (newStrategy.getTags() == SEARCH_STRATEGIES)
        {
            this.m_SearchStrategy = newStrategy.getSelectedTag().getID();
        }
        else throw new RuntimeException(
                "Unknown tag (not a search strategy tag): " + newStrategy);
    }

    /**
     * Gets the max tree depth
     * @return the max depth
     */
    public int getMaxDepth()
    {
        return m_MaxDepth;
    }

    /**
     * Sets the max tree depth
     * @param maxDepth The maximum tree depth
     */
    public void setMaxDepth(int maxDepth)
    {
        m_MaxDepth = maxDepth;
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
                "\tSplit point criterion: 1=mean (default), 2=median, 3=discretized, 4=range",
                "split", 1, "-split <num>"));

        // search strategy choice
        result.addElement(new Option(
                "\tSearch strategy: 1=breadth-first (default), 2=best-first",
                "search", 1, "-search <num>"));

        // max depth
        result.addElement(new Option(
                "\tMaximum depth of the tree. 0 for unlimited (default).",
                "maxDepth", 1, "-maxDepth <num>"));

        // min occupancy
        result.addElement(new Option(
                "\tMinimum occupancy of each node of the tree. Default=2",
                "minOcc", 1, "-minOcc <num>"));

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
        String splitStrategyStr = Utils.getOption("split", options);
        final int splitID = splitStrategyStr.isEmpty() ?
                DEFAULT_SPLIT_STRATEGY :
                Integer.parseInt(splitStrategyStr);
        this.setSplitStrategy(new SelectedTag(splitID, SPLIT_STRATEGIES));

        String searchStrategyStr = Utils.getOption("search", options);
        final int searchID = searchStrategyStr.isEmpty() ?
                DEFAULT_SEARCH_STRATEGY :
                Integer.parseInt(searchStrategyStr);
        this.setSearchStrategy(new SelectedTag(searchID, SEARCH_STRATEGIES));

        String maxDepthStr = Utils.getOption("maxDepth", options);
        this.setMaxDepth(maxDepthStr.isEmpty() ? DEFAULT_MAX_DEPTH : Integer.parseInt(maxDepthStr));

        String minOccStr = Utils.getOption("minOcc", options);
        this.setMinOccupancy(minOccStr.isEmpty() ? DEFAULT_MAX_DEPTH :Integer.parseInt(minOccStr));

        super.setOptions(options);
    }

    /** @inheritDoc */
    @SuppressWarnings({ "unchecked", "rawtypes" })
    public String[] getOptions()
    {
        Vector result = new Vector();

        result.add("-split");
        result.add("" + m_SplitStrategy);
        result.add("-search");
        result.add("" + m_SearchStrategy);
        result.add("-maxDepth");
        result.add("" + m_MaxDepth);
        result.add("-minOcc");
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

    /** @return a String describing this classifier. */
    public String globalInfo()
    {
        return "An adaptive propositionalization algorithm."; // TODO add more
    }

    /** @return A string representation of this model. */
    @Override
    public String toString()
    {
        return "Tree of splits: \n\n" +
                (splitTreeRoot == null ? "not-yet-created." : splitTreeRoot.toString()) + "\n\n" +
                (m_Classifier == null ? "no classifier model." : m_Classifier.toString());
    }

    @Override /** @inheritDoc */
    public double[] distributionForInstance(Instance newBag) throws Exception
    {
        // propositionalise the bag
        Instance propositionalisedTrainingData =
                SplitNode.propositionaliseBag(newBag, splitTreeRoot, m_propositionalisedDataset, propStrategy);

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

        final int numAttr = trainingBags.instance(0).relationalValue(1).numAttributes();

        // find the split strategy:
        SplitStrategy splitStrategy;
        switch (m_SplitStrategy)
        {
            case SPLIT_MEAN:
                splitStrategy = new MeanSplitStrategy(numAttr);
                break;
            case SPLIT_MEDIAN:
                splitStrategy = new MedianSplitStrategy(numAttr);
                break;
            case SPLIT_DISCRETIZED:
                splitStrategy = new DiscretizedSplitStrategy(numAttr);
                break;
            case SPLIT_RANGE:
                splitStrategy = new RangeSplitStrategy(numAttr);
                break;
            default:
                throw new IllegalArgumentException("Unknown split strategy code: " + m_SplitStrategy);
        }

        // find the search strategy:
        SearchStrategy searchStrategy;
        switch (m_SearchStrategy)
        {
            case SEARCH_BREADTH_FIRST:
                searchStrategy = new BreadthFirstSearchStrategy();
                break;
            case SEARCH_BEST_FIRST:
                searchStrategy = new BestFirstSearchStrategy();
                break;
            default:
                throw new IllegalArgumentException("Unknown search strategy code: " + m_SearchStrategy);
        }

        // create the tree of splits:
        splitTreeRoot = SplitNode.buildTree(trainingBags, splitStrategy, m_MaxDepth,
                m_MinOccupancy, m_Classifier, searchStrategy, propStrategy);

        // retrain m_classifier with the best attribute:
        Instances propositionalisedTrainingData = SplitNode.propositionaliseDataset(trainingBags, splitTreeRoot, propStrategy);
        m_Classifier.buildClassifier(propositionalisedTrainingData);
        m_propositionalisedDataset = new Instances(propositionalisedTrainingData, 0);
    }
}

// <editor-fold defaultstate="collapsed" desc="===Utility Classes===">

/** For storing a pair (A,B) */
class Pair<A, B> implements Serializable
{
    public final A key;
    public final B value;

    Pair(final A val, final B classVal)
    {
        this.key = val;
        this.value = classVal;
    }

    @Override
    public String toString()
    {
        return "(" + key + ", " + value + ")";
    }
}

/** For storing a comparable pair */
class CompPair<A extends Comparable<A>,B extends Comparable<B>> extends Pair<A,B>
        implements Comparable<CompPair<A,B>>
{
    CompPair(final A key, final B val)
    {
        super(key, val);
    }

    /** @inheritDoc */
    @Override
    public int compareTo(final CompPair<A,B> o)
    {
        int diff = key.compareTo(o.key);
        return diff == 0 ? value.compareTo(o.value) : diff;
    }
}

/** Data structure for storing the tree-building param */
final class TreeBuildingParams implements Serializable
{
    public final int maxDepth;
    public final int maxNodeCount;
    public final int minOccupancy;
    public final Instances trainingBags;
    public final int instCount;
    public final SplitStrategy splitStrategy;
    public final PropositionalisationStrategy propStrategy;
    public final Classifier classifier;

    TreeBuildingParams(final int maxDepth, final int maxNodeCount, final int minOccupancy, final Instances trainingBags,
                       final int instCount, final SplitStrategy splitStrategy,
                       final PropositionalisationStrategy propStrategy, final Classifier classifier)
    {
        this.maxDepth = maxDepth;
        this.maxNodeCount = maxNodeCount;
        this.minOccupancy = minOccupancy;
        this.propStrategy = propStrategy;
        this.classifier = classifier;
        this.trainingBags = trainingBags;
        this.instCount = instCount;
        this.splitStrategy = splitStrategy;
    }
}


// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="===Split Strategies===" >
/**
 * A strategy for generating candidate splits
 */
interface SplitStrategy extends Serializable
{
    /**
     * Generate all candidate splits using the current split strategy
     * @param trainingData The training data (as bags)
     * @param ignore The bitSet of instances to ignore.
     * @return A list of candidate splits
     */
    List<CompPair<Integer, Double>> generateSplitPoints(final Instances trainingData,
                                                    final BitSet ignore);
}

abstract class CenterSplitStrategy implements SplitStrategy
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
            for (Instance inst : bag.relationalValue(AdaProp.REL_INDEX))
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
            for (Instance inst : bag.relationalValue(AdaProp.REL_INDEX))
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
            for (Instance inst : bag.relationalValue(AdaProp.REL_INDEX))
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
class DiscretizedSplitStrategy implements SplitStrategy
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
            for (Instance inst : bag.relationalValue(AdaProp.REL_INDEX))
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
}

// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="===Search Strategies===">

abstract class SearchStrategy implements Serializable
{
    /**
     * Build up the entire tree using this search strategy.
     *
     * @param params The parameters for building the tree.
     * @param instCount The number of (single) instances.
     * @param trainingBags The training dataset (of bags).
     * @return The root of the built tree.
     */
    abstract RootSplitNode buildTree(final TreeBuildingParams params, final int instCount,
                                      final Instances trainingBags) throws Exception;

    /**
     * Build up the default root (a single root, with no specified split).
     *
     * @return the default root.
     */
    protected RootSplitNode buildRoot(final TreeBuildingParams params)
    {
        // root starts at depth=0, with 2 prop-attr (indices 1 and 2).
        RootSplitNode root = RootSplitNode.toRootNode(new SplitNode(1, 2, 0), params.propStrategy);
        root.setNodeCount(1);
        return root;
    }
}

/** Breadth-first search */
class BreadthFirstSearchStrategy extends SearchStrategy
{
    /**
     * Determine whether the node can be expanded.
     *
     * @param node The node to check.
     * @param params Tree building parameters.
     * @param ignoredInst The bitset indicating which instances are to be ignored.
     * @return true if the node can be expanded, false otherwise.
     */
    protected boolean isExpandable(final SplitNode node, final TreeBuildingParams params, final BitSet ignoredInst)
    {
        final int numInstInNode = params.instCount - ignoredInst.cardinality();
        return numInstInNode >= params.minOccupancy && node.curDepth < params.maxDepth;
    }

    @Override /** @inheritDoc */
    public RootSplitNode buildTree(final TreeBuildingParams params, final int instCount, final Instances trainingBags)
            throws Exception
    {
        // build the root:
        int nextPropIndex = 3;
        RootSplitNode root = buildRoot(params);
        final BitSet rootIgnoredInst = new BitSet(instCount);

        if (isExpandable(root, params, rootIgnoredInst)) {
            root.computeBestSplit(params, rootIgnoredInst, root);
        } else {
            return root; // computation is complete.
        }

        // structure the tree into an queue via breadth-first-search:
        final int numAttrPerRegion = params.propStrategy.getNumPropAttrPerRegion();
        int numNodes = 1;
        Queue<Pair<SplitNode,BitSet>> queue = new LinkedList<Pair<SplitNode,BitSet>>();
        queue.add(new Pair<SplitNode, BitSet>(root, rootIgnoredInst));

        while(!queue.isEmpty())
        {
            // take the first node, check if it's children can be expanded further:
            final Pair<SplitNode,BitSet> nodeMapPair = queue.remove();
            final SplitNode node = nodeMapPair.key;
            final BitSet ignoredInst = nodeMapPair.value;
            final int nextDepth = node.curDepth + 1;

            // partition the data-set into left and right sets:
            RegionPartitioner counter = new RegionPartitioner(instCount);
            node.filterDataset(trainingBags, ignoredInst, counter);

            // build the left and right nodes
            node.left = new SplitNode(nextPropIndex, nextPropIndex+numAttrPerRegion, nextDepth);
            if (isExpandable(node.left, params, counter.leftIgnore))
            {
                numNodes++;
                root.setNodeCount(numNodes);
                node.left.computeBestSplit(params, counter.leftIgnore, root);
                nextPropIndex += 2*numAttrPerRegion;
                queue.add(new Pair<SplitNode, BitSet>(node.left, counter.leftIgnore));
            }

            node.right = new SplitNode(nextPropIndex, nextPropIndex+numAttrPerRegion, nextDepth);
            if (isExpandable(node.right, params, counter.rightIgnore))
            {
                numNodes++;
                root.setNodeCount(numNodes);
                node.right.computeBestSplit(params, counter.rightIgnore, root);
                nextPropIndex += 2*numAttrPerRegion;
                queue.add(new Pair<SplitNode, BitSet>(node.right, counter.rightIgnore));
            }
        }

        root.setNodeCount(numNodes);
        return root;
    }
}

/** Best-first search */
class BestFirstSearchStrategy extends SearchStrategy
{
    @Override /** @inheritDoc */
    public RootSplitNode buildTree(final TreeBuildingParams params, final int instCount, final Instances trainingBags)
            throws Exception
    {
        // build the root:
        int nextPropIndex = 3;
        RootSplitNode root = buildRoot(params);
        final BitSet rootIgnoredInst = new BitSet(instCount);

        if (instCount >= params.minOccupancy) {
            root.computeBestSplit(params, rootIgnoredInst, root);
        } else {
            return root; // computation is complete.
        }

        // structure the search by keeping a list of expandable nodes
        //  (i.e. those which have at least one empty child.
        final int numAttrPerRegion = params.propStrategy.getNumPropAttrPerRegion();
        int nodeCount = 1;
        LinkedList<Pair<SplitNode, BitSet>> expandableLeafNodes = new LinkedList<Pair<SplitNode, BitSet>>();

        // initialise the border with the two children of the root.
        root.left = new SplitNode(-1, -1, 1);
        root.right = new SplitNode(-1, -1, 1);
        RegionPartitioner rootCounter = new RegionPartitioner(instCount);
        root.filterDataset(trainingBags, rootIgnoredInst, rootCounter);
        expandableLeafNodes.add(new Pair<SplitNode, BitSet>(root.left, rootCounter.leftIgnore));
        expandableLeafNodes.add(new Pair<SplitNode, BitSet>(root.right, rootCounter.rightIgnore));

        while(!expandableLeafNodes.isEmpty() && nodeCount <= params.maxNodeCount)
        {
            // iterate over all split nodes and find the one with the least error
            Pair<SplitNode, BitSet> bestSplit = null;
            int bestSplitAttrIndex = -1;
            double minErr = Double.MAX_VALUE;

            // adjust counters:
            nodeCount++;
            root.setNodeCount(nodeCount);

            for (Pair<SplitNode, BitSet> nodeMapPair: expandableLeafNodes)
            {
                final SplitNode node = nodeMapPair.key;
                final BitSet ignoredInst = nodeMapPair.value;

                // try expansion:
                node.propLeftIndex = nextPropIndex;
                node.propRightIndex = nextPropIndex + numAttrPerRegion;
                node.computeBestSplit(params, ignoredInst, root);

                final double nodeErr = node.trainingSetError;
                if (nodeErr < minErr)
                {
                    bestSplit = nodeMapPair;
                    bestSplitAttrIndex = node.splitAttrIndex;
                    minErr = nodeErr;
                }

                // reset the node (so that the other nodes in the border are unaffected):
                node.splitAttrIndex = -1;
                node.propLeftIndex = -1;
                node.propRightIndex = -1;
            }

            if (bestSplit == null)
            {
                // no best-split found.
                nodeCount--;
                root.setNodeCount(nodeCount);
                break;
            }
            else
            {
                // "use up" the bestSplit:
                expandableLeafNodes.remove(bestSplit);
                final SplitNode bestNode = bestSplit.key;
                final BitSet bestIgnoredInst = bestSplit.value;
                bestNode.splitAttrIndex = bestSplitAttrIndex;
                bestNode.propLeftIndex = nextPropIndex;
                bestNode.propRightIndex = nextPropIndex + numAttrPerRegion;
                nextPropIndex += 2*numAttrPerRegion;

                // create 2 child nodes:
                final int nextDepth = bestNode.curDepth + 1;
                bestNode.left = new SplitNode(-1, -1, nextDepth);
                bestNode.right = new SplitNode(-1, -1, nextDepth);
                RegionPartitioner counter = new RegionPartitioner(instCount);
                bestNode.filterDataset(params.trainingBags, bestIgnoredInst, counter);

                // add the child nodes to the expandable node border:
                expandableLeafNodes.add(new Pair<SplitNode, BitSet>(bestNode.left, counter.leftIgnore));
                expandableLeafNodes.add(new Pair<SplitNode, BitSet>(bestNode.right, counter.rightIgnore));
            }
        }

        // clear (reset) the remaining leaf nodes:
        for (Pair<SplitNode, BitSet> nodeMapPair: expandableLeafNodes)
        {
            final SplitNode node = nodeMapPair.key;
            node.left = null;
            node.right = null;
            node.propLeftIndex = -1;
            node.propRightIndex = -1;
            node.splitAttrIndex = -1;
        }

        root.setNodeCount(nodeCount);
        return root;
    }
}

// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="===Propositionalisation Strategies===">

/**
 * Represents a method for propositionalisation of a set of instances.
 * This strategy is used to convert a set of instances
 * (i.e. those which fall into a region)
 * into a vector of attributes.
 */
interface PropositionalisationStrategy extends Serializable
{
    /**
     * Propositionalise the bag and place the resultant vector in the
     *  result array, starting at the specified index.
     *
     * @param bag The bag of instances to propositionalise
     * @param result The resultant array to place the results into.
     * @param resultStartIndex The starting location (inclusive) to place the result.
     */
    public void propositionalise(Instances bag, BitSet ignore, double[] result, int resultStartIndex);

    /**
     * @return the number of attributes per region in the propositionalised data-set
     */
    public int getNumPropAttrPerRegion();

    /**
     * @return the attributes of the propositionalised data-set
     */
    public ArrayList<Attribute> getPropAttributes(final int numRegions);
}

class CountBasedPropositionalisationStrategy implements PropositionalisationStrategy
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

class SummaryStatsBasedPropositionalisationStrategy implements PropositionalisationStrategy
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
            for (Instance inst : bag) {
                sumStat.addInstance(inst);
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
        ArrayList<Attribute> attrInfo = new ArrayList<Attribute>(numRegions*propNumAttr + 1);

        for (int region=0; region<numRegions; region++)
        {
            // for each summary stat
            for (int attr=0; attr<propNumAttr; attr++)
            {
                final String attrName = SummaryStatCalculator.SUMMARY_STATS[attr] + " region " + region;
                attrInfo.add(new Attribute(attrName)); // TODO better names for attr?
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
                this.min = attrVal;
            }
        }

        public void storeResults(final double[] result, final int resultStartIndex)
        {
            // store the summary stats
            result[resultStartIndex    ] = this.count;
            result[resultStartIndex + 1] = this.sum;
            result[resultStartIndex + 2] = this.min;
            result[resultStartIndex + 3] = this.max;
            final double avg = (this.count == 0) ? 0 : this.sum / this.count;
            result[resultStartIndex + 4] = avg;
        }
    }
}

/** A (mutable) data structure for keeping track of two counters and an instIndex. */
class RegionPartitioner implements Serializable
{
    public int leftCount;
    public final BitSet leftIgnore;
    public int rightCount;
    public final BitSet rightIgnore;
    public int instIndex;

    RegionPartitioner(int numInst)
    {
        leftCount = 0;
        leftIgnore = new BitSet(numInst);
        rightCount = 0;
        rightIgnore = new BitSet(numInst);
        instIndex = 0;
    }
}
// </editor-fold>

/**
 * Represents a single split point (a node in the adaSplitTree).
 * This Node is either a leaf (left=right=null) or a branch (both left and right are
 *  not null).
 */
class SplitNode implements Serializable
{
    //<editor-fold defaultstate="collapsed" desc="===Init===" >
    static final long serialVersionUID = AdaProp.serialVersionUID + 1000L;

    /** The attribute to split on */
    int splitAttrIndex;

    /** The value of the attribute */
    double splitPoint;

    /** error on training set when building this node */
    double trainingSetError;

    /** node for handling values less than the split point */
    SplitNode left;

    /** greater than or equal to the split point */
    SplitNode right;

    /** The depth of the current node (from the root) */
    final int curDepth;

    /** The index of the propositionalised data-set to store the left-count result */
    int propLeftIndex;

    /** The index of the propositionalised data-set to store the right-count result */
    int propRightIndex;

    SplitNode(final int propLeftIndex, final int propRightIndex, final int splitAttrIndex, final double splitPoint,
              final SplitNode left, final SplitNode right, final int curDepth)
    {
        this.propLeftIndex = propLeftIndex;
        this.propRightIndex = propRightIndex;
        this.splitAttrIndex = splitAttrIndex;
        this.splitPoint = splitPoint;
        this.left = left;
        this.right = right;
        this.curDepth = curDepth;
    }

    /**
     * Create a split node with no current split and no child nodes.
     *
     * @param propLeftIndex The prop-left-index
     * @param propRightIndex The prop-right-index
     * @param curDepth The current depth
     */
    SplitNode(final int propLeftIndex, final int propRightIndex, final int curDepth)
    {
        this(propLeftIndex, propRightIndex, -1, 0, null, null, curDepth);
    }

    @Override
    public String toString()
    {
        return "\tSplit on attr" + splitAttrIndex + " at " + splitPoint +
                ". left=" + propLeftIndex + ", right=" + propRightIndex + ".\n" +
                (left == null ? "" : left.toString()) + (right == null ? "" : right.toString());

    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="===Tree-building===" >

    /**
     * Compute the best splittling hyperplane for this node.
     *
     * First, a set of candidate splits are genereted.
     * Then, the best split (least training-set error) is chosen
     * and set it in the current node.
     *
     * @param params The tree building parameters.
     * @param ignoredInst The instances in the data-set to ignore (because they fall outside the current node).
     * @param root The root of this tree.
     */
    void computeBestSplit(final TreeBuildingParams params, final BitSet ignoredInst, final RootSplitNode root)
            throws Exception
    {
        List<CompPair<Integer, Double>> candidateSplits =
                params.splitStrategy.generateSplitPoints(params.trainingBags, ignoredInst);

        double minErr = Double.MAX_VALUE;
        CompPair<Integer, Double> bestSplit = null;
        for (CompPair<Integer, Double> curSplit : candidateSplits)
        {
            this.splitAttrIndex = curSplit.key;
            this.splitPoint = curSplit.value;
            double err = evaluateCurSplit(params.trainingBags, params.classifier, root, params.propStrategy);
            if (err < minErr)
            {
                minErr = err;
                bestSplit = curSplit;
            }
        }

        // set the best split:
        this.splitAttrIndex = bestSplit.key;
        this.splitPoint = bestSplit.value;
        this.trainingSetError = minErr;
    }

    /**
     * Build up the tree of splits.
     *
     *
     * @param trainingBags the MI bags for use as training data. Must be Non-empty.
     * @param splitStrategy The strategy to split each node.
     * @param maxDepth The maximum depth of the tree.
     * @param minOccupancy The minimum occupancy of each node.
     * @param propStrategy
     * @return The root of the split-tree
     */
    public static RootSplitNode buildTree(Instances trainingBags, final SplitStrategy splitStrategy, final int maxDepth,
                                          final int minOccupancy, final Classifier classifier,
                                          final SearchStrategy searchStrategy,
                                          final PropositionalisationStrategy propStrategy) throws Exception
    {
        // count the number of instances in all the bags:
        int instCount = 0;
        for (Instance bag : trainingBags)
        {
            instCount += bag.relationalValue(AdaProp.REL_INDEX).size();
        }

        final int maxNodeCount = 1 << maxDepth;
        TreeBuildingParams params = new TreeBuildingParams(maxDepth, maxNodeCount, minOccupancy, trainingBags,
                instCount, splitStrategy, propStrategy, classifier);

        return searchStrategy.buildTree(params, instCount, trainingBags);

    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="===Propositionalisation===">
    /**
     * Propositionalise the set of bags into a set of instances.
     * @param bags The MI dataset.
     * @param root The root node of the tree to propositionalise with.
     * @return The propositionalised version of the dataset.
     */
    public static Instances propositionaliseDataset(Instances bags, RootSplitNode root,
                                                    PropositionalisationStrategy propStrategy)
    {
        // build up instance header
        final ArrayList<Attribute> attrInfo = new ArrayList<Attribute>(root.getAttrInfo()); // shallow copy
        attrInfo.add((Attribute) bags.classAttribute().copy()); // class

        Instances propositionalisedDataset = new Instances(bags.relationName() +"-prop", attrInfo, bags.numInstances());
        propositionalisedDataset.setClassIndex(attrInfo.size() - 1);

        // propositionalise each bag and add it to the set
        for (Instance bag : bags) {
            propositionalisedDataset.add(propositionaliseBag(bag, root, propositionalisedDataset, propStrategy));
        }

        return propositionalisedDataset;
    }

    /**
     * Propositionalise the bag into a single instance.
     *
     * @param bag The (MI) bag to propositionalise.
     * @param root The root of the tree of splits.
     * @param propDatasetHeader The header for the propositionalised bags.
     * @return The propositionalised instance.
     */
    public static Instance propositionaliseBag(final Instance bag, final RootSplitNode root,
                                               final Instances propDatasetHeader,
                                               final PropositionalisationStrategy propStrategy)
    {
        final Instances bagInst = bag.relationalValue(AdaProp.REL_INDEX);
        final int numInst = bagInst.size();
        final BitSet ignore = new BitSet(numInst);
        final int numPropAttr = root.getNumPropAttr();

        final double[] attrValues = new double[numPropAttr+1];
        propStrategy.propositionalise(bagInst, ignore, attrValues, 0); // set overall attributes
        attrValues[numPropAttr] = bag.classValue(); // set class val

        // recursively fill in all the attribute values
        if (root.splitAttrIndex >= 0) {
            root.propositionaliseBag(bagInst, attrValues, ignore, propStrategy);
        }

        Instance prop = new DenseInstance(1.0, attrValues);
        prop.setDataset(propDatasetHeader);
        return prop;
    }

    /**
     * Propositionalise the bag by filtering all instances down the tree and counting the number of instances
     *   at each node.
     *
     * @param bag The (mi) bag to propositionalise.
     * @param attrVals The array in which to place the results.
     * @param ignore The bitset of instances (index as per this bag!) to ignore.
     */
    void propositionaliseBag(Instances bag, double[] attrVals, BitSet ignore,
                             PropositionalisationStrategy propStrategy)
    {
        final int numInstances = bag.size();
        final RegionPartitioner counter = new RegionPartitioner(numInstances);

        filterBag(bag, ignore, counter);
        propStrategy.propositionalise(bag, counter.leftIgnore, attrVals, propLeftIndex);
        propStrategy.propositionalise(bag, counter.rightIgnore, attrVals, propRightIndex);

        // recursively fill in the remaining values:
        if (left != null && left.splitAttrIndex >= 0) {
            left.propositionaliseBag(bag, attrVals, counter.leftIgnore, propStrategy);
        }
        if (right != null && right.splitAttrIndex >= 0) {
            right.propositionaliseBag(bag, attrVals,  counter.rightIgnore, propStrategy);
        }
    }

    /**
     * Filter the dataset across the split of this node.
     *
     * @param bags The dataset to filter.
     * @param ignore The bitset of which instances to ignore entirely (i.e. those which lie outside this node).
     * @param counter To keep track of the left and right instance counts.
     */
    void filterDataset(Instances bags, BitSet ignore,  RegionPartitioner counter)
    {
        for (Instance bag : bags) {
            filterBag(bag.relationalValue(AdaProp.REL_INDEX), ignore, counter);
        }
    }

    /**
     * Filter the bag across the split of this node.
     *
     * @param bag The bag of instances to filter.
     * @param ignore The bitset of which instances to ignore entirely (i.e. those which lie outside this node).
     * @param counter To keep track of the left and right instance counts.
     */
    void filterBag(Instances bag, BitSet ignore, RegionPartitioner counter)
    {
        for (Instance inst : bag) {
            filterInst(inst, ignore, counter);
        }
    }

    /**
     * Filter the instance across the split of this node.
     *
     * @param inst The instance to filter.
     * @param ignore The bitset of which instances to ignore entirely (i.e. those which lie outside this node).
     * @param regionPartitioner To partition the region
     */
    private void filterInst(final Instance inst, final BitSet ignore, final RegionPartitioner regionPartitioner)
    {
        if (ignore.get(regionPartitioner.instIndex))
        {
            regionPartitioner.leftIgnore.set(regionPartitioner.instIndex);
            regionPartitioner.rightIgnore.set(regionPartitioner.instIndex);
        }
        else
        {
            // check which partition this instance falls into:
            if (inst.value(splitAttrIndex) <= splitPoint)
            {
                // ignored in the right branch ==> this instance falls in the left-branch.
                regionPartitioner.rightIgnore.set(regionPartitioner.instIndex);
                regionPartitioner.leftCount++;
            }
            else
            {
                regionPartitioner.leftIgnore.set(regionPartitioner.instIndex);
                regionPartitioner.rightCount++;
            }

        }
        regionPartitioner.instIndex++;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="===Split Evaulation===">

    /**
     * A way to evaluate each split point -- TODO
     * NOTE: currently very inefficient - performs the propositionalisation from scratch each time!
     * TODO A far better way is to keep parent's propositionalised dataset and call propBag from the current node.
     */
    public static double evaluateCurSplit(Instances bags, Classifier classifier, RootSplitNode root,
                                          final PropositionalisationStrategy propStrategy) throws Exception
    {
        Instances propDataset = SplitNode.propositionaliseDataset(bags, root, propStrategy);
        classifier.buildClassifier(propDataset);
        Evaluation evaluation = new Evaluation(propDataset);
        evaluation.evaluateModel(classifier, propDataset);
        return evaluation.incorrect();
    }
    //</editor-fold>
}

class RootSplitNode extends SplitNode
{
    /** The number of nodes in the entire tree. */
    private int nodeCount;

    /** Get the nodeCount */
    int getNodeCount() { return this.nodeCount; }

    /** Set the nodeCount and update the attribute-information */
    void setNodeCount(int nodeCount)
    {
        this.nodeCount = nodeCount;
        UpdateAttrInfo();
    }

    /** A list attributes in the propositionalised dataset. */
    private ArrayList<Attribute> attrInfo;

    /** The propositionalisation strategy */
    private final PropositionalisationStrategy propStrategy;

    /** Get the list of attributes */
    ArrayList<Attribute> getAttrInfo() { return this.attrInfo; }

    /** Update the list of attributes, with the new nodeCount */
    private void UpdateAttrInfo()
    {
        attrInfo = propStrategy.getPropAttributes(getNumRegions());
    }

    RootSplitNode(final int propLeftIndex, final int propRightIndex, final int splitAttrIndex, final double splitPoint,
                  final SplitNode left, final SplitNode right, final int curDepth,
                  final PropositionalisationStrategy propStrategy)
    {
        super(propLeftIndex, propRightIndex, splitAttrIndex, splitPoint, left, right, curDepth);
        this.propStrategy = propStrategy;
        attrInfo = new ArrayList<Attribute>();
    }

    /**
     * Convert a node to a RootSplitNode
     */
    public static RootSplitNode toRootNode(SplitNode node, PropositionalisationStrategy propStrategy)
    {
        return new RootSplitNode(node.propLeftIndex, node.propRightIndex,
                node.splitAttrIndex, node.splitPoint, node.left, node.right, node.curDepth, propStrategy);
    }

    private int getNumRegions()
    {
        // 2 regions per node, plus an extra one for the entire bag
        return 2*nodeCount+1;
    }

    public int getNumPropAttr()
    {
        return getNumRegions() * propStrategy.getNumPropAttrPerRegion();
    }
}