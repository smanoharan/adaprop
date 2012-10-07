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
 * Author: Siva Manoharan
 */
public class AdaProp extends SingleClassifierEnhancer
        implements MultiInstanceCapabilitiesHandler, OptionHandler
{
    /**
     * For serialization:
     *  format: 1[dd][mm][yyyy]00..0[digit revision number]L
     */
    static final long serialVersionUID = 1200920120000013L;

    /** The tree of splits */
    protected RootSplitNode splitTreeRoot;

    /** Contains the bags as propositionalised instances */
    protected Instances m_propositionalisedDataset;

    /** The instIndex of the relational attribute in the bag instance */
    public static final int REL_INDEX = 1;

    //<editor-fold defaultstate="collapsed" desc="===Option Handling===">
    private static final int SPLIT_MEAN = 1;
    private static final int SPLIT_MEDIAN = 2;
    private static final int SPLIT_DISCRETIZED = 3;
    private static final int SPLIT_RANGE = 4;
    private static final int DEFAULT_SPLIT_STRATEGY = SPLIT_MEAN;
    private static final int DEFAULT_MAX_DEPTH = 3;
    private static final int DEFAULT_MIN_OCCUPANCY = 5;

    public static final Tag [] SPLIT_STRATEGIES =
    {
        new Tag(SPLIT_MEAN, "Split by the mean value of an attribute"),
        new Tag(SPLIT_MEDIAN, "Split by the median value of an attribute"),
        new Tag(SPLIT_DISCRETIZED, "Split by any value of an attribute where class value changes"),
        new Tag(SPLIT_RANGE, "Split by the midpoint of the range of the values of an attribute")
    };

    /** The id of the instance-space splitting strategy to use */
    protected int m_SplitStrategy = DEFAULT_SPLIT_STRATEGY;

    /** The maximum depth of the tree of splits (0 for unlimited) */
    protected int m_MaxDepth = DEFAULT_MAX_DEPTH;

    /** The minimum occupancy of each leaf node in the tree */
    protected int m_MinOccupancy = DEFAULT_MIN_OCCUPANCY;

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
                "S", 1, "-S <num>"));

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
        String splitStrategyStr = Utils.getOption('S', options);
        this.setSplitStrategy(new SelectedTag(Integer.parseInt(splitStrategyStr), SPLIT_STRATEGIES));

        String maxDepthStr = Utils.getOption("maxDepth", options);
        this.setMaxDepth(maxDepthStr.length() == 0 ? DEFAULT_MAX_DEPTH : Integer.parseInt(maxDepthStr));

        String minOccStr = Utils.getOption("minOcc", options);
        this.setMinOccupancy(minOccStr.length() == 0 ? DEFAULT_MAX_DEPTH :Integer.parseInt(minOccStr));

        super.setOptions(options);
    }

    /** @inheritDoc */
    @SuppressWarnings({ "unchecked", "rawtypes" })
    public String[] getOptions()
    {
        Vector result = new Vector();

        result.add("-S");
        result.add("" + m_SplitStrategy);
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
                SplitNode.propositionaliseBag(newBag, splitTreeRoot, m_propositionalisedDataset);

        // use the base classifier for prediction.
        return m_Classifier.distributionForInstance(propositionalisedTrainingData);
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

        // create the tree of splits:
        splitTreeRoot = SplitNode.buildTree(trainingBags, splitStrategy, m_MaxDepth, m_MinOccupancy, m_Classifier);

        // retrain m_classifier with the best attribute:
        Instances propositionalisedTrainingData = SplitNode.propositionaliseDataset(trainingBags, splitTreeRoot);
        m_Classifier.buildClassifier(propositionalisedTrainingData);
        m_propositionalisedDataset = new Instances(propositionalisedTrainingData, 0);
    }
}

/** For storing a pair: value (double) and class (double) */
class Pair<A extends Comparable<A>,B extends Comparable<B>> implements Comparable<Pair<A,B>>
{
    public final A key;
    public final B value;

    Pair(final A val, final B classVal)
    {
        this.key = val;
        this.value = classVal;
    }

    /** @inheritDoc */
    @Override
    public int compareTo(final Pair<A,B> o)
    {
        int diff = key.compareTo(o.key);
        return diff == 0 ? value.compareTo(o.value) : diff;
    }

    @Override
    public String toString()
    {
        return "(" + key + ", " + value + ")";
    }
}

/**
 * A strategy for generating candidate splits
 */
interface SplitStrategy
{
    /**
     * Generate all candidate splits using the current split strategy
     * @param trainingData The training data (as bags)
     * @param ignore The bitSet of instances to ignore.
     * @return A list of candidate splits
     */
    List<Pair<Integer, Double>> generateSplitPoints(final Instances trainingData,
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
    public List<Pair<Integer, Double>> generateSplitPoints(
            final Instances trainingData, final BitSet ignore)
    {
        List<Pair<Integer, Double>> splits = new ArrayList<Pair<Integer, Double>>(numAttr);

        for(int attr=0; attr<numAttr; attr++)
        {
            splits.add(new Pair<Integer, Double>(attr, findCenter(trainingData, attr, ignore)));
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
        List<Pair<Double,Double>> vals = new ArrayList<Pair<Double,Double>>();
        int index = 0;
        for (Instance bag : trainingData)
        {
            for (Instance inst : bag.relationalValue(AdaProp.REL_INDEX))
            {
                if (!ignore.get(index++))
                {
                    vals.add(new Pair<Double,Double>(inst.value(attrIndex), bag.classValue()));
                }
            }
        }

        Collections.sort(vals);

        // iterate through the list, finding class-boundaries
        ArrayList<Double> splits = new ArrayList<Double>();
        Pair<Double, Double> last = vals.get(0);
        final int size = vals.size();
        for(int i=1; i<size; i++)
        {
            Pair<Double, Double> cur = vals.get(i);
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
    public List<Pair<Integer, Double>> generateSplitPoints(
            final Instances trainingData, final BitSet ignore)
    {
        List<Pair<Integer, Double>> splits = new ArrayList<Pair<Integer, Double>>(numAttr);

        for(int attr=0; attr<numAttr; attr++)
        {
            for (double split : findDiscretizedSplits(trainingData, attr, ignore))
            {
                splits.add(new Pair<Integer, Double>(attr, split));
            }
        }

        return splits;
    }
}

/**
 * A way to evaluate each split point
 */
interface SplitPointEvaluator
{
    /**
     * Evaluate the accuracy when splitting the trainingData on the specified attribute,
     *  using the specified split point.
     * @param trainingData The training instances
     * @param splitAttrIndex The attribute to split on
     * @param splitPoint The split value
     * @return The classification error.
     */
    double evaluateSplit(Instances trainingData, int splitAttrIndex, double splitPoint);
}


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

    /** The instIndex of the attribute to which node corresponds */
    int propAttributeIndex;

    /** Data structure for storing the tree-building param */
    final static class TreeBuildingParams
    {
        public final int maxDepth;
        public final int maxNodeCount;
        public final int minOccupancy;
        public final Instances trainingBags;
        public final int instCount;
        public final SplitStrategy splitStrategy;
        public final Classifier classifier;

        TreeBuildingParams(final int maxDepth, final int maxNodeCount, final int minOccupancy, final Instances trainingBags,
                           final int instCount, final SplitStrategy splitStrategy, final Classifier classifier)
        {
            this.maxDepth = maxDepth;
            this.maxNodeCount = maxNodeCount;
            this.minOccupancy = minOccupancy;
            this.classifier = classifier;
            this.trainingBags = trainingBags;
            this.instCount = instCount;
            this.splitStrategy = splitStrategy;
        }
    }

    SplitNode(final int propositionalisedAttributeIndex, final int splitAttrIndex, final double splitPoint,
              final SplitNode left, final SplitNode right, final int curDepth)
    {
        this.propAttributeIndex = propositionalisedAttributeIndex;
        this.splitAttrIndex = splitAttrIndex;
        this.splitPoint = splitPoint;
        this.left = left;
        this.right = right;
        this.curDepth = curDepth;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="===Tree-building===" >
    /**
     * Creates a new leaf node.
     *
     * @param propAttrIndex The propositionalisedAttributeIndex
     * @param curDepth depth of this leaf node.
     * @return A leaf node.
     */
    static SplitNode newLeafNode(int propAttrIndex, int curDepth)
    {
        return new SplitNode(propAttrIndex, -1, 0, null, null, curDepth);
    }

    /**
     * Out of the set of candidate splits, find the split which results in the least (training-set) error and set it
     *  in the current node.
     *
     * @param candidateSplits The set of splits to try.
     * @param bags The training data.
     * @param classifier The classifier to evaluate with.
     */
    private void setBestSplit(final List<Pair<Integer, Double>> candidateSplits, final Instances bags,
                              final Classifier classifier, final RootSplitNode root) throws Exception
    {
        double minErr = Double.MAX_VALUE;
        Pair<Integer, Double> bestSplit = null;
        for (Pair<Integer, Double> curSplit : candidateSplits)
        {
            this.splitAttrIndex = curSplit.key;
            this.splitPoint = curSplit.value;
            double err = evaluateCurSplit(bags, classifier, root);
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
     * Attempt to expand this leaf node.
     *
     * @param params The tree building parameters.
     * @param ignore The instances in the dataset to ignore (because they fall outside the current node).
     * @param root The root of this tree.
     * @return Whether this node was expanded (false iff this node remains a leaf).
     * @throws Exception
     */
    boolean expand(final TreeBuildingParams params, final BitSet ignore, final RootSplitNode root,
                   final int propIndex) throws Exception
    {
        // only expand if the stopping condition is not met
        if (this.curDepth < params.maxDepth && params.instCount - ignore.cardinality() >= params.minOccupancy)
        {
            this.left = newLeafNode(propIndex, this.curDepth + 1);
            this.right = newLeafNode(propIndex+1, this.curDepth + 1);
            root.setNodeCount(root.getNodeCount()+2);
            List<Pair<Integer, Double>> candidateSplits = params.splitStrategy.generateSplitPoints(params.trainingBags, ignore);
            this.setBestSplit(candidateSplits, params.trainingBags, params.classifier, root);
            return true;
        }
        else return false;
    }

    // Best First Search:
    private static RootSplitNode buildTreeViaBestFirstSearch(final TreeBuildingParams params, final int instCount,
                                                             final Instances trainingBags) throws Exception
    {
        // build the root:
        int propIndex = 0;
        RootSplitNode root = RootSplitNode.toRootNode(newLeafNode(propIndex++, 0));
        root.setNodeCount(1);
        root.expand(params, new BitSet(instCount), root, propIndex);
        propIndex += 2;

        // structure the search by keeping a list of expandable (but currently leaf) nodes.
        LinkedList<SplitNode> expandableLeafNodes = new LinkedList<SplitNode>();
        ArrayList<BitSet> ignoreList = new ArrayList<BitSet>(); // for storing the ignore-bitsets.
        ignoreList.add(new BitSet(instCount));
        ignoreList.add(null); ignoreList.add(null); // allocate 2 more slots in the array list for the child nodes
        expandableLeafNodes.add(root);
        int nodeCount = 0;

        while(!expandableLeafNodes.isEmpty() && nodeCount < params.maxNodeCount)
        {
            // iterate over all split nodes and find the one with the least error
            SplitNode bestSplit = null;
            BitSet bestSplitLeftIgnore = null;
            BitSet bestSplitRightIgnore = null;
            double minErr = Double.MAX_VALUE;

            for (SplitNode node : expandableLeafNodes)
            {
                // try splitting here:
                // partition the dataset into left and right sets:
                BitSet leftIgnore = new BitSet(instCount);
                BitSet rightIgnore = new BitSet(instCount);
                LeftRightCounter counter = new LeftRightCounter();
                node.filterDataset(trainingBags, ignoreList.get(node.propAttributeIndex), leftIgnore, rightIgnore, counter);

                // build the left and right nodes
                double trainingSetError = 0;
                int childNodeCount = 0;
                if (node.left == null)
                {
                    break;
                }
                if (node.left.expand(params, leftIgnore, root, propIndex))
                {
                    childNodeCount++;
                    propIndex += 2;
                    trainingSetError += node.left.trainingSetError;
                }

                if (node.right.expand(params, rightIgnore, root, propIndex))
                {
                    childNodeCount++;
                    propIndex += 2;
                    trainingSetError += node.right.trainingSetError;
                }

                if (childNodeCount > 0 && trainingSetError < minErr*childNodeCount)
                {
                    bestSplit = node;
                    bestSplitLeftIgnore = leftIgnore;
                    bestSplitRightIgnore = rightIgnore;
                    minErr = trainingSetError / childNodeCount;
                }

                // reset propIndex:
                propIndex -= (2*childNodeCount);
            }

            // "use up" the bestSplit:
            if (bestSplit == null)
            {
                // no best-split found.
                break;
            }
            else
            {
                expandableLeafNodes.remove(bestSplit);
                if (bestSplit.left != null)
                {
                    nodeCount++;
                    propIndex += 2;
                    ignoreList.add(null); ignoreList.add(null);
                    ignoreList.set(bestSplit.left.propAttributeIndex, bestSplitLeftIgnore);
                    expandableLeafNodes.add(bestSplit.left);
                }

                if (bestSplit.right != null)
                {
                    nodeCount++;
                    propIndex += 2;
                    ignoreList.add(null); ignoreList.add(null);
                    ignoreList.set(bestSplit.right.propAttributeIndex, bestSplitRightIgnore);
                    expandableLeafNodes.add(bestSplit.right);
                }
            }
        }

        // clear the remaining leaf nodes:
        for (SplitNode node : expandableLeafNodes)
        {
            node.left = null;
            node.right = null;
            node.splitAttrIndex = -1;
        }

        return root;
    }


    private static RootSplitNode buildTreeViaBreadthFirstSearch(final TreeBuildingParams params, final int instCount,
                                                                final Instances trainingBags)
            throws Exception
    {

        // build the root:
        int propIndex = 0;
        RootSplitNode root = RootSplitNode.toRootNode(newLeafNode(propIndex++, 0));
        root.setNodeCount(1);
        root.expand(params, new BitSet(instCount), root, propIndex);
        propIndex += 2;

        // structure the tree into an arraylist via breadth-first-search:
        Queue<SplitNode> nodeQueue = new LinkedList<SplitNode>();
        ArrayList<BitSet> ignoreList = new ArrayList<BitSet>(); // for storing the ignore-bitsets.
        ignoreList.add(new BitSet(instCount));
        ignoreList.add(null); ignoreList.add(null); // allocate 2 more slots in the array list for the child nodes
        nodeQueue.add(root);

        while(!nodeQueue.isEmpty())
        {
            // take the first node, check if it's children require further splitting:
            SplitNode node = nodeQueue.remove();
            if ( !node.isLeaf() )
            {
                // partition the dataset into left and right sets:
                BitSet leftIgnore = new BitSet(instCount);
                BitSet rightIgnore = new BitSet(instCount);
                LeftRightCounter counter = new LeftRightCounter();
                node.filterDataset(trainingBags, ignoreList.get(node.propAttributeIndex), leftIgnore, rightIgnore, counter);

                // build the left and right nodes
                if (node.left.expand(params, leftIgnore, root, propIndex))
                {
                    propIndex += 2;
                    ignoreList.add(null);
                    ignoreList.add(null);
                    nodeQueue.add(node.left);
                    ignoreList.set(node.left.propAttributeIndex, leftIgnore);
                }

                if (node.right.expand(params, rightIgnore, root, propIndex))
                {
                    propIndex += 2;
                    ignoreList.add(null);
                    ignoreList.add(null);
                    nodeQueue.add(node.right);
                    ignoreList.set(node.right.propAttributeIndex, rightIgnore);
                }
            }
        }

        return root;
    }

    /**
     * Build up the tree of splits.
     *
     * @param trainingBags the MI bags for use as training data. Must be Non-empty.
     * @param splitStrategy The strategy to split each node.
     * @param maxDepth The maximum depth of the tree.
     * @param minOccupancy The minimum occupancy of each node.
     * @return The root of the split-tree
     */
    public static RootSplitNode buildTree(Instances trainingBags, final SplitStrategy splitStrategy, final int maxDepth,
                                      final int minOccupancy, final Classifier classifier) throws Exception
    {
        // count the number of instances in all the bags:
        int instCount = 0;
        for (Instance bag : trainingBags)
        {
            instCount += bag.relationalValue(AdaProp.REL_INDEX).size();
        }

        // TODO make param
        final boolean doBestFirst = true;

        TreeBuildingParams params;
        if (doBestFirst)
        {
            params = new TreeBuildingParams(Integer.MAX_VALUE, 1 << maxDepth, minOccupancy, trainingBags,
                    instCount, splitStrategy, classifier);
            return buildTreeViaBestFirstSearch(params, instCount, trainingBags);
        }
        else
        {
            params = new TreeBuildingParams(maxDepth, Integer.MAX_VALUE, minOccupancy, trainingBags,
                    instCount, splitStrategy, classifier);
            return buildTreeViaBreadthFirstSearch(params, instCount, trainingBags);
        }


    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="===Propositionalisation===">
    /**
     * Propositionalise the set of bags into a set of instances.
     * @param bags The MI dataset.
     * @param root The root node of the tree to propositionalise with.
     * @return The propositionalised version of the dataset.
     */
    public static Instances propositionaliseDataset(Instances bags, RootSplitNode root)
    {
        // build up instance header
        final int numAttr = root.getNodeCount();
        final ArrayList<Attribute> attrInfo = new ArrayList<Attribute>(root.getAttrInfo()); // shallow copy
        attrInfo.add((Attribute) bags.classAttribute().copy()); // class

        Instances propositionalisedDataset = new Instances(bags.relationName() +"-prop", attrInfo, bags.numInstances());
        propositionalisedDataset.setClassIndex(numAttr);

        // propositionalise each bag and add it to the set
        for (Instance bag : bags)
        {
            propositionalisedDataset.add(propositionaliseBag(bag, root, propositionalisedDataset));
        }

        return propositionalisedDataset;
    }

    /**
     * Propositionalise the bag into a single instance.
     *
     * @param bag The (MI) bag to propositionalise.
     * @param root The root of the tree of splits.
     * @param propositionalisedDataset The header for the data-instances.
     * @return The propositionalised instance.
     */
    public static Instance propositionaliseBag(final Instance bag, final RootSplitNode root,
                                               final Instances propositionalisedDataset)
    {
        int numInst = bag.relationalValue(AdaProp.REL_INDEX).size();
        final double[] attrValues = new double[root.getNodeCount()+1];
        attrValues[0] = numInst; // set root count
        attrValues[root.getNodeCount()] = bag.classValue(); // set class val

        // recursively fill in all the attribute values
        root.propositionaliseBag(bag, attrValues, new BitSet(numInst));

        Instance prop = new DenseInstance(1.0, attrValues);
        prop.setDataset(propositionalisedDataset);
        return prop;
    }

    /** A (mutable) data structure for keeping track of two counters and an instIndex. */
    static class LeftRightCounter
    {
        public int leftCount;
        public int rightCount;
        public int instIndex;

        LeftRightCounter()
        {
            leftCount = 0;
            rightCount = 0;
            instIndex = 0;
        }
    }

    /**
     * Propositionalise the bag by filtering all instances down the tree and counting the number of instances
     *   at each node.
     *
     * @param bag The (mi) bag to propositionalise.
     * @param attrVals The array in which to place the results.
     * @param ignore The bitset of instances (index as per this bag!) to ignore.
     */
    void propositionaliseBag(Instance bag, double[] attrVals, BitSet ignore)
    {
        final int numInstances = bag.relationalValue(AdaProp.REL_INDEX).size();

        if (!this.isLeaf()) // only proceed if not a leaf;
        {
            BitSet leftIgnore = new BitSet(numInstances);
            BitSet rightIgnore = new BitSet(numInstances);

            LeftRightCounter counter = new LeftRightCounter();
            filterBag(bag, ignore, leftIgnore, rightIgnore, counter);
            attrVals[left.propAttributeIndex] = counter.leftCount;
            attrVals[right.propAttributeIndex] = counter.rightCount;

            // recursively fill in the remaining values:
            left.propositionaliseBag(bag, attrVals, leftIgnore);
            right.propositionaliseBag(bag, attrVals,  rightIgnore);
        }
    }

    /**
     * Filter the dataset across the split of this node.
     *
     * @param bags The dataset to filter.
     * @param ignore The bitset of which instances to ignore entirely (i.e. those which lie outside this node).
     * @param leftIgnore The resultant bitset of the left subtree.
     * @param rightIgnore The resultant bitset of the right subtree.
     * @param counter To keep track of the left and right instance counts.
     */
    void filterDataset(Instances bags, BitSet ignore, BitSet leftIgnore, BitSet rightIgnore, LeftRightCounter counter)
    {
        for (Instance bag : bags)
        {
            filterBag(bag, ignore, leftIgnore, rightIgnore, counter);
        }
    }

    /**
     * Filter the bag across the split of this node.
     *
     * @param bag The bag to filter.
     * @param ignore The bitset of which instances to ignore entirely (i.e. those which lie outside this node).
     * @param leftIgnore The resultant bitset of the left subtree.
     * @param rightIgnore The resultant bitset of the right subtree.
     * @param counter To keep track of the left and right instance counts.
     */
    void filterBag(Instance bag, BitSet ignore, BitSet leftIgnore, BitSet rightIgnore, LeftRightCounter counter)
    {
        for (Instance inst : bag.relationalValue(AdaProp.REL_INDEX))
        {
            filterInst(inst, ignore, leftIgnore, rightIgnore, counter);
        }
    }

    /**
     * Filter the instance across the split of this node.
     *
     * @param inst The instance to filter.
     * @param ignore The bitset of which instances to ignore entirely (i.e. those which lie outside this node).
     * @param leftIgnore The resultant bitset of the left subtree.
     * @param rightIgnore The resultant bitset of the right subtree.
     * @param counter To keep track of the left and right instance counts.
     */
    private void filterInst(Instance inst, BitSet ignore, BitSet leftIgnore, BitSet rightIgnore, LeftRightCounter counter)
    {
        if (ignore.get(counter.instIndex))
        {
            leftIgnore.set(counter.instIndex);
            rightIgnore.set(counter.instIndex);
        }
        else
        {
            // check which partition this instance falls into:
            if (inst.value(splitAttrIndex) <= splitPoint)
            {
                // ignored in the right branch ==> this instance falls in the left-branch.
                rightIgnore.set(counter.instIndex);
                counter.leftCount++;
            }
            else
            {
                leftIgnore.set(counter.instIndex);
                counter.rightCount++;
            }
        }
        counter.instIndex++;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="===Split Evaulation===">

    /**
     * A way to evaluate each split point -- TODO
     * NOTE: currently very inefficient - performs the propositionalisation from scratch each time!
     * TODO A far better way is to keep parent's propositionalised dataset and call propBag from the current node.
     */
    public static double evaluateCurSplit(Instances bags, Classifier classifier, RootSplitNode root) throws Exception
    {
        Instances propDataset = SplitNode.propositionaliseDataset(bags, root);
        classifier.buildClassifier(propDataset);
        Evaluation evaluation = new Evaluation(propDataset);
        evaluation.evaluateModel(classifier, propDataset);
        return evaluation.incorrect();
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="===common member functions====">
    /**
     * Returns whether or not this node is a leaf.
     * @return true iff this node is a leaf.
     */
    public boolean isLeaf()
    {
        return this.splitAttrIndex == -1;
    }

    @Override
    public String toString()
    {
        if (left == null)
        {
            // this is a leaf:
            return "\t["+ propAttributeIndex + "] leaf.\n";
        }
        else
        {
            return "\t[" + propAttributeIndex + "] Split on attr" +
                    splitAttrIndex + " at " + splitPoint + ". left=" +
                    left.propAttributeIndex + ", right=" +
                    right.propAttributeIndex + ".\n" +
                    left.toString() + right.toString();
        }
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

    /** Get the list of attributes */
    ArrayList<Attribute> getAttrInfo() { return this.attrInfo; }

    /** Update the list of attributes, with the new nodeCount */
    private void UpdateAttrInfo()
    {
        attrInfo = new ArrayList<Attribute>();
        for (int i=0; i<nodeCount; i++)
        {
            attrInfo.add(new Attribute("region " + i)); // TODO better names for attr?
        }
    }

    RootSplitNode(final int propositionalisedAttributeIndex, final int splitAttrIndex, final double splitPoint,
                  final SplitNode left, final SplitNode right, final int curDepth)
    {
        super(propositionalisedAttributeIndex, splitAttrIndex, splitPoint, left, right, curDepth);
        attrInfo = new ArrayList<Attribute>();
    }

    /**
     * Convert a node to a RootSplitNode
     */
    public static RootSplitNode toRootNode(SplitNode node)
    {
        return new RootSplitNode(node.propAttributeIndex, node.splitAttrIndex, node.splitPoint,
                node.left, node.right, node.curDepth);
    }
}