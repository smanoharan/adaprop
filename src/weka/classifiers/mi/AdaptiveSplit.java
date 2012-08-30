package weka.classifiers.mi;

import weka.classifiers.Classifier;
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
public class AdaptiveSplit extends SingleClassifierEnhancer
        implements MultiInstanceCapabilitiesHandler, OptionHandler
{
    /**
     * For serialization:
     *  format: 1[dd][mm][yyyy]00..0[digit revision number]L
     */
    static final long serialVersionUID = 1280820120000009L;

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
    private static final int DEFAULT_MAX_DEPTH = 3;
    private static final int DEFAULT_MIN_OCCUPANCY = 2;

    public static final Tag [] SPLIT_STRATEGIES =
    {
        new Tag(SPLIT_MEAN, "Split by the mean value of an attribute"),
        new Tag(SPLIT_MEDIAN, "Split by the median value of an attribute"),
        new Tag(SPLIT_DISCRETIZED, "Split by any value of an attribute where class value changes")
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
    // ==================================================================================

    /** The tree of splits */
    protected SplitNode splitTreeRoot;

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
//            String valString = Utils.getOption('M', options);
//            Type newValue = isValid(valString) ? parse(valString) : defaultValue;
//            setProperty(newValue);
//            OR
//            setB(Utils.getFlag('B', options));

        // split strategy
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

        // find the split & evaluate strategies:
        SplitPointEvaluator spe = null; // TODO
        SplitStrategy splitStrategy = null;
        switch (m_SplitStrategy)
        {
            case SPLIT_MEDIAN:
                splitStrategy = new MedianSplitStrategy(numAttr);
                break;
            case SPLIT_DISCRETIZED:
                splitStrategy = new DiscretizedSplitStrategy(numAttr);
                break;
            default:
                splitStrategy = new MeanSplitStrategy(numAttr);
                break;
        }

        // create the tree of splits:
        splitTreeRoot = SplitNode.buildTree(
                trainingBags, splitStrategy, m_MaxDepth, m_MinOccupancy, m_Classifier);

        // retrain m_classifier with the best attribute:
        Instances propositionalisedTrainingData =
                SplitNode.propositionaliseDataset(trainingBags, splitTreeRoot);
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
     * Find the mean of all instances in trainingData for the attribute at index=attrIndex.
     * Assumes that the attribute is numeric. <== TODO may cause problems
     *
     * @param trainingData The dataset of mi-bags
     * @param attrIndex The index of the attribute to find the mean for
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
            for (Instance inst : bag.relationalValue(AdaptiveSplit.REL_INDEX))
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
     * Find the median of all instances in trainingData for the attribute at index=attrIndex.
     * Assumes that the attribute is numeric. <== TODO may cause problems
     *
     * @param trainingData The dataset of mi-bags
     * @param attrIndex The index of the attribute to find the mean for
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
            for (Instance inst : bag.relationalValue(AdaptiveSplit.REL_INDEX))
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
            for (Instance inst : bag.relationalValue(AdaptiveSplit.REL_INDEX))
            {
                if (!ignore.get(index++))
                {
                    vals.add(new Pair(inst.value(attrIndex), bag.classValue()));
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
 *
 *  TODO - represent using array instead (instead of linked-tree)
 *      similar to min-heap array representation
 *      (may have empty slots, but that shouldn't be a problem)
 *      Better yet: serialise the process with a queue --> solves pre-order problem.
 */
class SplitNode implements Serializable
{
    static final long serialVersionUID = AdaptiveSplit.serialVersionUID + 1000L;

    /** The attribute to split on */
    private final int splitAttrIndex;

    /** The value of the attribute */
    private final double splitPoint;

    /** node for handling values less than the split point */
    private final SplitNode left;

    /** greater than or equal to the split point */
    private final SplitNode right;

    /** The number of nodes in this tree and it's subtrees */
    private final int nodeCount;

    /** The index of the attribute to which node corresponds */
    private int propositionalisedAttributeIndex;

    /**
     * Find the best split point on the given dataset.
     */
    public SplitNode(SplitStrategy splitStrategy, Instances bags, int maxDepth,
                     int minOccupancy, BitSet ignore, int flattenedCount,
                     Classifier classifier )
    {
        // stopping condition:
        if (maxDepth <= 0 || flattenedCount - ignore.cardinality() < minOccupancy)
        {
            splitAttrIndex = -1;
            splitPoint = 0;
            left = null;
            right = null;
            nodeCount = 1;
        }
        else
        {
            List<Pair<Integer, Double>> candidateSplits
                    = splitStrategy.generateSplitPoints(bags, ignore);

            // find the best split (least err)
            double minErr = Double.MAX_VALUE;
            Pair<Integer, Double> bestSplit = null;
            for (Pair<Integer, Double> curSplit : candidateSplits)
            {
                double err = evaluateSplit(bags, curSplit.key, curSplit.value, classifier);

                if (err < minErr)
                {
                    minErr = err;
                    bestSplit = curSplit;
                }
            }

            // assign best values
            splitAttrIndex = bestSplit.key;
            splitPoint = bestSplit.value;

            // recursive split:
            //  partition into left & right
            BitSet leftIgnore = new BitSet(flattenedCount);
            BitSet rightIgnore = new BitSet(flattenedCount);
            partitionDataset(bags, ignore, leftIgnore, rightIgnore);

            left = new SplitNode(splitStrategy, bags, maxDepth - 1, minOccupancy,
                    leftIgnore, flattenedCount, classifier);
            right = new SplitNode(splitStrategy, bags, maxDepth - 1, minOccupancy,
                    rightIgnore, flattenedCount, classifier);
            nodeCount = left.nodeCount + right.nodeCount + 1;
        }
    }

    // returns true if not a leaf
    private boolean navigateSplit(Instance inst, BitSet ignore, BitSet leftIgnore,
                                  BitSet rightIgnore, int index)
    {
        if (ignore.get(index))
        {
            leftIgnore.set(index);
            rightIgnore.set(index);
            return false;
        }
        else
        {
            // check which partition this instance falls into:
            if (inst.value(splitAttrIndex) <= splitPoint)
            {
                rightIgnore.set(index);
            }
            else
            {
                leftIgnore.set(index);
            }
            return true;
        }
    }

    /** Places the result into left/right ignore bitsets. Returns num instances */
    int partitionDataset(Instances bags, BitSet ignore, BitSet leftIgnore,
                         BitSet rightIgnore)
    {
        int index = 0;
        int count = 0;
        for (Instance bag : bags)
        {
            for (Instance inst : bag.relationalValue(AdaptiveSplit.REL_INDEX))
            {
                if (navigateSplit(inst, ignore, leftIgnore, rightIgnore, index++))
                {
                    count++;
                }
            }
        }

        return count;
    }

    // TODO
    int partitionBag(Instance bag, BitSet ignore, BitSet leftIgnore, BitSet rightIgnore)
    {
        int index = 0;
        int count = 0;
        for (Instance inst : bag.relationalValue(AdaptiveSplit.REL_INDEX))
        {
            if (navigateSplit(inst, ignore, leftIgnore, rightIgnore, index++))
            {
                count++;
            }
        }
        return count;
    }

    /**
     * TODO
     *
     * places the results (incl subtrees) in the appropriate slot in the attrVals.
     * Thus this fn is recursive.
     *
     * can only be used at test time (not training time)
     *
     */
    private void nodePropositionaliseBag(Instance bag, double[] attrVals, BitSet ignore)
    {
        final int numInstances = bag.relationalValue(AdaptiveSplit.REL_INDEX).size();
        if (this.nodeCount == 1) // is leaf
        {
            attrVals[propositionalisedAttributeIndex] = numInstances - ignore.cardinality();
        }
        else
        {
            BitSet leftIgnore = new BitSet(numInstances);
            BitSet rightIgnore = new BitSet(numInstances);
            attrVals[propositionalisedAttributeIndex] =
                    partitionBag(bag, ignore, leftIgnore, rightIgnore);
            left.nodePropositionaliseBag(bag, attrVals, leftIgnore);
            right.nodePropositionaliseBag(bag, attrVals,  rightIgnore);
        }
    }

    /**
     * Build up the tree of splits, using the given training bags.
     *
     * TODO
     *
     * @param trainingBags the MI bags for use as training data. Must be Non-empty.
     * @param splitStrategy
     * @param maxDepth
     * @param minOccupancy
     * @return The root of the split-tree
     */
    public static SplitNode buildTree(Instances trainingBags,
        final SplitStrategy splitStrategy, final int maxDepth, final int minOccupancy,
        final Classifier classifier)
    {
        // count the number of instances:
        int flattenedCount = 0;
        for (Instance bag : trainingBags)
        {
            flattenedCount += bag.relationalValue(AdaptiveSplit.REL_INDEX).size();
        }

        BitSet ignore = new BitSet(flattenedCount);

        // build the entire tree
        SplitNode root = new SplitNode(splitStrategy, trainingBags, maxDepth,
                minOccupancy, ignore, flattenedCount, classifier);

        // structure the tree into an arraylist via bfs:
        int index = 0;
        Queue<SplitNode> nodeQueue = new LinkedList<SplitNode>();
        nodeQueue.add(root);

        while(!nodeQueue.isEmpty())
        {
            SplitNode node = nodeQueue.remove();
            node.propositionalisedAttributeIndex = index++;

            if (node.left != null) nodeQueue.add(node.left);
            if (node.right != null) nodeQueue.add(node.right);
        }

        return root;
    }

    public static Instances propositionaliseDataset(Instances bags, SplitNode root)
    {
        // construct attribute header (and instances header):
        // TODO this should only be done once (after training is complete)
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
        for (int i=0;i<root.nodeCount;i++)
        {
            attInfo.add(new Attribute("region " + i)); // TODO better names for attr
        }
        attInfo.add((Attribute) bags.classAttribute().copy()); // class
        Instances propositionalisedDataset = new Instances("prop", attInfo, bags.numInstances());
        propositionalisedDataset.setClassIndex(root.nodeCount);

        // propositionalise each bag and add it to the set
        for (Instance bag : bags)
        {
            propositionalisedDataset.add(propositionaliseBag(bag, root, propositionalisedDataset));
        }

        return propositionalisedDataset;
    }

    public static Instance propositionaliseBag(final Instance bag, final SplitNode root,
                                               final Instances propositionalisedDataset)
    {
        int numInst = bag.relationalValue(AdaptiveSplit.REL_INDEX).size();

        final double[] attValues = new double[root.nodeCount+1];
        for (int i=0;i<root.nodeCount;i++)
        {
            attValues[i]=1;
        }
        root.nodePropositionaliseBag(bag, attValues, new BitSet(numInst));
        attValues[root.nodeCount] = bag.classValue(); // set class val

        Instance prop = new DenseInstance(1.0, attValues);
        prop.setDataset(propositionalisedDataset);
        return prop;
    }

    /**
     * A way to evaluate each split point
     */
    public static double evaluateSplit(Instances trainingData, int splitAttrIndex,
                                 double splitPoint, Classifier classifier)
    {
        // TODO efficiency concerns
        // setup attr
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
        attInfo.add(new Attribute("less-than"));
        attInfo.add(new Attribute("greater-than"));
        attInfo.add((Attribute) trainingData.classAttribute().copy()); // class

        // create propositionalised dataset
        final int numBags = trainingData.numInstances();
        Instances propositionalisedDataset = new Instances("prop", attInfo, numBags);

        // TODO update this or make a const
        propositionalisedDataset.setClassIndex(2);

        for (Instance bag : trainingData)
        {
            // propositionalise the bag and add it to the set
            // TODO efficiency concerns
            Instance propositionalisedBag = propositionaliseBagViaOneSplit(bag.relationalValue(AdaptiveSplit.REL_INDEX), splitAttrIndex, splitPoint, bag.classValue(), propositionalisedDataset);

            propositionalisedDataset.add(propositionalisedBag);
        }

        // eval on propositionalised dataset
        // TODO, not sure if the following works...
        // TODO efficiency reasons.. is it better to compute non-cv error rate?
        try
        {
            classifier.buildClassifier(propositionalisedDataset);

            // count num errors
            int numErr = 0;
            for (Instance inst : propositionalisedDataset)
            {
                if (classifier.classifyInstance(inst) != inst.classValue())
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
    static Instance propositionaliseBagViaOneSplit(Instances bagInstances, int attrIndex,
                                                   double splitPoint, double classVal,
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

    @Override
    public String toString()
    {
        if (left == null)
        {
            // this is a leaf:
            return "\t["+ propositionalisedAttributeIndex + "] leaf.\n";
        }
        else
        {
            return "\t[" + propositionalisedAttributeIndex + "] Split on attr" +
                    splitAttrIndex + " at " + splitPoint + ". left=" +
                    left.propositionalisedAttributeIndex + ", right=" +
                    right.propositionalisedAttributeIndex + ".\n" +
                    left.toString() + right.toString();
        }
    }
}
