package weka.classifiers.mi.adaprop;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.mi.AdaProp;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * Represents a single split point (a node in the adaSplitTree).
 * This Node is either a leaf (left=right=null) or a branch (both left and right are
 *  not null).
 */
public class SplitNode implements Serializable
{
    //<editor-fold defaultstate="collapsed" desc="===Init===" >
    static final long serialVersionUID = AdaProp.serialVersionUID + 1000L;
    /** The instIndex of the relational attribute in the bag instance */
    public static final int REL_INDEX = 1;

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
        final StringBuilder result = new StringBuilder("region 0\n");
        final String prefix = "|   ";
        this.toStringRecursive(result, prefix);
        return result.toString();
    }

    private void toStringRecursive(StringBuilder result, String prefix)
    {
        final String newPrefix = prefix + "|   ";
        final String splitPtStr = String.format("%.3f", splitPoint);

        result.append(prefix).append("attr-").append(splitAttrIndex).append(" <= ")
                .append(splitPtStr).append(": region ").append(propLeftIndex).append("\n");
        if (this.left != null && this.left.splitAttrIndex >= 0)
        {
            this.left.toStringRecursive(result, newPrefix);
        }

        result.append(prefix).append("attr-").append(splitAttrIndex).append(" >  ")
                .append(splitPtStr).append(": region ").append(propRightIndex).append("\n");
        if (this.right != null && this.right.splitAttrIndex >= 0)
        {
            this.right.toStringRecursive(result, newPrefix);
        }
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
     * @param maxTreeSize The maximum size of the tree.
     * @param minOccupancy The minimum occupancy of each node.
     * @param propStrategy
     * @return The root of the split-tree
     */
    public static RootSplitNode buildTree(Instances trainingBags, final SplitStrategy splitStrategy, final int maxTreeSize,
                                          final int minOccupancy, final Classifier classifier,
                                          final SearchStrategy searchStrategy,
                                          final PropositionalisationStrategy propStrategy) throws Exception
    {
        // count the number of instances in all the bags:
        int instCount = 0;
        for (Instance bag : trainingBags)
        {
            instCount += bag.relationalValue(REL_INDEX).size();
        }

        TreeBuildingParams params = new TreeBuildingParams(maxTreeSize, minOccupancy, trainingBags,
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
        final Instances bagInst = bag.relationalValue(REL_INDEX);
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
            filterBag(bag.relationalValue(REL_INDEX), ignore, counter);
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

/** Data structure for storing the tree-building param */
final class TreeBuildingParams implements Serializable
{
    public final int maxNodeCount;
    public final int minOccupancy;
    public final Instances trainingBags;
    public final int instCount;
    public final SplitStrategy splitStrategy;
    public final PropositionalisationStrategy propStrategy;
    public final Classifier classifier;

    TreeBuildingParams(final int maxNodeCount, final int minOccupancy, final Instances trainingBags,
                       final int instCount, final SplitStrategy splitStrategy,
                       final PropositionalisationStrategy propStrategy, final Classifier classifier)
    {
        this.maxNodeCount = maxNodeCount;
        this.minOccupancy = minOccupancy;
        this.propStrategy = propStrategy;
        this.classifier = classifier;
        this.trainingBags = trainingBags;
        this.instCount = instCount;
        this.splitStrategy = splitStrategy;
    }
}

/** A (mutable) data structure for keeping track of information for partitioning regions. */
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