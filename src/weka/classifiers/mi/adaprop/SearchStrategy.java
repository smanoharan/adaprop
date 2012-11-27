package weka.classifiers.mi.adaprop;

import weka.core.Instances;
import weka.core.Tag;

import java.io.Serializable;
import java.util.BitSet;
import java.util.LinkedList;
import java.util.Queue;

public abstract class SearchStrategy implements Serializable
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
        int propLeftIndex = params.propStrategy.getNumPropAttrPerRegion();
        int propRightIndex = propLeftIndex*2;
        RootSplitNode root = RootSplitNode.toRootNode(
                new SplitNode(propLeftIndex, propRightIndex, 0),
                params.propStrategy);
        root.setNodeCount(1);
        return root;
    }

    // <editor-fold desc="===Option Handling===">
    private static final int SEARCH_BREADTH_FIRST = 1;
    private static final int SEARCH_BEST_FIRST = 2;
    public static final int DEFAULT_STRATEGY = SEARCH_BREADTH_FIRST;
    public static final String DESCRIPTION =
            "Search strategy: 1=breadth-first (default), 2=best-first";

    public static final Tag[] STRATEGIES =
    {
            new Tag(SEARCH_BREADTH_FIRST, "Build the tree using breadth first search"),
            new Tag(SEARCH_BEST_FIRST, "Build the tree using best first search")
    };

    /**
     * Get the strategy object corresponding to the specified
     *  strategy ID
     *
     * @param strategyID The ID representing the strategy
     * @return The strategy object corresponding to the strategyID
     */
    public static SearchStrategy getStrategy(final int strategyID)
    {
        switch (strategyID)
        {
            case SEARCH_BREADTH_FIRST:
                return new BreadthFirstSearchStrategy();
            case SEARCH_BEST_FIRST:
                return new BestFirstSearchStrategy();
            default:
                throw new IllegalArgumentException(
                        "Unknown search strategy code: " + strategyID);
        }
    }
    // </editor-fold>
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
        return (params.splitStrategy.canExpand(params.trainingBags, ignoredInst))
                && (numInstInNode >= params.minOccupancy);
    }

    @Override /** @inheritDoc */
    public RootSplitNode buildTree(final TreeBuildingParams params, final int instCount, final Instances trainingBags)
            throws Exception
    {
        // build the root:
        final int numAttrPerRegion = params.propStrategy.getNumPropAttrPerRegion();
        int nextPropIndex = 3*numAttrPerRegion;
        RootSplitNode root = buildRoot(params);
        final BitSet rootIgnoredInst = new BitSet(instCount);

        if (params.maxNodeCount > 0 && isExpandable(root, params, rootIgnoredInst)) {
            root.computeBestSplit(params, rootIgnoredInst, root);
        } else {
            root.setNodeCount(0);
            return root; // computation is complete.
        }

        // structure the tree into an queue via breadth-first-search:
        int numNodes = 1;
        Queue<Pair<SplitNode,BitSet>> queue = new LinkedList<Pair<SplitNode,BitSet>>();
        queue.add(new Pair<SplitNode, BitSet>(root, rootIgnoredInst));

        while(!queue.isEmpty() && numNodes < params.maxNodeCount)
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
            if (numNodes < params.maxNodeCount && isExpandable(node.right, params, counter.rightIgnore))
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
        final int numAttrPerRegion = params.propStrategy.getNumPropAttrPerRegion();
        int nextPropIndex = 3 * numAttrPerRegion;
        RootSplitNode root = buildRoot(params);
        final BitSet rootIgnoredInst = new BitSet(instCount);

        if (instCount >= params.minOccupancy && params.maxNodeCount > 0) {
            root.computeBestSplit(params, rootIgnoredInst, root);
        } else {
            root.setNodeCount(0);
            return root; // computation is complete.
        }

        // structure the search by keeping a list of expandable nodes
        //  (i.e. those which have at least one empty child.
        int nodeCount = 1;
        LinkedList<Pair<SplitNode, BitSet>> expandableLeafNodes = new LinkedList<Pair<SplitNode, BitSet>>();

        // initialise the border with the two children of the root.
        root.left = new SplitNode(-1, -1, 1);
        root.right = new SplitNode(-1, -1, 1);
        RegionPartitioner rootCounter = new RegionPartitioner(instCount);
        root.filterDataset(trainingBags, rootIgnoredInst, rootCounter);
        if (params.splitStrategy.canExpand(params.trainingBags, rootCounter.leftIgnore))
        {
            expandableLeafNodes.add(new Pair<SplitNode, BitSet>(root.left, rootCounter.leftIgnore));
        }
        if (params.splitStrategy.canExpand(params.trainingBags, rootCounter.rightIgnore))
        {
            expandableLeafNodes.add(new Pair<SplitNode, BitSet>(root.right, rootCounter.rightIgnore));
        }

        while(!expandableLeafNodes.isEmpty() && nodeCount < params.maxNodeCount)
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
                if (params.splitStrategy.canExpand(params.trainingBags, counter.leftIgnore))
                {
                    expandableLeafNodes.add(new Pair<SplitNode, BitSet>(bestNode.left, counter.leftIgnore));
                }
                if (params.splitStrategy.canExpand(params.trainingBags, counter.rightIgnore))
                {
                    expandableLeafNodes.add(new Pair<SplitNode, BitSet>(bestNode.right, counter.rightIgnore));
                }
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