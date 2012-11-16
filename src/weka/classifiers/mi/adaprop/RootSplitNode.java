package weka.classifiers.mi.adaprop;

import weka.core.Attribute;

import java.util.ArrayList;

public class RootSplitNode extends SplitNode
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
