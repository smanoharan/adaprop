package weka.classifiers.mi;

import org.junit.Before;
import org.junit.BeforeClass;
import weka.core.*;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Author: Siva Manoharan, 1117707 */
public class AdaPropTestBase
{
    public static final int NUM_ATTR = 5; // number of attr in si-dataset
    public static final int NUM_BAGS = 3; // number of mi bags
    public static final int NUM_INST_PER_BAG = 4;

    /** Instance, under test */
    protected AdaProp adaProp;

    /** Contains the mi bags for testing */
    protected static Instances miData;

    /** Header for the single-instance relation */
    protected static Instances siHeader;

    /** Header for the resultant propositionalised dataset */
    protected static Instances propHeader;

    /** For comparing doubles */
    static final double TOLERANCE = 0.000001;

    /* TODO Tests:
    *  - GetCapabilities
    *  - MultiInstanceGetCapabilities
    *  - ListOptions
    *  - SetOptions
    *  - GetOptions
    *
    *  - main? (e.g. with an artificial dataset)
    *  - buildClassifier?
    *  - distributionForInstance
    */

    @BeforeClass
    /** Setup the instance headers */
    public static void setupClass() throws Exception
    {
        setupSingleInstanceHeader();
        setupMultiInstanceData();
        setupPropositionalisedHeader();
    }

    /** Setup the dataset for the propositionalised instances */
    private static void setupPropositionalisedHeader()
    {
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
        attInfo.add(new Attribute("total"));
        attInfo.add(new Attribute("less-than"));
        attInfo.add(new Attribute("greater-than"));
        attInfo.add((Attribute) miData.classAttribute().copy());
        propHeader = new Instances("prop-header", attInfo, 0);
        propHeader.setClassIndex(3);
    }

    /** Initialise the siHeader */
    private static void setupSingleInstanceHeader()
    {
        // si-header has 5 numeric attributes + 1 numeric class.
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();

        for(int i=0; i< NUM_ATTR; i++)
        {
            attInfo.add(new Attribute("attr" + i));
        }

        attInfo.add(new Attribute("class"));

        siHeader = new Instances("si-header", attInfo, 0);
        siHeader.setClassIndex(NUM_ATTR);
    }

    // initialise numInst new single-instances and return the set.
    protected static Instances newRelation(final int startVal, final int numInst)
    {
        // initialise relation as a copy of siHeader
        Instances relation = new Instances(siHeader, numInst);

        int val = startVal;
        for (int instIndex = 0; instIndex < numInst; instIndex++)
        {
            Instance inst = new DenseInstance(NUM_ATTR);

            for (int attrIndex = 0; attrIndex < NUM_ATTR; attrIndex++)
            {
                // increase value in a predictable pattern
                inst.setValue(attrIndex, val++);
            }

            // add inst to dataset
            inst.setDataset(relation);
            relation.add(inst);
        }

        return relation;
    }

    /** Create the attribute corresponding to the attribute field */
    protected static Attribute createRelationAttribute()
    {
        Attribute attr = new Attribute("bag", siHeader);
        for (int bagIndex = 0; bagIndex < NUM_BAGS; bagIndex++)
        {
            final int startVal = bagIndex*NUM_ATTR*NUM_INST_PER_BAG;
            attr.addRelation(newRelation(startVal, NUM_INST_PER_BAG));
        }
        return attr;
    }

    protected static Attribute createClassAttribute()
    {
        final List<String> attributeValues = new ArrayList<String>(2);
        attributeValues.add("class-0");
        attributeValues.add("class-1");
        return new Attribute("class", attributeValues);
    }

    /** Initialise the miData, assuming siHeader has been init'd */
    private static void setupMultiInstanceData()
    {
        // this has 3 attr: bag-id (numeric), bag (relational), class (numeric)
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
        attInfo.add(new Attribute("bag-id"));
        attInfo.add(createRelationAttribute());
        attInfo.add(createClassAttribute());

        miData = new Instances("mi-header", attInfo, 0);
        miData.setClassIndex(2);

        // populate mi-header
        for (int bagIndex = 0; bagIndex < NUM_BAGS; bagIndex++)
        {
            Instance bag = new DenseInstance(3);
            bag.setValue(0, bagIndex);
            bag.setValue(1, bagIndex);
            int bagClass = bagIndex < 2 ? 0 : 1;
            bag.setValue(2, bagClass);

            // add bag to dataset
            bag.setDataset(miData);
            miData.add(bag);
        }
    }

    @Before
    /** Create a new instance of AdaProp */
    public void setUp() throws Exception
    {
        this.adaProp = new AdaProp();
    }

    // check that the string is not null and is not empty.
    protected static void assertNotNullOrEmpty(String toTest)
    {
        assertNotNull(toTest);
        assertTrue(toTest.length() > 0);
    }

    protected static void assertOptionEquals(
            final Option actual, final String expDesc,
            final int expNumArgs, final String expSynopsis)
    {
        assertEquals(actual.name() + " Desc: ",     expDesc,        actual.description());
        assertEquals(actual.name() + " NumArgs: ",  expNumArgs,     actual.numArguments());
        assertEquals(actual.name() + " Synopsis: ", expSynopsis,    actual.synopsis());
    }

    /**
     * Assuming that exp and act have the same format,
     *  check that they have the same values for all attributes.
     *
     * @param exp Expected
     * @param act Actual
     */
    private static void assertInstanceEquals(String msg, Instance exp, Instance act)
    {
        // check number of attributes is equal
        int numAttr = act.numAttributes();
        assertEquals("Number of attributes", exp.numAttributes(), numAttr);

        // for each attribute
        for (int attrIndex = 0; attrIndex < numAttr; attrIndex++)
        {
            String msgAttr = ", attribute: " + attrIndex;
            assertEquals(msg + msgAttr, exp.value(attrIndex), act.value(attrIndex), TOLERANCE);
        }
    }

    /**
     * Assuming that exp and act have the same format,
     *  check that their instances have the same values
     *
     * @param exp Expected
     * @param act Actual
     */
    protected static void assertDatasetEquals(Instances exp, Instances act)
    {
        // check num of instances is the same
        int actNumInst = act.numInstances();
        assertEquals("Number of instances \n\n" + exp.toString() + "\n\n" + act.toString() + "\n\n",
                exp.numInstances(), actNumInst);

        // check each instance is equal
        for (int instIndex = 0; instIndex < actNumInst; instIndex++)
        {
            final Instance expInst = exp.get(instIndex);
            final Instance actInst = act.get(instIndex);
            final String msg = "Instance: [" + instIndex + "] -- " + expInst.toString() + " -- " + actInst.toString();

            assertInstanceEquals(msg, expInst, actInst);
        }
    }



    protected static void assertListEquals(String msg, List<Double> exp, List<Double> act)
    {
        // check sizes are equal
        assertEquals(msg + " size", exp.size(), act.size());

        // check each elem
        for (int i=0;i<exp.size();i++)
        {
            assertEquals(msg + " index " + i, exp.get(i), act.get(i), TOLERANCE);
        }
    }

    protected static void assertPairListEquals(String msg, List<CompPair<Integer, Double>> exp, List<CompPair<Integer, Double>> act)
    {
        // check sizes are equal
        assertEquals(msg + " size", exp.size(), act.size());

        // check each elem
        for (int i=0;i<exp.size();i++)
        {
            String msgSuffix = exp.get(i).toString() + " != " + act.get(i).toString();
            assertEquals(msg + " (key) index " + i + msgSuffix, exp.get(i).key, act.get(i).key, TOLERANCE);
            assertEquals(msg + " (val) index " + i + msgSuffix, exp.get(i).value, act.get(i).value, TOLERANCE);
        }
    }

    protected static RootSplitNode createRootSplit(final int attrIndex, final double splitPt)
    {
        RootSplitNode root = new RootSplitNode(1, 2, attrIndex, splitPt, null, null, 0, new CountBasedPropositionalisationStrategy());
        root.setNodeCount(1);
        return root;
    }
}
