package weka.classifiers.mi;

import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Author: Siva Manoharan, 1117707 */
public class AdaptiveSplitTest
{
    public static final int NUM_ATTR = 5; // number of attr in si-dataset
    public static final int NUM_BAGS = 3; // number of mi bags
    public static final int NUM_INST_PER_BAG = 4;

    /** Instance, under test */
    private AdaptiveSplit adaptiveSplit;

    private static Instances miData;

    /** Header for the single-instance relation */
    private static Instances siHeader;
    private static final double TOLERANCE = 0.000001;


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
    private static Instances newRelation(final int startVal, final int numInst)
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
    private static Attribute createRelationAttribute()
    {
        Attribute attr = new Attribute("bag", siHeader);
        for (int bagIndex = 0; bagIndex < NUM_BAGS; bagIndex++)
        {
            final int startVal = bagIndex*NUM_ATTR*NUM_INST_PER_BAG;
            attr.addRelation(newRelation(startVal, NUM_INST_PER_BAG));
        }
        return attr;
    }

    /** Initialise the miData, assuming siHeader has been init'd */
    private static void setupMultiInstanceData()
    {
        // this has 3 attr: bag-id (numeric), bag (relational), class (numeric)
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
        attInfo.add(new Attribute("bag-id"));
        attInfo.add(createRelationAttribute());
        attInfo.add(new Attribute("class"));

        miData = new Instances("mi-header", attInfo, 0);
        miData.setClassIndex(2);

        // populate mi-header
        for (int bagIndex = 0; bagIndex < NUM_BAGS; bagIndex++)
        {
            Instance bag = new DenseInstance(3);
            bag.setValue(0, bagIndex);
            bag.setValue(1, bagIndex);
            bag.setValue(2, bagIndex % 2); // alternate between class=0 and 1.

            // add bag to dataset
            bag.setDataset(miData);
            miData.add(bag);
        }
    }

    @Before
    /** Create a new instance of AdaptiveSplit */
    public void setUp() throws Exception
    {
        this.adaptiveSplit = new AdaptiveSplit();
    }

    // check that the string is not null and is not empty.
    private static void assertNotNullOrEmpty(String toTest)
    {
        assertNotNull(toTest);
        assertTrue(toTest.length() > 0);
    }

    @Test
    public void testGlobalInfoIsNotNullOrEmpty()
    {
        assertNotNullOrEmpty(adaptiveSplit.globalInfo());
    }

    @Test
    public void testToStringIsNotNullOrEmpty() throws Exception
    {
        assertNotNullOrEmpty(adaptiveSplit.toString());
    }

    @Test
    public void testFindMean() throws Exception
    {
        // actual values for mean:
        //      there are 12 instances, 5 attributes
        //      the values are the natural numbers in sequence.
        //      for example, inst1 = {0,1,2,3,4} ; inst2 = {5,6,7,8,9} etc.
        //      so the mean(attr-i)=(\sum_{j=0}^{num_inst-1} (i+j*num_attr))/12
        // in fact, sum(attr-i)
        //      = num_inst*i + num_attr*(\sum_{j=0}^{num_inst-1}(j))
        //      = num_inst*i + num_attr*(1+2+3+...+num_inst-1)
        //      = num_inst*i + num_attr*((num_inst * num_inst-1) / 2)
        // thus, mean(attr-i) = i + num_attr*(num_inst-1)/2

        // for each attribute:
        final int numInst = NUM_INST_PER_BAG * NUM_BAGS;
        for (int attrIndex = 0; attrIndex < NUM_ATTR; attrIndex++)
        {
            final double expectedMean = attrIndex + (NUM_ATTR*(numInst-1)/2.0);
            final double actualMean = adaptiveSplit.findMean(miData, attrIndex);
            final String msg = "Mean for attribute " + attrIndex;
            assertEquals(msg, expectedMean, actualMean, TOLERANCE);
        }
    }

    @Test
    public void testEvalSplit() throws Exception
    {
        // TODO
    }

    @Test
    public void testPropositionaliseBag() throws Exception
    {
        // TODO
    }
}

