package weka.classifiers.mi;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Generate an artificial dataset, for testing AdaProp works as expected.
 *
 * Author: Siva Manoharan
 */
public class ArtificialDatasetGenerator
{
    /** To allow running from the command line */
    public static void main(String[] args)
    {
        // validate args:
        if (args.length != 6)
        {
            String msg = String.format("Too %s args (%d) specified. Expected 6 arguments.",
                    (args.length < 5 ? "few" : "many"), args.length);
            System.err.println(msg);
            System.err.println("Usage: generate <numAttr> <numBags> <minInstPerBag> <maxInstPerBag> <numSplits>" +
                    "<randomNumberSeed>");
            return;
        }


        // read args:
        int curIndex = 0;
        int numAttr = Integer.parseInt(args[curIndex++]);
        int numBags = Integer.parseInt(args[curIndex++]);
        int minInstPerBag = Integer.parseInt(args[curIndex++]);
        int maxInstPerBag = Integer.parseInt(args[curIndex++]);
        int numSplits = Integer.parseInt(args[curIndex++]);
        long seed = Long.parseLong(args[curIndex++]);

        // print argument parameters:
        System.out.println("% Artificially generated file. Arguments: " + 
          "\n%\t Number of Attributes: " + numAttr + 
          "\n%\t Number of Bags:       " + numBags + 
          "\n%\t Minimum Bag Size:     " + minInstPerBag + 
          "\n%\t Maximum Bag Size:     " + maxInstPerBag + 
          "\n%\t Number of Splits:     " + numSplits + 
          "\n%\t Seed: " + seed + "\n"); 
        
        // output in arff format
        System.out.println(generate(numAttr, numBags, minInstPerBag, maxInstPerBag, numSplits, seed).toString());
    }

    /**
     * Generate a random MI dataset, using the specified parameters.
     *
     * @param numAttr The number of attributes for each single-instance.
     * @param numBags The number of bags in the MI dataset.
     * @param minInstPerBag The minimum number of instances in each bag.
     * @param maxInstPerBag The maximum number of instances in each bag.
     * @param numSplits The number of splits (depth of split tree).
     * @param seed The seed for the random number generator.
     * @return The randomly generated dataset.
     */
    public static Instances generate(int numAttr, int numBags, int minInstPerBag, int maxInstPerBag, int numSplits, long seed)
    {
        final Random random = new Random(seed);
        final Instances siHeader = buildSingleInstHeader(numAttr);
        final ArrayList<Instances> bags = new ArrayList<Instances>(numBags);

        // generate the bags:
        for (int i=0; i<numBags; i++)
        {
            bags.add(generateBag(siHeader, numAttr, minInstPerBag, maxInstPerBag, random));
        }

        final ArrayList<Integer> classVals = determineClassVal(numAttr, bags, numSplits, random);
        final String relName = String.format("artificialDataset_a%d_b%d_r%d-%d_d%d_s%d",
                numAttr, numBags, minInstPerBag, maxInstPerBag, numSplits, seed);
        return buildMultiInstanceDataset(siHeader, bags, classVals, relName);
    }

    // a data structure (tuple) for storing a bag and a count
    private static class BagCountPair implements Comparable<BagCountPair>
    {
        public final int bagIndex;
        public final int count;

        private BagCountPair(final int bagIndex, final int count)
        {
            this.bagIndex = bagIndex;
            this.count = count;
        }

        @Override
        public int compareTo(final BagCountPair o)
        {
            return this.count - o.count;
        }
    }

    // a data structure (tuple) for storing Splits
    private static class Split implements Comparable<Split>
    {
        public final int attrIndex;
        public final double splitPt;
        public final boolean positiveRegionIsLeftOfSplit;

        private Split(final int attrIndex, final double splitPt, final boolean positiveRegion)
        {
            this.attrIndex = attrIndex;
            this.splitPt = splitPt;
            this.positiveRegionIsLeftOfSplit = positiveRegion;
        }

        @Override
        public int compareTo(final Split o)
        {
            return this.attrIndex - o.attrIndex;
        }

        @Override
        public String toString()
        {
            return "Attribute " + attrIndex + (positiveRegionIsLeftOfSplit ? " <= " : "  > ")  + splitPt + ".";
        }
    }


    /**
     * Determine the class value of each of the bags, by randomly generating positive and negative regions.
     * @param numAttr Number of attributes in the single instance dataset.
     * @param bags The bags of instances.
     * @param numSplits The number of splits (the depth of the split tree).
     * @param random The random number generator.
     * @return The class labels for each bag.
     */
    private static ArrayList<Integer> determineClassVal(int numAttr, ArrayList<Instances> bags,
                                                        int numSplits, Random random)
    {
        if (numSplits > numAttr) throw new RuntimeException("Cannot have more splits than attributes!");
        final int numBags = bags.size();

        // set all bags to the positive class (0) for now.
        final ArrayList<Integer> classVals = new ArrayList<Integer>(numBags);
        for (int i=0; i<numBags; i++)
        {
            classVals.add(0);
        }

        // randomly select numSplit attributes to split on (without replacement)
        ArrayList<Split> splits = new ArrayList<Split>(numSplits);
        Set<Integer> usedAttr = new TreeSet<Integer>();
        for (int i=0; i<numSplits; i++)
        {
            // find the next split:
            int splitAttr = random.nextInt(numAttr);
            while (usedAttr.contains(splitAttr)) // ensure attr is not repeated
            {
                splitAttr = random.nextInt(numAttr);
            }
            usedAttr.add(splitAttr);

            final double mean = findMidpt(bags, splitAttr);
            splits.add(new Split(splitAttr, mean, random.nextBoolean()));
        }

        // print out the splits (as comments)
        System.out.println("% Splits:");
        for (Split split : splits)
        {
          System.out.println("%\t" + split.toString() );
        }

        // for each bag, count how many instances fall in the 'positive' region
        ArrayList<BagCountPair> bagCountPairs = new ArrayList<BagCountPair>(numBags);
        for (int i=0; i<numBags; i++)
        {
            Instances bag = bags.get(i);
            int count = 0;
            for (Instance inst : bag)
            {
                boolean inRegion = true;
                for (Split split : splits)
                {
                    if (split.positiveRegionIsLeftOfSplit != (inst.value(split.attrIndex) <= split.splitPt))
                    {
                        inRegion = false;
                        break;
                    }
                }

                if (inRegion)
                {
                    count++;
                }
            }
            bagCountPairs.add(new BagCountPair(i, count));
        }

        // order the bags by count
        Collections.sort(bagCountPairs);

        // find cut-off point (atm, right in the middle). Mark all below the cutoff as negative (1).
        final int cutoffIndex = numBags / 2;
        System.out.println("% Class value is based on whether COUNT <= " + bagCountPairs.get(cutoffIndex).count);

        for (int i=0; i<cutoffIndex; i++)
        {
            BagCountPair bcp = bagCountPairs.get(i);
            classVals.set(bcp.bagIndex, 1);
        }

        return classVals;
    }

    /**
     * Find the midpoint of all instances in all bags over the given attribute.
     * @param bags The bags of instances
     * @param attrIndex The attribute to find midpt for
     * @return The midpoint value
     */
    private static double findMidpt(final ArrayList<Instances> bags, final int attrIndex)
    {
        double sum = 0;
        int count = 0;
        for (Instances bag : bags)
        {
            for (Instance inst : bag)
            {
                sum += inst.value(attrIndex);
                count++;
            }
        }
        return sum / count;
    }

    /**
     * Builds the Instance header for each (single) instance. (I.e. not for the bags).
     *
     * @param numAttr The number of attributes in the single-inst dataset
     * @return The single instance header
     */
    private static Instances buildSingleInstHeader(final int numAttr)
    {
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>(numAttr);
        for (int i=0; i<numAttr; i++)
        {
            attInfo.add(new Attribute("attr-"+i));
        }
        return new Instances("siHeader", attInfo, 0);
    }

    /**
     * Assemble a list of relations and class values into a multi-instance dataset (i.e. relation of bags).
     *
     * @param siHeader The instance header for the single instance relation.
     * @param bags The bags, specified as a list of relations.
     * @param classVals The class value of each bag.
     * @param relName The name of the MI dataset.
     * @return The MI dataset.
     */
    private static Instances buildMultiInstanceDataset(Instances siHeader, ArrayList<Instances> bags,
                                                       ArrayList<Integer> classVals, String relName)
    {
        final int numBags = bags.size();
        final ArrayList<Attribute> attInfo = new ArrayList<Attribute>(3);

        // bag-ids:
        final List<String> bagNames = new ArrayList<String>(numBags);
        for (int i=0; i<numBags; i++)
        {
            bagNames.add("bag-" + i);
        }
        attInfo.add(new Attribute("bag-id", bagNames));

        // bags:
        Attribute bagAttr = new Attribute("bag", siHeader);
        for (Instances bag : bags)
        {
            bagAttr.addRelation(bag);
        }
        attInfo.add(bagAttr);

        // class values:
        attInfo.add(new Attribute("class", Arrays.asList("positive", "negative")));

        // construct full data:
        Instances dataset = new Instances(relName, attInfo, numBags);
        for (int i=0; i<numBags; i++)
        {
            Instance bagInst = new DenseInstance(1, new double[] {i,i, classVals.get(i)});
            bagInst.setDataset(dataset);
            dataset.add(bagInst);
        }

        return dataset;
    }

    /**
     * Randomly generate a bag of N instances (minNumInst <= N <= maxNumInst), each with the specified number of
     *  attributes.
     *
     * @param siHeader The header for each instance.
     * @param numAttr The number of attributes.
     * @param minNumInst The minimum number of instances in the bag, inclusive.
     * @param maxNumInst The maximum number of instances in the bag, inclusive.
     * @param random The random number generator.
     * @return
     */
    private static Instances generateBag(Instances siHeader, int numAttr, int minNumInst, int maxNumInst, Random random)
    {
        final int numInst = random.nextInt(maxNumInst + 1 - minNumInst) + minNumInst; // + 1 since max is exclusive.
        Instances bag = new Instances(siHeader, 0);
        for (int i=0; i<numInst; i++)
        {
            Instance inst = generateInstance(numAttr, random);
            bag.add(inst);
            inst.setDataset(bag);
        }
        return bag;
    }

    /**
     * Generate an instance with the specified number of attributes, with each value
     *  generated randomly using the provided random number generator.
     *
     * @param numAttr The number of attributes
     * @param random The random number generator
     * @return The generated instance
     */
    private static Instance generateInstance(int numAttr, Random random)
    {
        double[] attrVals = new double[numAttr];
        for (int i=0; i<numAttr; i++)
        {
            attrVals[i] = random.nextDouble();
        }
        return new DenseInstance(1, attrVals);
    }
}
