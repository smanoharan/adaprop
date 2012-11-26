package weka.classifiers.mi.adaprop;

import org.junit.Test;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.mi.MIWrapper;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

import static junit.framework.Assert.assertEquals;

/**
 * Test the evaluation strategies.
 *
 * @author Siva Manoharan
 */
public class EvaluationTest extends TestBase
{
    private MisClassificationErrorEvaluationStrategy misclassificationErrorEvalStrategy;
    private RMSEEvaluationStrategy rmseEvaluationStrategy;

    @Override /** @inheritDoc */
    public void setUp() throws Exception
    {
        misclassificationErrorEvalStrategy = new MisClassificationErrorEvaluationStrategy();
        rmseEvaluationStrategy = new RMSEEvaluationStrategy();
        super.setUp();
    }

    // wrap classifier in MIWrapper and run eval
    private static void assertErrorMeasureIs(final double exp, final EvaluationStrategy evalStrategy,
                                             final Instances dataset,  final Classifier baseLearner) throws Exception
    {
        final String msg = baseLearner.getClass().getSimpleName() + ", " +
                evalStrategy.getClass().getSimpleName() + ", " + dataset.relationName() + ":";
        final MIWrapper classifier = new MIWrapper();
        classifier.setClassifier(baseLearner);
        final double act = evalStrategy.evaluateDataset(dataset, classifier);
        assertEquals(msg, exp,  act, LARGE_TOLERANCE);
    }

    @Test
    public void MisClassificationErrorShouldBeAsExpectedOnSimpleMIData() throws Exception
    {
        final EvaluationStrategy evalStrategy = misclassificationErrorEvalStrategy;
        final Instances dataset = simpleMIdata;

        // with ZeroR - expect 2 instances to be misclassified
        assertErrorMeasureIs(2, evalStrategy, dataset, new ZeroR());

        // with OneR - expect no misclassifications
        assertErrorMeasureIs(0, evalStrategy, dataset, new OneR());

        // with IBk - expect no misclassifications
        assertErrorMeasureIs(0, evalStrategy, dataset, new IBk(1));
    }

    @Test
    public void MisClassificationErrorShouldBeAsExpectedOnComplexMIData() throws Exception
    {
        final EvaluationStrategy evalStrategy = misclassificationErrorEvalStrategy;
        final Instances dataset = complexMIdata;

        // with ZeroR - expect 2 instances to be misclassified
        assertErrorMeasureIs(2, evalStrategy, dataset, new ZeroR());

        // with OneR - expect 2 instances to be misclassified
        assertErrorMeasureIs(2, evalStrategy, dataset, new OneR());

        // with IBk - expect no misclassifications
        assertErrorMeasureIs(0, evalStrategy, dataset, new IBk(1));
    }

    @Test
    public void RootMeanSquaredErrorShouldBeAsExpectedOnSimpleMIData() throws Exception
    {
        final EvaluationStrategy evalStrategy = rmseEvaluationStrategy;
        final Instances dataset = simpleMIdata;

        // with ZeroR - expect 0.4900
        assertErrorMeasureIs(0.4900, evalStrategy, dataset, new ZeroR());

        // with OneR - expect 0.0010
        assertErrorMeasureIs(0.0010, evalStrategy, dataset, new OneR());

        // with IBk - expect 0.0377
        assertErrorMeasureIs(0.0377, evalStrategy, dataset, new IBk(1));
    }

    @Test
    public void RootMeanSquaredErrorShouldBeAsExpectedOnComplexMIData() throws Exception
    {
        final EvaluationStrategy evalStrategy = rmseEvaluationStrategy;
        final Instances dataset = complexMIdata;

        // with ZeroR - expect 0.5000
        assertErrorMeasureIs(0.5000, evalStrategy, dataset, new ZeroR());

        // with OneR - expect 0.6645
        assertErrorMeasureIs(0.6645, evalStrategy, dataset, new OneR());

        // with IBk - expect 0.0356
        assertErrorMeasureIs(0.0356, evalStrategy, dataset, new IBk(1));
    }
}
