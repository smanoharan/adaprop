package weka.classifiers.mi.adaprop;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Tag;

import java.io.Serializable;
import java.util.Random;

/**
 * Defines the strategy for evaluating datasets
 */
public abstract class EvaluationStrategy implements Serializable
{
    /**
     * Evaluate the classifier on the dataset, returning the chosen metric
     *
     * @param dataset The dataset to evaluate
     * @param classifier The classifier to evaluate
     * @return The chosen error metric when training the classifier on the dataset
     * @throws Exception
     */
    public double evaluateDataset(Instances dataset, Classifier classifier) throws Exception
    {
        classifier.buildClassifier(dataset);
        Evaluation evaluation = new Evaluation(dataset);
        return evaluateModel(evaluation, classifier, dataset);
    }

    /**
     * Determine the error rate for this evaluation.
     * @param eval The Evaluation object
     * @return The error-rate
     */
    protected abstract double evaluateModel(Evaluation eval, Classifier classifier, Instances dataset)
            throws Exception;

    // <editor-fold desc="===Option Handling===">
    private static final int EVAL_MISCLASSIFICATION_ERROR = 1;
    private static final int EVAL_RMSE = 2;
    private static final int EVAL_CV_MISCLASSIFICATION_ERROR = 3;
    private static final int EVAL_CV_RMSE = 4;
    public static final int DEFAULT_STRATEGY = EVAL_MISCLASSIFICATION_ERROR;
    public static final String DESCRIPTION =
            "Split Evaluation strategy: 1=mis-classification-error (default), 2=root-mean-squared-error, " +
                    "3=cross-validated-mis-classification-error, 4=cross-validated-root-mean-squared-error";

    public static final Tag[] STRATEGIES =
            {
                    new Tag(EVAL_MISCLASSIFICATION_ERROR, "By Misclassification error"),
                    new Tag(EVAL_RMSE, "By Root mean squared error"),
                    new Tag(EVAL_CV_MISCLASSIFICATION_ERROR, "By Cross-validated Misclassification error"),
                    new Tag(EVAL_CV_RMSE, "By Cross-validated Root mean squared error")
            };

    /**
     * Get the strategy object corresponding to the specified
     *  strategy ID
     *
     * @param strategyID The ID representing the strategy
     * @return The strategy object corresponding to the strategyID
     */
    public static EvaluationStrategy getStrategy(final int strategyID)
    {
        switch (strategyID)
        {
            case EVAL_MISCLASSIFICATION_ERROR:
                return new MisClassificationErrorEvaluationStrategy();
            case EVAL_RMSE:
                return new RMSEEvaluationStrategy();
            case EVAL_CV_MISCLASSIFICATION_ERROR:
                return new MisClassificationCrossValidatedErrorEvaluationStrategy(new Random(), 2);
            case EVAL_CV_RMSE:
                return new RMSECrossValidatedErrorEvaluationStrategy(new Random(), 2);
            default:
                throw new IllegalArgumentException(
                        "Unknown evaluation strategy code: " + strategyID);
        }
    }
    // </editor-fold>
}

class MisClassificationErrorEvaluationStrategy extends EvaluationStrategy
{
    @Override
    public double evaluateModel(final Evaluation eval, Classifier classifier, Instances dataset)
            throws Exception
    {
        eval.evaluateModel(classifier, dataset);
        return eval.incorrect();
    }
}

class RMSEEvaluationStrategy extends EvaluationStrategy
{
    @Override
    public double evaluateModel(final Evaluation eval, Classifier classifier, Instances dataset)
            throws Exception
    {

        eval.evaluateModel(classifier, dataset);
        return eval.rootMeanSquaredError();
    }
}

class MisClassificationCrossValidatedErrorEvaluationStrategy extends EvaluationStrategy
{
    private final Random random;
    private final int numFolds;

    public MisClassificationCrossValidatedErrorEvaluationStrategy(Random random, int numFolds)
    {
        this.random = random;
        this.numFolds = numFolds;
    }

    @Override
    public double evaluateModel(final Evaluation eval, Classifier classifier, Instances dataset)
            throws Exception
    {
        eval.crossValidateModel(classifier, dataset, numFolds, random);
        return eval.incorrect();
    }
}

class RMSECrossValidatedErrorEvaluationStrategy extends EvaluationStrategy
{
    private final Random random;
    private final int numFolds;

    public RMSECrossValidatedErrorEvaluationStrategy(Random random, int numFolds)
    {
        this.random = random;
        this.numFolds = numFolds;
    }

    @Override
    public double evaluateModel(final Evaluation eval, Classifier classifier, Instances dataset)
            throws Exception
    {
        eval.crossValidateModel(classifier, dataset, numFolds, random);
        return eval.rootMeanSquaredError();
    }
}