package cz.muni.neural.network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import cz.muni.neural.network.matrix.DoubleMatrix;
import cz.muni.neural.network.model.ForwardPropagationResult;
import cz.muni.neural.network.model.LabeledPoint;
import cz.muni.neural.network.model.Result;
import cz.muni.neural.network.util.Utils;

/**
 * @author Pavol Loffay
 */
public class NeuralNetwork {
    private static final double EPSILON_INIT_THETA = 0.12D;
    private static final double BIAS = 1D;

    /**
     * Parameters
     */
    private final List<Layer> layers;
    private final int numOfLayers;
    private final double gradientAlpha;
    private final long gradientNumberOfIter;
    private final boolean regularize;
    private final double lambdaRegul;
    private final Function<Double, Double> hypothesis;
    private final Function<Double, Double> hypothesisDer;

    private List<DoubleMatrix> thetas;


    private NeuralNetwork(List<Layer> layers, double gradientAlpha, long gradientNumberOfIter, boolean regularize,
                         double lambdaRegul,
                          Function<Double, Double> hypothesis, Function<Double, Double> hypothesisDer) {
        this.layers = layers;
        this.numOfLayers = layers.size();
        this.gradientAlpha = gradientAlpha;
        this.gradientNumberOfIter = gradientNumberOfIter;
        this.regularize = regularize;
        this.lambdaRegul = lambdaRegul;
        this.hypothesis = hypothesis;
        this.hypothesisDer = hypothesisDer;

        this.thetas = createThetas(true);
    }

    public void train(List<LabeledPoint> labeledPoints) {
        gradientDescent(labeledPoints);
    }

    /**
     * returns the probability
     */
    public Result predict(LabeledPoint labeledPoint) {

        ForwardPropagationResult forwardPropagationResult = forwardPropagation(Arrays.asList(labeledPoint));
        DoubleMatrix result = forwardPropagationResult.getLastActivation();

        return new Result(result.maxValueInRow());
    }

    private void gradientDescent(List<LabeledPoint> labeledPoints) {

        for (int i = 0; i < gradientNumberOfIter; i++) {

            List<DoubleMatrix> thetasGrad = forwardAndBackPropagation(labeledPoints);

            for (int layer = 0; layer < numOfLayers - 1; layer++) {

                DoubleMatrix gradient = thetasGrad.get(layer).scalarMultiply(gradientAlpha);
                DoubleMatrix theta = this.thetas.get(layer).subtract(gradient);

                this.thetas.set(layer, theta);
            }

            //vypisujeme chybu pouze u kazde 5. iterace
            String cost = i % 5 == 0 ? ", error = " + computeCost(labeledPoints) : "";
            System.out.println("Gradient iteration = " + i + cost);
        }
    }

    /**
     * Forward propagation and backward propagation
     * @return thetas gradient
     */
    private List<DoubleMatrix> forwardAndBackPropagation(List<LabeledPoint> labeledPoints) {

        // initialize with zeros
        List<DoubleMatrix> thetasGrad = createThetas(false);

        for (LabeledPoint labeledPoint: labeledPoints) {
            if (labeledPoint.getFeatures().length != layers.get(0).getNumberOfUnits()) {
                throw new IllegalArgumentException("Labeled point has wrong number of features");
            }

            /**
             * Forward propagation
             */
            ForwardPropagationResult forwardPropagationResult = forwardPropagation(Arrays.asList(labeledPoint));
            List<DoubleMatrix> zetas = forwardPropagationResult.getZetas();
            List<DoubleMatrix> activations = forwardPropagationResult.getActivations();
            List<DoubleMatrix> deltas = new ArrayList<>(numOfLayers - 1);

            /**
             * Back propagation
             *
             * Example for 3 layer network
             *
             *                        index   0   , 1
             * Deltas are stored as follows delta3, delta2 ...
             *
             * zetas are stored z2, z3
             * activations a1, a2, a3
             *
             * a1 = X with bias
             */
            DoubleMatrix lastDelta =
                    activations.get(numOfLayers - 1).subtract(logicalResultColumnVector(labeledPoint));
            deltas.add(lastDelta);
            for (int layer = 1; layer < numOfLayers - 1; layer++) {

                DoubleMatrix zeta = zetas.get((numOfLayers - 1) - layer - 1).applyOnEach(hypothesisDer);
                zeta = zeta.addFirstRow(BIAS);

                DoubleMatrix delta = thetas.get((numOfLayers - 1) - layer).transpose()
                        .matrixMultiply(deltas.get(layer -1));

                delta = delta.multiplyByElements(zeta);
                delta = delta.removeFirstRow();
                deltas.add(delta);
            }

            for (int layer = 0; layer < numOfLayers - 1; layer++) {

                DoubleMatrix activation = activations.get(layer).transpose();
                DoubleMatrix delta = deltas.get((numOfLayers - 1) - layer - 1).matrixMultiply(activation);

                DoubleMatrix thetaGrad = thetasGrad.get(layer).sum(delta);
                thetasGrad.set(layer, thetaGrad);
            }
        }


        for (int i = 0; i < numOfLayers - 1; i++) {
            DoubleMatrix thetaGrad = thetasGrad.get(i).scalarMultiply(1D/(double)labeledPoints.size());
            thetasGrad.set(i, thetaGrad);
        }

        /**
         * Regularization
         */
        if (this.regularize) {
            thetasGrad = regularizeThetasGrad(thetasGrad, labeledPoints.size());
        }
        return thetasGrad;
    }

    private ForwardPropagationResult forwardPropagation(List<LabeledPoint> labeledPoints) {
        List<DoubleMatrix> zetas = new ArrayList<>(numOfLayers - 1);
        List<DoubleMatrix> activations = new ArrayList<>(numOfLayers -1);

        DoubleMatrix a1 = Utils.labeledPointsToDoubleMatrix(labeledPoints);
        a1 = a1.transpose();
        a1 = a1.addFirstRow(BIAS);
        activations.add(a1);

        // forward propagation
        for (int layer = 0; layer < numOfLayers - 1; layer++) {

            DoubleMatrix zet = this.thetas.get(layer).matrixMultiply(activations.get(layer));
            DoubleMatrix activation = zet.applyOnEach(hypothesis);

            if (layer < numOfLayers - 2) {
                // skip last
                activation = activation.addFirstRow(BIAS);
            }

            activations.add(activation);
            zetas.add(zet);
        }

        ForwardPropagationResult forwardPropagationResult = new ForwardPropagationResult(activations, zetas);
        return forwardPropagationResult;
    }


    private List<DoubleMatrix> regularizeThetasGrad(List<DoubleMatrix> thetasGrad, long numberOfSamples) {
        double regulator = lambdaRegul / ((double) numberOfSamples);
        for (int i = 0; i < thetasGrad.size(); i++) {
            DoubleMatrix thetaGrad = thetasGrad.get(i);

            DoubleMatrix thetaRegul = thetas.get(i).removeFirstColumn().addFirstColumn(0).scalarMultiply(regulator);
            thetaGrad = thetaGrad.sum(thetaRegul);

            thetasGrad.set(i, thetaGrad);
        }

        return thetasGrad;
    }

    private DoubleMatrix logicalResultColumnVector(LabeledPoint labeledPoint) {
        int numberOfClasses = layers.get(layers.size() -1).getNumberOfUnits();

        double[][] arr = new double[numberOfClasses][1];
        
        if (numberOfClasses > 1) {
            //classification
            for (int col = 0; col < numberOfClasses; col++) {
                arr[col][0] = labeledPoint.getLabel() == col ? 1D : 0D;
            }
        } else {
            //prediction
            arr[0][0] = labeledPoint.getLabel();
        }

        return new DoubleMatrix(arr);
    }

    /**
     *
     * @param random, if false thetas are initialized with 0
     * @return
     */
    private List<DoubleMatrix> createThetas(boolean random) {
        List<DoubleMatrix> thetasReturn = new ArrayList<>(numOfLayers - 1);

        for (int layer = 0; layer < numOfLayers - 1; layer++) {

            int rows = layers.get(layer + 1).getNumberOfUnits();
            int cols =   layers.get(layer).getNumberOfUnits() + 1;

            DoubleMatrix theta = random ?
                    Utils.randomMatrix(EPSILON_INIT_THETA, rows, cols) :
                    new DoubleMatrix(0, rows, cols);

            thetasReturn.add(theta);
        }

        return thetasReturn;
    }

    private double computeCost(List<LabeledPoint> labeledPoints) {
        
        int points = labeledPoints.size();
        int numberOfClasses = layers.get(layers.size() -1).getNumberOfUnits();
        double cost = 0;
                
        if (numberOfClasses > 1) {
            //classification
            int ok = 0;
            for (LabeledPoint lp : labeledPoints) {
                Result r = predict(lp);
                if (lp.getLabel() == r.getMaxIndex()) {
                    ok++;
                }
            }
            cost = 1 - (ok / (double)points);
        } else {
            //prediction
            ForwardPropagationResult forwardPropagationResult = forwardPropagation(labeledPoints);
            double[] labels = new double[points];
        
            for (int i = 0; i < points; i++) {
                labels[i] = labeledPoints.get(i).getLabel();
            }
            cost = Utils.rmse(forwardPropagationResult.getLastActivation().getRow(0), labels);
        }
        
        return cost;
    }

    /**
     * Only point how to build the network
     */
    public static class Builder {
        private double gradientAlpha = 0.5;
        private long gradientIterations = 50;

        private boolean regularize;
        private double regularizeLambda = 1;

        private Function<Double, Double> hypothesis = new Functions.Sigmoid();
        private Function<Double, Double> hypothesisDer = new Functions.SigmoidGradient();

        private List<Layer> layers = new ArrayList<>();


        private Builder() {
        }

        public Builder withGradientAlpha(double alpha) {
            this.gradientAlpha = alpha;
            return this;
        }

        public Builder withGradientIterations(long iterations) {
            this.gradientIterations = iterations;
            return this;
        }

        public Builder withRegularize(boolean regularize) {
            this.regularize = regularize;
            return this;
        }

        public Builder withRegularizeLambda(double lambda) {
            this.regularizeLambda = lambda;
            this.regularize = true;
            return this;
        }

        public Builder withHypothesisFn(Function<Double, Double> fn) {
            this.hypothesis = fn;
            return this;
        }

        public Builder withHypothesisDerivation(Function<Double, Double> fn) {
            this.hypothesisDer = fn;
            return this;
        }

        public BuilderLayers withInputLayer(int numberOfFeatures) {
            Layer inputLayer = new Layer(numberOfFeatures);
            layers.add(inputLayer);
            return new BuilderLayers(this);
        }
    }

    public static final class BuilderLayers {
        private Builder builder;

        private BuilderLayers(Builder builder) {
            this.builder = builder;
        }

        public BuilderLayers addLayer(int units) {
            builder.layers.add(new Layer(units));
            return this;
        }
        
        public BuilderLayers addLayers(List<Integer> newLayers) {
            for (Integer i : newLayers) {
                builder.layers.add(new Layer(i));
            }
            return this;
        }

        public NeuralNetwork addLastLayer(int classesToClassify) {
            builder.layers.add(new Layer(classesToClassify));

            return new NeuralNetwork(builder.layers, builder.gradientAlpha,
                    builder.gradientIterations, builder.regularize, builder.regularizeLambda,
                    builder.hypothesis, builder.hypothesisDer);
        }
    }

    public static Builder newBuilder() {
        return new Builder();
    }
}
