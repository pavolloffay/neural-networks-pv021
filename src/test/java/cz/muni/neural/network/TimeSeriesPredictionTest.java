package cz.muni.neural.network;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.number.OrderingComparison.greaterThanOrEqualTo;
import static org.hamcrest.number.OrderingComparison.lessThanOrEqualTo;
import static org.junit.Assert.assertThat;

import java.io.IOException;
import java.util.List;

import org.junit.Test;

import cz.muni.neural.network.model.LabeledPoint;
import cz.muni.neural.network.model.Result;
import cz.muni.neural.network.util.CSVReader;
import cz.muni.neural.network.util.OHLCReader;
import cz.muni.neural.network.util.TestUtils;
import cz.muni.neural.network.util.Utils;

/**
 * @author Pavol Loffay
 */
public class TimeSeriesPredictionTest {

    
    @Test
    public void testOnCSVPrediction() throws IOException {

        int TRAIN = 500;
        int TEST = 100; 
        double ALPHA = 0.5;
        int ITER = 200;
        boolean REGULARIZE = true;
        double LAMBDA = 1;

        List<LabeledPoint> trainPoints = CSVReader.read(TestUtils.CSV_TRAIN_PATH, ";", TRAIN);
        
        double mean = Utils.mean(trainPoints);
        double deviation = Utils.deviation(trainPoints, mean);
        
        trainPoints = Utils.sigmoidNormalize(trainPoints, mean, deviation);
        
        int features = trainPoints.get(0).getFeatures().length;

        NeuralNetwork network = NeuralNetwork.newBuilder()
                .withGradientAlpha(ALPHA)
                .withGradientIterations(ITER)
                .withRegularize(REGULARIZE)
                .withRegularizeLambda(LAMBDA)
                .withClassify(false)
                .withInputLayer(features)
                .addLayer(15)
                .addLastLayer(1);

        /**
         * train
         */
        network.train(trainPoints);

        /**
         * test
         */
        List<LabeledPoint> testPoints = CSVReader.read(TestUtils.CSV_TEST_PATH, ";", TEST);
        
        testPoints = Utils.sigmoidNormalize(testPoints, mean, deviation);
        

        double[] labels = new double[testPoints.size()];
        double[] predictions = new double[testPoints.size()];
        for (int i = 0; i < testPoints.size(); i++) {

            LabeledPoint labeledPoint = testPoints.get(i);
            Result result = network.predict(labeledPoint);

            System.out.println(result);
            System.out.println("Label = " + labeledPoint.getLabel() + " predicted = " + result.getData()[0]);  
            labels[i] = labeledPoint.getLabel();
            predictions[i] = result.getData()[0];
        }

        Double rmse = Utils.rmse(labels, predictions);

        System.out.println("Test examples = " + testPoints.size());
        System.out.println("RMSE of normalized data = " + rmse);

        assertThat(rmse,  is(lessThanOrEqualTo(new Double(1))));
    }
    
    @Test
    public void testOnCSVClassification() throws IOException {

        int TRAIN = 500;
        int TEST = 100; 
        double ALPHA = 0.5;
        int ITER = 200;
        boolean REGULARIZE = true;
        double LAMBDA = 1;

        //data already normalized in CSV
        List<LabeledPoint> trainPoints = CSVReader.read(TestUtils.CSV_CLASS_TRAIN_PATH, ";", TRAIN);
        int features = trainPoints.get(0).getFeatures().length;

        NeuralNetwork network = NeuralNetwork.newBuilder()
                .withGradientAlpha(ALPHA)
                .withGradientIterations(ITER)
                .withRegularize(REGULARIZE)
                .withRegularizeLambda(LAMBDA)
                .withClassify(true)
                .withInputLayer(features)
                .addLayer(30)
                .addLastLayer(3);

        /**
         * train
         */
        network.train(trainPoints);

        /**
         * test
         */
        List<LabeledPoint> testPoints = CSVReader.read(TestUtils.CSV_CLASS_TEST_PATH, ";", TEST);
        

        int ok = 0;
        for (LabeledPoint labeledPoint: testPoints) {

            Result result = network.predict(labeledPoint);

            System.out.println(result);
            System.out.println("Label = " + labeledPoint.getLabel() + " predicted = " + result.getMaxIndex());
            if (labeledPoint.getLabel() == result.getMaxIndex()) {
                ok++;
            }
        }

        Double success = (ok / (double)testPoints.size()) * 100D;

        System.out.println("\n\nSuccessfully predicted = " + ok);
        System.out.println("Test examples = " + testPoints.size());
        System.out.println("Success = " + success + "%");

        assertThat(success,  is(greaterThanOrEqualTo(new Double(60))));
    }
    
    @Test
    public void testOnOHLCPrediction() throws IOException {

        int FEATURES = 20;
        int TRAIN = 500;
        int TEST = 100; 
        double ALPHA = 0.5;
        int ITER = 200;
        boolean REGULARIZE = true;
        double LAMBDA = 1;

        List<LabeledPoint> allPoints = OHLCReader.read(TestUtils.OHLC_PATH, FEATURES, TRAIN+TEST, true, 300);
        
        if (allPoints.size() < TRAIN + TEST) {
            System.out.println("Not enough points!");
            return;
        }
        
        double mean = Utils.mean(allPoints);
        double deviation = Utils.deviation(allPoints, mean);
        
        List<LabeledPoint> trainPoints = allPoints.subList(0, TRAIN);
        List<LabeledPoint> testPoints = allPoints.subList(TRAIN, TRAIN+TEST);
        
        List<LabeledPoint> normalizedTrainPoints = Utils.sigmoidNormalize(trainPoints, mean, deviation);
        List<LabeledPoint> normalizedTestPoints = Utils.sigmoidNormalize(testPoints, mean, deviation);

        NeuralNetwork network = NeuralNetwork.newBuilder()
                .withGradientAlpha(ALPHA)
                .withGradientIterations(ITER)
                .withRegularize(REGULARIZE)
                .withRegularizeLambda(LAMBDA)
                .withClassify(false)
                .withInputLayer(FEATURES)
                .addLayer(15)
                .addLastLayer(1);

        /**
         * train
         */
        network.train(normalizedTrainPoints);
        

        double[] labels = new double[normalizedTestPoints.size()];
        double[] predictions = new double[normalizedTestPoints.size()];
        for (int i = 0; i < normalizedTestPoints.size(); i++) {

            LabeledPoint labeledPoint = normalizedTestPoints.get(i);
            Result result = network.predict(labeledPoint);

            System.out.println(result);
            System.out.println("Label = " + labeledPoint.getLabel() + " predicted = " + result.getData()[0]);  
            labels[i] = labeledPoint.getLabel();
            predictions[i] = result.getData()[0];
        }

        Double rmse = Utils.rmse(labels, predictions);
        
        System.out.println("Test examples = " + normalizedTestPoints.size());
        System.out.println("RMSE of normalized data = " + rmse);

        assertThat(rmse,  is(lessThanOrEqualTo(new Double(0.1))));
    }
}
