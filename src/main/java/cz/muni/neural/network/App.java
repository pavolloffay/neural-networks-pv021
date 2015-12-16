package cz.muni.neural.network;

import java.io.IOException;
import java.util.List;

import org.apache.commons.cli.ParseException;

import cz.muni.neural.network.model.LabeledPoint;
import cz.muni.neural.network.model.Result;
import cz.muni.neural.network.util.CSVReader;
import cz.muni.neural.network.util.MNISTReader;
import cz.muni.neural.network.util.OHLCReader;
import cz.muni.neural.network.util.TestUtils;
import cz.muni.neural.network.util.Utils;

/**
 * @author Pavol Loffay
 */
public class App {
    private static String OHLC_PATH = TestUtils.OHLC_PATH;

    private static int FEATURES = 20;

    public static void main(String[] args) throws IOException, ParseException {

        CliArgumentsParser cliArgumentParser = new CliArgumentsParser(args);

        CliArgumentsParser.Params cliParams = null;
        try {
            cliParams = cliArgumentParser.parse();

            if (cliParams.isHelp()) {
                System.out.println(CliArgumentsParser.helpMessage);
            }
        } catch (ParseException | NumberFormatException ex) {
            System.out.println("\n\nError parsing Parameters!\n");
            System.out.println(CliArgumentsParser.helpMessage);
            return;
        }

        if (cliParams.isNumbers()) {
            numbersClassificationExperiment(cliParams);
        } else if (cliParams.isNumbers() == false && cliParams.isHelp() == false) {
            OHLCEBaseline(cliParams);
            OHLCExperiment(cliParams);
        }
    }

    /**
     * Predict just by using last value
     *
     * @throws IOException
     */
    private static void OHLCEBaseline(CliArgumentsParser.Params params) throws IOException {

        int testSize = params.getSizeTest() == null ? 70000 : params.getSizeTest().intValue();
        int period = params.getPeriod() == null ? 60 : params.getPeriod();
        String file = params.getTestFile() == null ? TestUtils.OHLC_PATH : params.getTestFile();

        List<LabeledPoint> testPoints = OHLCReader.read(file, FEATURES, testSize, true, period);

        if (testPoints.size() < testSize) {
            System.out.println("Not enough points in file!");
            return;
        }

        double[] labels = new double[testSize];
        double[] predictions = new double[testSize];
        for (int i = 0; i < testSize; i++) {

            LabeledPoint labeledPoint = testPoints.get(i);

            labels[i] = labeledPoint.getLabel();
            predictions[i] = labeledPoint.getFeatures()[FEATURES - 1];
        }


        Double rmse = Utils.rmse(labels, predictions);

        System.out.println("Test examples = " + testPoints.size());
        System.out.printf("RMSE: %.9f", rmse);
        System.out.println();
    }

    private static void OHLCExperiment(CliArgumentsParser.Params params) throws IOException {

        int trainSize = params.getSizeTrain() == null ? 280000 : params.getSizeTrain().intValue();
        int testSize = params.getSizeTest() == null ? 69000 : params.getSizeTest().intValue();
        int[] layers = params.getLayers() == null ? new int[]{ 15, 1 } : params.getLayers();
        int period = params.getPeriod() == null ? 60 : params.getPeriod();
        String file = params.getTestFile() == null ? TestUtils.OHLC_PATH : params.getTestFile();

        int features = 20;
        NeuralNetwork neuralNetwork = buildNeuralNetworkWithLayers(features, layers, params.getNetworkBuilder());

        List<LabeledPoint> allPoints = OHLCReader.read(file, features, trainSize + testSize, true, period);

        if (allPoints.size() < trainSize + testSize) {
            System.out.println("Not enough points in file!");
            return;
        }

        double mean = Utils.mean(allPoints);
        double deviation = Utils.deviation(allPoints, mean);

        List<LabeledPoint> trainPoints = allPoints.subList(0, testSize);
        List<LabeledPoint> testPoints = allPoints.subList(trainSize, trainSize + testSize);

        System.out.println("Normalizing. Mean: " + mean + ", Deviation: " + deviation);

        List<LabeledPoint> normalizedTrainPoints = Utils.sigmoidNormalize(trainPoints, mean, deviation);
        List<LabeledPoint> normalizedTestPoints = Utils.sigmoidNormalize(testPoints, mean, deviation);

        System.out.println("Running experiment with network:\n %s" + neuralNetwork.toString());
        neuralNetwork.train(normalizedTrainPoints);

        double[] labels = new double[normalizedTestPoints.size()];
        double[] predictions = new double[normalizedTestPoints.size()];
        for (int i = 0; i < normalizedTestPoints.size(); i++) {

            LabeledPoint normalizedLabeledPoint = normalizedTestPoints.get(i);
            LabeledPoint labeledPoint = testPoints.get(i);
            Result result = neuralNetwork.predict(normalizedLabeledPoint);

            double prediction = Utils.sigmoidDenormalize(result.getData()[0], mean, deviation);

            System.out.println(result);
            System.out.println("Label = " + labeledPoint.getLabel() + " predicted = " + prediction);
            labels[i] = labeledPoint.getLabel();
            predictions[i] = prediction;
        }

        Double rmse = Utils.rmse(labels, predictions);

        try {
            CSVReader.write(TestUtils.CSV_RESULT_PATH, ";", labels, predictions);
        } catch (Exception e) {
            System.out.println("Result file writing failed.");
        }

        System.out.println("Test examples = " + normalizedTestPoints.size());
        System.out.printf("RMSE: %.9f\n", rmse);
    }

    public static void numbersClassificationExperiment(CliArgumentsParser.Params params) throws IOException {

        int trainSize = params.getSizeTrain() == null ? 500 : params.getSizeTrain().intValue();
        int testSize = params.getSizeTest() == null ? 50 : params.getSizeTest().intValue();
        int[] layers = params.getLayers() == null ? new int[] {30, 10} : params.getLayers();

        List<LabeledPoint> trainPoints = MNISTReader.read(TestUtils.IMAGES_TRAIN_PATH,
                TestUtils.LABELS_TRAIN_PATH, trainSize);

        int features = trainPoints.get(0).getFeatures().length;
        NeuralNetwork neuralNetwork = buildNeuralNetworkWithLayers(features, layers, params.getNetworkBuilder());

        /**
         * train
         */
        System.out.println("Running experiment with network:\n %s" + neuralNetwork.toString());
        neuralNetwork.train(trainPoints);

        /**
         * test
         */
        List<LabeledPoint> testPoints = MNISTReader.read(TestUtils.IMAGES_TEST_PATH,
                TestUtils.LABELS_TEST_PATH, testSize);

        int ok = 0;
        for (LabeledPoint labeledPoint : testPoints) {

            Result result = neuralNetwork.predict(labeledPoint);

            System.out.println("Label = " + labeledPoint.getLabel() + " predicted = " + result.getMaxIndex());
            if (labeledPoint.getLabel() == result.getMaxIndex()) {
                ok++;
            }
        }

        Double success = (ok / (double) testPoints.size()) * 100D;

        System.out.println("\n\nSuccessfully predicted = " + ok);
        System.out.println("Test examples = " + testPoints.size());
        System.out.println("Success = " + success + "%");
    }

    private static NeuralNetwork buildNeuralNetworkWithLayers(int numberOfFeatures, int[] layers,
                                                       NeuralNetwork.Builder neuralBuilder) {

        NeuralNetwork.BuilderLayers builderLayers = neuralBuilder.withInputLayer(numberOfFeatures);
        for (int i = 0; i < layers.length - 1; i++) {
            builderLayers.addLayer(layers[i]);
        }

        return builderLayers.addLastLayer(layers[layers.length - 1]);
    }
}
