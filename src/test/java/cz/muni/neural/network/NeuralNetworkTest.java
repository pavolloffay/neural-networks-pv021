package cz.muni.neural.network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import cz.muni.neural.network.reader.MNISTReader;
import cz.muni.neural.network.reader.MNISTReaderTest;

/**
 * @author Pavol Loffay
 */
public class NeuralNetworkTest {

    private static int NUMBER_OF_TRAINING_EXAMPLES = 50;

    private static double GRADIENT_ALPHA = 0.005;
    private static long GRADIENT_NUM_ITER = 50;

    @Test
    public void testCreation() throws Exception {

        List<LabeledPoint> labeledPointList = MNISTReader.read(MNISTReaderTest.IMAGES_TEST_PATH,
                MNISTReaderTest.LABELS_TEST_PATH, NUMBER_OF_TRAINING_EXAMPLES);

        int features = labeledPointList.get(0).getFeatures().length;

        //layers
        List<Layer> layers = new ArrayList<>();
        Layer layer1 = new Layer(features);
        Layer layer2 = new Layer(25);
        Layer layer3 = new Layer(10);
        layers.add(layer1);
        layers.add(layer2);
        layers.add(layer3);

        NeuralNetwork network = new NeuralNetwork(layers, GRADIENT_ALPHA, GRADIENT_NUM_ITER);
        network.train(labeledPointList);
    }

    @Test
    public void testWithDummyData() {
        List<LabeledPoint> labeledPoints = createPoints(50, 15);
        List<Layer> layers = create3Layers(labeledPoints.get(0).getFeatures().length);

        NeuralNetwork network = new NeuralNetwork(layers, GRADIENT_ALPHA, GRADIENT_NUM_ITER);
        network.train(labeledPoints);
    }

    public static List<Layer> create3Layers(int features) {
        List<Layer> layers = new ArrayList<>();
        Layer layer1 = new Layer(features);
        Layer layer2 = new Layer(25);
        Layer layer3 = new Layer(10);
        layers.add(layer1);
        layers.add(layer2);
        layers.add(layer3);

        return layers;
    }

    public static List<LabeledPoint> createPoints(int number, int features) {

        List<LabeledPoint> labeledPoints = new ArrayList<>();

        Random rand = new Random();
        double[] x = new double[features];
        for (int i = 0; i < features; i++) {
            x[i] = rand.nextDouble() + 5;
        }

        for (int i = 0; i < number; i++) {
            LabeledPoint labeledPoint = new LabeledPoint(5, Arrays.copyOf(x, features));
            labeledPoints.add(labeledPoint);
        }

        return labeledPoints;
    }
}
