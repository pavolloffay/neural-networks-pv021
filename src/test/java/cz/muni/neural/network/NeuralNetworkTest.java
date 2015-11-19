package cz.muni.neural.network;

import java.util.ArrayList;
import java.util.List;

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
    public void testTrain() throws Exception {

        List<LabeledPoint> labeledPointList = MNISTReader.read(MNISTReaderTest.IMAGES_PATH,
                MNISTReaderTest.LABELS_PATH, NUMBER_OF_TRAINING_EXAMPLES);

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
//        network.train(labeledPointList);
    }
}
