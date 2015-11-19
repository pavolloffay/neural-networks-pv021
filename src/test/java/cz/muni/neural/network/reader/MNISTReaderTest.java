package cz.muni.neural.network.reader;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;


import java.net.URL;
import java.util.List;

import org.junit.Test;

import cz.muni.neural.network.LabeledPoint;

/**
 * @author Pavol Loffay
 */
public class MNISTReaderTest {

    public static final String IMAGES_TEST = "t10k-images-idx3-ubyte";
    public static final String LABELS_TEST = "t10k-labels-idx1-ubyte";

    public static final String IMAGES_TRAIN = "train-images-idx3-ubyte";
    public static final String LABELS_TRAIN = "train-labels-idx1-ubyte";

    public static final String IMAGES_TEST_PATH;
    public static final String LABELS_TEST_PATH;
    public static final String IMAGES_TRAIN_PATH;
    public static final String LABELS_TRAIN_PATH;
    static {
        ClassLoader classLoader = MNISTReaderTest.class.getClassLoader();

        URL testFile = classLoader.getResource(IMAGES_TEST);
        IMAGES_TEST_PATH = testFile.getPath();
        testFile = classLoader.getResource(LABELS_TEST);
        LABELS_TEST_PATH = testFile.getPath();
        testFile = classLoader.getResource(LABELS_TRAIN);
        LABELS_TRAIN_PATH = testFile.getPath();
        testFile = classLoader.getResource(IMAGES_TRAIN);
        IMAGES_TRAIN_PATH = testFile.getPath();
    }

    @Test
    public void testRead() throws Exception {

        int count = 10;
        List<LabeledPoint> labeledPointList = MNISTReader.read(IMAGES_TEST_PATH, LABELS_TEST_PATH, count);

//        labeledPointList.forEach(x -> System.out.println(x));

        assertThat(labeledPointList.size(), is(count));
    }
}
