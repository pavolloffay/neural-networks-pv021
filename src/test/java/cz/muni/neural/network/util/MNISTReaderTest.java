package cz.muni.neural.network.util;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

import java.util.List;

import org.junit.Test;

import cz.muni.neural.network.LabeledPoint;
import cz.muni.neural.network.TestUtils;

/**
 * @author Pavol Loffay
 */
public class MNISTReaderTest {

    @Test
    public void testRead() throws Exception {

        int count = 10;
        List<LabeledPoint> labeledPointList = MNISTReader.read(TestUtils.IMAGES_TEST_PATH,
                TestUtils.LABELS_TEST_PATH, count);

//        labeledPointList.forEach(x -> System.out.println(x));

        assertThat(labeledPointList.size(), is(count));
    }
}
