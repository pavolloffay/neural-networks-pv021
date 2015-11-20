package cz.muni.neural.network;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

/**
 * @author Pavol Loffay
 */
public class ListSetTest {

    @Test
    public void testList() {
        List<Double> list = new ArrayList<>();

        list.add(9D);
        list.add(5D);

        list.set(0, 0D);

        assertThat(list.get(0), is(0D));
    }
}
