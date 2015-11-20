package cz.muni.neural.network;

/**
 * @author Pavol Loffay
 */
public class Layer {

    private int numberOfUnits;

    public Layer(int neurons) {
        this.numberOfUnits = neurons;
    }

    public int getNumberOfUnits() {
        return numberOfUnits;
    }
}
