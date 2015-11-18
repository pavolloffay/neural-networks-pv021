package cz.muni.neural.network;

/**
 * @author Pavol Loffay
 */
public class LabeledPoint {

    private double label;
    private double[] features;

    public LabeledPoint(double label, double[] features) {
        this.label = label;
        this.features = features;
    }


    public double getLabel() {
        return label;
    }

    public double[] getFeatures() {
        return features;
    }
}
