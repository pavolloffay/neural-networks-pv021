package cz.muni.neural.network;

import java.util.function.Function;

/**
 * @author Pavol Loffay
 */
public class Functions {

    public static class Sigmoid implements Function<Double, Double> {

        @Override
        public Double apply(Double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
    }

    public static class SigmoidGradient implements Function<Double, Double> {

        @Override
        public Double apply(Double x) {
            return (1.0 / (1.0 + Math.exp(-x))) * (1 - (1.0 / (1.0 + Math.exp(-x))));
        }
    }
    
    /**
     * Inverse sigmoid
     */
    public static class Logit implements Function<Double, Double> {

        @Override
        public Double apply(Double x) {
            return -Math.log((1/x) - 1);
        }
    }

}
