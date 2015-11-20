package cz.muni.neural.network.model;

import java.util.List;

import cz.muni.neural.network.matrix.DoubleMatrix;

/**
 * @author Pavol Loffay
 */
public class ForwardPropagationResult {

    private List<DoubleMatrix> activations;
    private List<DoubleMatrix> zetas;


    public ForwardPropagationResult(List<DoubleMatrix> activations,
                                    List<DoubleMatrix> zetas) {
        this.activations = activations;
        this.zetas = zetas;
    }

    public List<DoubleMatrix> getActivations() {
        return activations;
    }

    public List<DoubleMatrix> getZetas() {
        return zetas;
    }

    /**
     * returns last activation = result of network
     * @return
     */
    public DoubleMatrix getLastActivation() {
        return activations.get(activations.size() - 1);
    }
}
