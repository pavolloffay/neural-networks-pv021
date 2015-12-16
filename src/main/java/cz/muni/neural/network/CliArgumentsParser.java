package cz.muni.neural.network;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

/**
 * @author Pavol Loffay
 */
public class CliArgumentsParser {

    public static final String helpMessage =
            "SYNOPSIS\n"
                    + "    App -e <experiment> [options]\n"
                    + "OPTIONS\n"
                    + "    -alpha   <num>      learning rate\n"
                    + "    -n       <num>      number of iterations\n"
                    + "    -layers    28 30 1  network architecture (without input layer)\n\n"
                    + "    -lambda  <num>      regularization parameter lambda\n"
                    + "    -s_train <num>      number of learn points\n"
                    + "    -s_test  <num>      number of test points\n"
                    + "    -file <num>         test and train file, only for time series\n"
                    + "    -period <num>       period, only for time series\n"
                    + "    -h                  help\n";

    private final String[] args;
    private final Options options = new Options();
    private final CommandLineParser parser = new DefaultParser();


    public CliArgumentsParser(String[] args) {
        this.args = args;

        // experiment
        options.addOption(Option.builder("e").type(String.class)
                .hasArg().desc("experiment - [numbers, ts]").required().build());

        // gradient
        options.addOption(Option.builder("alpha").type(Long.class)
                .hasArg().desc("gradient descent learning rate").build());
        options.addOption(Option.builder("n").type(Long.class)
                .hasArg().desc("number of gradient descent iterations").build());
        
        // regularization
        options.addOption(Option.builder("lambda").type(Long.class)
                .hasArg().desc("regularization lambda").build());

        // experiment size
        options.addOption(Option.builder("s_train").type(Long.class)
                .hasArg().desc("number of train points").build());
        options.addOption(Option.builder("s_test").type(Long.class)
                .hasArg().desc("number of test points").build());

        // files
        options.addOption(Option.builder("file").type(Long.class)
                .hasArg().desc("train and test file for OHLC time series").build());

        // layers
        options.addOption(Option.builder("layers")
                .numberOfArgs(Option.UNLIMITED_VALUES).desc("network architecture, first layer is input, last output").build());

        // period
        options.addOption(Option.builder("period").type(Long.class)
                .hasArg().desc("period of OHLC data in seconds").build());

        options.addOption(Option.builder("h")
                .desc("Help").build());
    }

    public Params parse() throws ParseException {
        final CommandLine params = parser.parse(options, args);

        String experiment = params.getOptionValue("e", "numbers");
        Params resultParams = experimentDefaultSettings(experiment);

        customParameters(params, resultParams);

        return resultParams;
    }

    public Params experimentDefaultSettings(String experiment) {

        Params resultParams = new Params();
        resultParams.networkBuilder = NeuralNetwork.newBuilder();
        if (experiment.equals("numbers")) {
            resultParams.networkBuilder.withGradientAlpha(0.1)
                   .withGradientIterations(200)
                   .withClassify(true);

            resultParams.numbers = true;
        } else {
            resultParams.networkBuilder.withGradientAlpha(1)
                   .withGradientIterations(150)
                   .withClassify(false);

            resultParams.numbers = false;
        }

        return resultParams;
    }

    public Params customParameters(CommandLine params, Params resultParams) {

        if (params.hasOption("h")) {
            resultParams.help = true;
            return resultParams;
        }

        if (params.hasOption("s_train")) {
            resultParams.sizeTrain = Long.parseLong(params.getOptionValue("s_train"));
        }
        if (params.hasOption("s_test")) {
            resultParams.sizeTest = Long.parseLong(params.getOptionValue("s_test"));
        }

        if (params.hasOption("file")) {
            resultParams.testFile = params.getOptionValue("file");
        }

        if (params.hasOption("alpha")) {
            Double alpha = Double.parseDouble(params.getOptionValue("alpha"));
            resultParams.networkBuilder.withGradientAlpha(alpha);
        }
        if (params.hasOption("n")) {
            Long n = Long.parseLong(params.getOptionValue("n"));
            resultParams.networkBuilder.withGradientIterations(n);
        }
        if (params.hasOption("lambda")) {
            Double lambda = Double.parseDouble(params.getOptionValue("lambda"));
            resultParams.networkBuilder.withRegularize(true)
                    .withRegularizeLambda(lambda);
        }

        if (params.hasOption("layers")) {
            String[] layers = params.getOptionValues("layers");
            buildLayers(layers, resultParams);
        }

        if (params.hasOption("period")) {
            Integer period = Integer.parseInt(params.getOptionValue("period"));
            resultParams.period = period;
        }

        return resultParams;
    }

    private void buildLayers(String[] layers, Params params) {
        params.layers = new int[layers.length];
        for (int i = 0; i < layers.length; i++) {
            int layer = Integer.parseInt(layers[i]);

            params.layers[i] = layer;
        }
    }

    public class Params {
        private NeuralNetwork.Builder networkBuilder;
        private int[] layers;
        private Long sizeTrain;
        private Long sizeTest;
        private String trainFile;
        private String testFile;
        private Integer period;

        private boolean help;
        private boolean numbers;

        public Long getSizeTrain() {
            return sizeTrain;
        }

        public void setSizeTrain(Long sizeTrain) {
            this.sizeTrain = sizeTrain;
        }

        public Long getSizeTest() {
            return sizeTest;
        }

        public void setSizeTest(Long sizeTest) {
            this.sizeTest = sizeTest;
        }

        public String getTrainFile() {
            return trainFile;
        }

        public void setTrainFile(String trainFile) {
            this.trainFile = trainFile;
        }

        public String getTestFile() {
            return testFile;
        }

        public void setTestFile(String testFile) {
            this.testFile = testFile;
        }

        public boolean isNumbers() {
            return numbers;
        }

        public void setNumbers(boolean numbers) {
            this.numbers = numbers;
        }

        public boolean isHelp() {
            return help;
        }

        public void setHelp(boolean help) {
            this.help = help;
        }

        public NeuralNetwork.Builder getNetworkBuilder() {
            return networkBuilder;
        }

        public void setNetworkBuilder(NeuralNetwork.Builder networkBuilder) {
            this.networkBuilder = networkBuilder;
        }

        public int[] getLayers() {
            return layers;
        }

        public void setLayers(int[] layers) {
            this.layers = layers;
        }

        public Integer getPeriod() {
            return period;
        }

        public void setPeriod(Integer period) {
            this.period = period;
        }
    }
}
