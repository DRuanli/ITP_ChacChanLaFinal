package application.config;

/**
 * Configuration for mining parameters.
 * Immutable configuration object following Builder pattern.
 */
public class MiningConfiguration {
    private final String databasePath;
    private final double tau;
    private final int k;
    private final String algorithmName;
    private final boolean verbose;
    //private final PruningConfiguration pruningConfig;

    private MiningConfiguration(Builder builder) {
        this.databasePath = builder.databasePath;
        this.tau = builder.tau;
        this.k = builder.k;
        this.algorithmName = builder.algorithmName;
        this.verbose = builder.verbose;
        //this.pruningConfig = builder.pruningConfig;
    }

    // Getters
    public String getDatabasePath() { return databasePath; }
    public double getTau() { return tau; }
    public int getK() { return k; }
    public String getAlgorithmName() { return algorithmName; }
    public boolean isVerbose() { return verbose; }
    //public PruningConfiguration getPruningConfig() { return pruningConfig; }

    @Override
    public String toString() {
        return String.format("MiningConfig[db=%s, tau=%.2f, k=%d, algo=%s]",
                           databasePath, tau, k, algorithmName);
    }

    /**
     * Builder for MiningConfiguration
     */
    public static class Builder {
        private String databasePath;
        private double tau = 0.7;
        private int k = 10;
        private String algorithmName = "TUFCI";
        private boolean verbose = false;
        //private PruningConfiguration pruningConfig = PruningConfiguration.allEnabled();

        public Builder databasePath(String path) {
            this.databasePath = path;
            return this;
        }

        public Builder tau(double tau) {
            this.tau = tau;
            return this;
        }

        public Builder k(int k) {
            this.k = k;
            return this;
        }

        public Builder algorithmName(String name) {
            this.algorithmName = name;
            return this;
        }

        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }

        //public Builder pruningConfig(PruningConfiguration config) {
        //    this.pruningConfig = config;
        //    return this;
        //}

        public MiningConfiguration build() {
            if (databasePath == null || databasePath.isEmpty()) {
                throw new IllegalArgumentException("Database path is required");
            }
            if (tau <= 0 || tau > 1) {
                throw new IllegalArgumentException("Tau must be in (0, 1]");
            }
            if (k <= 0) {
                throw new IllegalArgumentException("K must be positive");
            }
            return new MiningConfiguration(this);
        }
    }
}