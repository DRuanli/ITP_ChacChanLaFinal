package infrastructure.factory;

import application.config.MiningConfiguration;
import domain.mining.TUFCI;
import domain.support.SupportCalculator;
import infrastructure.persistence.UncertainDatabase;

/**
 * Factory for creating Closure-Aware Top-K Miner instances
 *
 * @author Dang Nguyen Le
 */
public class MinerFactory {

    /**
     * Create Closure-Aware Top-K Miner with default configuration.
     *
     * The minimum support threshold is dynamically derived from the Top-K heap.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold (0 < tau <= 1)
     * @param k number of top itemsets to return
     * @return configured ClosureAwareTopKMiner instance
     */
    public static TUFCI createMiner(
        UncertainDatabase database,
        double tau,
        int k
    ) {
        return new TUFCI(database, tau, k);
    }

    /**
     * Create miner with custom support calculator.
     *
     * The minimum support threshold is dynamically derived from the Top-K heap.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold
     * @param k number of top itemsets to return
     * @param calculator custom support calculation strategy
     * @return configured ClosureAwareTopKMiner instance
     */
    public static TUFCI createMiner(
        UncertainDatabase database,
        double tau,
        int k,
        SupportCalculator calculator
    ) {
        return new TUFCI(database, tau, k, calculator);
    }


    /**
     * Create miner from full MiningConfiguration.
     *
     * @param database uncertain database to mine
     * @param config complete mining configuration
     * @return configured TUFCI instance
     */
    public static TUFCI createMiner(
        UncertainDatabase database,
        MiningConfiguration config
    ) {
        return new TUFCI(database, config.getTau(), config.getK());
    }

}
