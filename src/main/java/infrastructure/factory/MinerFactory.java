package infrastructure.factory;

import application.config.MiningConfiguration;
import domain.mining.AbstractFrequentItemsetMiner;
import domain.mining.TUFCI;
import domain.mining.TUFCI_DFS;
import domain.mining.TUFCI_BFS;
import domain.support.SupportCalculator;
import infrastructure.persistence.UncertainDatabase;

/**
 * Factory for creating TUFCI Miner instances.
 *
 * Supports three search strategies:
 * - Best-First Search (TUFCI) - recommended for production
 * - Depth-First Search (TUFCI_DFS) - for comparison/educational purposes
 * - Breadth-First Search (TUFCI_BFS) - for comparison/educational purposes
 *
 * @author Dang Nguyen Le
 */
public class MinerFactory {

    /**
     * Search strategy enumeration.
     */
    public enum SearchStrategy {
        /** Best-First Search - uses PriorityQueue, optimal for Top-K */
        BEST_FIRST,

        /** Depth-First Search - uses Stack (LIFO), explores depth before breadth */
        DFS,

        /** Breadth-First Search - uses Queue (FIFO), explores level-by-level */
        BFS
    }

    /**
     * Create TUFCI Miner with Best-First Search (default).
     *
     * @param database uncertain database to mine
     * @param tau probability threshold (0 < tau <= 1)
     * @param k number of top itemsets to return
     * @return configured TUFCI instance
     */
    public static TUFCI createMiner(
        UncertainDatabase database,
        double tau,
        int k
    ) {
        return new TUFCI(database, tau, k);
    }

    /**
     * Create miner with specified search strategy.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold (0 < tau <= 1)
     * @param k number of top itemsets to return
     * @param strategy search strategy (BEST_FIRST, DFS, or BFS)
     * @return configured miner instance
     */
    public static AbstractFrequentItemsetMiner createMiner(
        UncertainDatabase database,
        double tau,
        int k,
        SearchStrategy strategy
    ) {
        return switch (strategy) {
            case BEST_FIRST -> new TUFCI(database, tau, k);
            case DFS -> new TUFCI_DFS(database, tau, k);
            case BFS -> new TUFCI_BFS(database, tau, k);
        };
    }

    /**
     * Create miner with custom support calculator.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold
     * @param k number of top itemsets to return
     * @param calculator custom support calculation strategy
     * @return configured TUFCI instance
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
     * Create miner with custom support calculator and search strategy.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold
     * @param k number of top itemsets to return
     * @param calculator custom support calculation strategy
     * @param strategy search strategy (BEST_FIRST, DFS, or BFS)
     * @return configured miner instance
     */
    public static AbstractFrequentItemsetMiner createMiner(
        UncertainDatabase database,
        double tau,
        int k,
        SupportCalculator calculator,
        SearchStrategy strategy
    ) {
        return switch (strategy) {
            case BEST_FIRST -> new TUFCI(database, tau, k, calculator);
            case DFS -> new TUFCI_DFS(database, tau, k, calculator);
            case BFS -> new TUFCI_BFS(database, tau, k, calculator);
        };
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

    /**
     * Create DFS miner (convenience method).
     *
     * @param database uncertain database to mine
     * @param tau probability threshold
     * @param k number of top itemsets to return
     * @return configured TUFCI_DFS instance
     */
    public static TUFCI_DFS createDFSMiner(
        UncertainDatabase database,
        double tau,
        int k
    ) {
        return new TUFCI_DFS(database, tau, k);
    }

    /**
     * Create BFS miner (convenience method).
     *
     * @param database uncertain database to mine
     * @param tau probability threshold
     * @param k number of top itemsets to return
     * @return configured TUFCI_BFS instance
     */
    public static TUFCI_BFS createBFSMiner(
        UncertainDatabase database,
        double tau,
        int k
    ) {
        return new TUFCI_BFS(database, tau, k);
    }

}