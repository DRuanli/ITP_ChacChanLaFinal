package domain.support;

import domain.model.Tidset;

/**
 * SupportCalculator - Strategy Pattern interface for probabilistic support computation.
 *
 * In uncertain databases, support is not a simple count but a PROBABILISTIC measure:
 *   - Each item has existence probability p in each transaction
 *   - Support follows a probability distribution, not a single value
 *   - Probabilistic support = max{s : P(support ≥ s) ≥ τ}
 *
 * Mathematical Background:
 *   For itemset X in uncertain database:
 *   - Each transaction t has probability p_t that X appears
 *   - Total support S = Σ(indicator variables), where each is Bernoulli(p_t)
 *   - P(S = s) follows Poisson Binomial distribution
 *   - Generating Function: Π(1-p_t + p_t·x) encodes this distribution
 *
 * @author Dang Nguyen Le
 */
public interface SupportCalculator {

    /**
     * Compute probabilistic support from transaction probability array.
     *
     * Definition: ProbSupport_τ(X) = max{s : P(support(X) ≥ s) ≥ τ}
     *
     * @param transactionProbs array where transactionProbs[t] = P(itemset in transaction t)
     * @param minRequiredSupport hint for early termination (may be ignored)
     * @return probabilistic support value
     *
     * Note: This method is part of interface contract but computeSupportAndProbability()
     *       or computeSupportAndProbabilitySparse() are preferred in practice.
     */
    int computeProbabilisticSupport(double[] transactionProbs, int minRequiredSupport);

    /**
     * Compute probability that itemset achieves at least given support.
     *
     * @param transactionProbs array of transaction probabilities
     * @param support target support value
     * @return P(support ≥ s), the frequentness at support level s
     *
     * Note: This method is part of interface contract but computeSupportAndProbability()
     *       or computeSupportAndProbabilitySparse() are preferred in practice.
     */
    double computeProbability(double[] transactionProbs, int support);

    /**
     * Get name of this calculator strategy (for logging/debugging).
     *
     * @return human-readable strategy name
     */
    String getStrategyName();

    /**
     * Compute both support and probability in single call.
     *
     * More efficient than calling computeProbabilisticSupport() and
     * computeProbability() separately because distribution is computed only once.
     *
     * @param transactionProbs array of transaction probabilities
     * @return array [support, probability] where:
     *         support = max s where P(support ≥ s) ≥ τ
     *         probability = P(support ≥ computed_support)
     */
    double[] computeSupportAndProbability(double[] transactionProbs);

    /**
     * Compute support and probability directly from sparse Tidset.
     *
     * OPTIMIZATION: Avoids allocating dense array for sparse itemsets.
     *
     * Memory comparison:
     *   - Dense: O(totalTransactions) array allocation
     *   - Sparse: O(tidset.size()) - much smaller for sparse itemsets
     *
     * Example: Database has 100,000 transactions, itemset appears in 500
     *   - Dense: allocate double[100000] = 800KB
     *   - Sparse: only process 500 entries = ~4KB
     *
     * Default implementation converts to dense (for calculators that don't
     * support sparse). Override for memory-efficient sparse computation.
     *
     * @param tidset sparse transaction ID set with probabilities
     * @param totalTransactions total number of transactions in database
     * @return array [support, probability]
     */
    default double[] computeSupportAndProbabilitySparse(Tidset tidset, int totalTransactions) {
        // Default: convert sparse tidset to dense probability array
        // Subclasses can override for memory-efficient sparse computation
        double[] probs = tidset.toTransactionProbabilities(totalTransactions);

        // Delegate to dense computation
        return computeSupportAndProbability(probs);
    }
}