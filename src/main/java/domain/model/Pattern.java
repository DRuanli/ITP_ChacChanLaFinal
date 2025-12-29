package domain.model;


/**
 * Pattern - Represents a frequent itemset with its mining metrics.
 *
 * In Frequent Itemset Mining, a Pattern is the result of mining:
 *   - Itemset: the set of items (e.g., {Bread, Milk})
 *   - Support: how many transactions contain this itemset
 *   - Probability: P(support >= threshold) in uncertain databases
 *
 * This is an immutable value object (all fields are final).
 * Used as output of mining algorithm and stored in TopKHeap.
 *
 * @author Dang Nguyen Le
 */
public class Pattern {

    /**
     * The itemset (set of items) this pattern represents.
     * Example: {Bread, Milk, Butter}
     *
     * Public final: accessible but immutable (can't reassign)
     */
    public final Itemset itemset;

    /**
     * Probabilistic support of this itemset.
     *
     * Definition: ProbSupport_τ(X) = max{s : P(support(X) ≥ s) ≥ τ}
     *
     * In uncertain databases, each item has existence probability.
     * Support is the maximum count s such that the probability
     * of having at least s occurrences is >= threshold τ.
     *
     * Example: support=45 means itemset appears in 45 transactions
     *          with probability >= τ (e.g., 0.7)
     */
    public final int support;

    /**
     * Probability that itemset has at least 'support' occurrences.
     *
     * Definition: P(support(X) ≥ support)
     *
     * This is the frequentness value at the computed support level.
     * Always >= τ (threshold) by definition of probabilistic support.
     *
     * Example: probability=0.82 means 82% chance that this itemset
     *          appears in at least 'support' transactions.
     */
    public final double probability;

    /**
     * Constructor - creates immutable pattern.
     *
     * @param itemset    the itemset
     * @param support    probabilistic support value
     * @param probability probability of achieving support
     */
    public Pattern(Itemset itemset, int support, double probability) {
        this.itemset = itemset;
        this.support = support;
        this.probability = probability;
    }

    /**
     * Returns a string representation of this pattern.
     * Format: {items} [sup=X, prob=Y.YY]
     *
     * @return human-readable pattern representation
     */
    @Override
    public String toString() {
        return String.format("%s [sup=%d, prob=%.3f]",
            itemset.toString(), support, probability);
    }
}