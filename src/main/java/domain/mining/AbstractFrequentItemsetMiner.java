package domain.mining;

import infrastructure.persistence.UncertainDatabase;
import domain.model.FrequentItemset;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * AbstractFrequentItemsetMiner - Template Method Pattern for mining algorithms.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * TEMPLATE METHOD PATTERN
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * The Template Method Pattern defines the skeleton of an algorithm in a base class,
 * letting subclasses override specific steps without changing structure.
 *
 * Structure:
 *   AbstractFrequentItemsetMiner (this class)
 *       │
 *       ├── mine()                    ← Template Method (final, defines skeleton)
 *       │     ├── computeAllSingletonSupports()    ← Abstract (subclass implements)
 *       │     ├── initializeTopKWithClosedSingletons()    ← Abstract (subclass implements)
 *       │     ├── performBestFirstMining()      ← Abstract (subclass implements)
 *       │     └── getTopKResults()              ← Abstract (subclass implements)
 *       │
 *       └── ClosureAwareTopKMiner (subclass)
 *             └── Implements all abstract methods
 *
 * Benefits:
 *   1. Code reuse: Common logic (timing, observers) in base class
 *   2. Inversion of control: Base class calls subclass methods
 *   3. Easy extension: Add new mining algorithms by subclassing
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * THREE-PHASE MINING ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Phase 1: COMPUTE ALL 1-ITEMSETS
 *   - Scan database to compute support for all single items
 *   - No filtering by minimum support (threshold derived dynamically)
 *   - Typically runs in parallel for performance
 *   - Output: List of all 1-itemset patterns (sorted by support)
 *
 * Phase 2: INITIALIZE DATA STRUCTURES & FILL TOP-K
 *   - Check closure of 1-itemsets and fill Top-K heap with closed ones
 *   - Build 2-itemset cache during closure checking
 *   - Derive dynamic minimum support threshold from Top-K heap
 *   - Prune 1-itemsets below dynamic threshold
 *   - Initialize priority queue with all remaining 1-itemsets
 *
 * Phase 3: RECURSIVE MINING
 *   - Main mining loop using priority queue
 *   - Generate candidates, check closure, update Top-K heap
 *   - Continue until early termination or exhaustion
 *
 * @author Dang Nguyen Le
 */
public abstract class AbstractFrequentItemsetMiner {

    /**
     * The uncertain database to mine.
     * Contains transactions with probabilistic item occurrences.
     */
    protected UncertainDatabase database;

    /**
     * Probability threshold τ (tau).
     * Pattern is frequent if P(support ≥ s) ≥ τ.
     */
    protected double tau;

    /**
     * Number of top patterns to return.
     * Mining finds the K patterns with highest support.
     */
    protected int k;

    /**
     * Constructor with parameter validation.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold (0 < τ ≤ 1)
     * @param k number of top patterns to find (≥ 1)
     * @throws IllegalArgumentException if parameters are invalid
     */
    public AbstractFrequentItemsetMiner(UncertainDatabase database, double tau, int k) {
        // Validate all parameters before storing
        validateParameters(database, tau, k);

        this.database = database;
        this.tau = tau;
        this.k = k;
    }

    /**
     * Validate mining parameters.
     *
     * @throws IllegalArgumentException if any parameter is invalid
     */
    private void validateParameters(UncertainDatabase database, double tau, int k) {
        // Database must exist
        if (database == null) {
            throw new IllegalArgumentException("Database cannot be null");
        }

        // Tau must be in (0, 1] - probability threshold
        // tau = 0 would accept any support (meaningless)
        // tau > 1 is impossible (probability cannot exceed 1)
        if (tau <= 0 || tau > 1) {
            throw new IllegalArgumentException("Tau must be in (0, 1]");
        }

        // K must be at least 1 (find at least one pattern)
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1");
        }
    }

    /**
     * ═══════════════════════════════════════════════════════════════════════════
     * TEMPLATE METHOD: Defines the mining algorithm skeleton
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * This method is FINAL - subclasses cannot override it.
     * The algorithm structure is fixed; only specific steps vary.
     *
     * Algorithm:
     *   1. Phase 1: Compute frequent 1-itemsets
     *   2. Phase 2: Initialize data structures
     *   3. Phase 3: Perform recursive mining
     *   4. Return top-K results
     *
     * Each phase is timed and observers are notified.
     *
     * @return list of top-K frequent closed itemsets
     */
    public final List<FrequentItemset> mine() {
        // ═══════════════════════════════════════════════════════════════
        // PHASE 1: Compute ALL 1-itemsets (no filtering)
        // ═══════════════════════════════════════════════════════════════
        long start1 = System.nanoTime();

        // Subclass implements this: scans database for frequent single items
        List<FrequentItemset> frequent1Itemsets = computeAllSingletonSupports();

        long phase1Time = (System.nanoTime() - start1) / 1_000_000;  // Convert to ms

        // ═══════════════════════════════════════════════════════════════
        // PHASE 2: Initialize data structures and fill Top-K
        // ═══════════════════════════════════════════════════════════════
        long start2 = System.nanoTime();

        // Subclass implements this: builds PQ, caches, etc.
        initializeTopKWithClosedSingletons(frequent1Itemsets);

        long phase2Time = (System.nanoTime() - start2) / 1_000_000;

        // ═══════════════════════════════════════════════════════════════
        // PHASE 3: Recursive mining (main loop)
        // ═══════════════════════════════════════════════════════════════
        long start3 = System.nanoTime();

        // Subclass implements this: priority queue processing, closure checking
        performBestFirstMining(frequent1Itemsets);

        long phase3Time = (System.nanoTime() - start3) / 1_000_000;

        // ═══════════════════════════════════════════════════════════════
        // RETURN: Get final top-K results
        // ═══════════════════════════════════════════════════════════════
        return getTopKResults();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ABSTRACT METHODS - Subclasses must implement these
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Phase 1: Compute probabilistic support for ALL singleton (1-itemset) patterns.
     *
     * <p><b>Key Point:</b> This phase computes support for <b>ALL</b> vocabulary items
     * without filtering. No threshold-based pruning occurs in Phase 1.</p>
     *
     * <p><b>Subclass Responsibilities:</b></p>
     * <ul>
     *   <li>Create singleton itemset for each vocabulary item</li>
     *   <li>Compute probabilistic support using SupportCalculator</li>
     *   <li>Cache tidsets and support values for later phases</li>
     *   <li>Sort singletons by support descending</li>
     * </ul>
     *
     * <p><b>Performance:</b> Typically parallelized for efficiency.</p>
     *
     * <p><b>Note:</b> Dynamic minimum support threshold is derived in Phase 2 from
     * Top-K heap, not in this phase.</p>
     *
     * @return list of ALL singleton patterns (sorted by support DESC, no filtering)
     */
    protected abstract List<FrequentItemset> computeAllSingletonSupports();

    /**
     * Phase 2: Initialize Top-K heap with closed singleton patterns and prepare search.
     *
     * <p><b>Main Objectives:</b></p>
     * <ol>
     *   <li>Populate Top-K heap with closed 1-itemsets (establishes initial threshold)</li>
     *   <li>Seed priority queue for Phase 3 best-first search</li>
     *   <li>Build 2-itemset cache (side effect of closure checking)</li>
     * </ol>
     *
     * <p><b>Subclass Responsibilities:</b></p>
     * <ul>
     *   <li>Check closure of singletons (in support descending order)</li>
     *   <li>Insert closed singletons into Top-K heap</li>
     *   <li>Cache all 2-itemsets encountered during closure checking</li>
     *   <li>Derive dynamic minimum support threshold from Top-K heap</li>
     *   <li>Build frequent items array for canonical order</li>
     *   <li>Seed priority queue with promising 2-itemsets for Phase 3</li>
     * </ul>
     *
     * <p><b>Critical Work:</b> Closure checking of singletons is computationally intensive
     * as it generates O(n²) 2-itemsets where n = vocabulary size.</p>
     *
     * @param singletonPatterns all singleton patterns from Phase 1 (sorted by support DESC)
     */
    protected abstract void initializeTopKWithClosedSingletons(List<FrequentItemset> singletonPatterns);

    /**
     * Phase 3: Perform best-first search to discover larger closed itemsets.
     *
     * <p><b>Search Strategy:</b> BEST-FIRST (not depth-first or breadth-first)</p>
     * <ul>
     *   <li><b>Data Structure:</b> Priority queue ordered by support descending</li>
     *   <li><b>Ordering:</b> Always processes highest-support candidate next</li>
     *   <li><b>Iteration:</b> Iterative loop (while !queue.isEmpty()), NOT recursive</li>
     *   <li><b>Termination:</b> Early termination when best candidate < threshold</li>
     * </ul>
     *
     * <p><b>Why Best-First for Top-K:</b></p>
     * <ul>
     *   <li>High-support patterns enter Top-K first → threshold rises quickly</li>
     *   <li>More aggressive pruning earlier in the search</li>
     *   <li>Optimal for Top-K mining (proven in academic literature)</li>
     *   <li>Can terminate as soon as best candidate fails threshold test</li>
     * </ul>
     *
     * <p><b>Algorithm Pseudocode:</b></p>
     * <pre>
     * while (!priorityQueue.isEmpty()) {
     *     candidate = priorityQueue.poll()        // Get BEST (highest support)
     *     if (candidate.support < threshold) break  // Early termination
     *
     *     if (isClosed(candidate))
     *         topK.insert(candidate)              // Update Top-K
     *
     *     extensions = generateExtensions(candidate)
     *     priorityQueue.addAll(extensions)        // NOT recursive - just enqueue
     * }
     * </pre>
     *
     * <p><b>Subclass Responsibilities:</b></p>
     * <ul>
     *   <li>Poll candidates from priority queue in best-first order</li>
     *   <li>Check closure for each candidate (with multiple optimizations)</li>
     *   <li>Generate canonical extensions (maintaining search order)</li>
     *   <li>Insert closed patterns into Top-K heap</li>
     *   <li>Add promising extensions back to priority queue</li>
     * </ul>
     *
     * @param singletonPatterns singleton patterns from Phase 1 (may be used for reference)
     */
    protected abstract void performBestFirstMining(List<FrequentItemset> singletonPatterns);

    /**
     * Get final top-K results after mining.
     *
     * Subclass responsibility:
     *   - Extract patterns from top-K heap
     *   - Sort by support (descending)
     *   - Return final result list
     *
     * @return list of top-K closed frequent patterns
     */
    protected abstract List<FrequentItemset> getTopKResults();
}