package domain.mining;

import domain.support.SupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;
import domain.model.*;
import infrastructure.topK.TopKHeap;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

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
 *       │     ├── executePhase3()      ← Abstract (subclass implements)
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

    // ════════════════════════════════════════════════════════════════════════════
    // IMMUTABLE CONFIGURATION (private final - encapsulated)
    // These define the mining problem and should never change after construction
    // ════════════════════════════════════════════════════════════════════════════

    /**
     * The uncertain database to mine.
     * Contains transactions with probabilistic item occurrences.
     * IMMUTABLE: Set at construction, never changes.
     */
    private final UncertainDatabase database;

    /**
     * Probability threshold τ (tau).
     * Pattern is frequent if P(support ≥ s) ≥ τ.
     * IMMUTABLE: Set at construction, never changes.
     * INVARIANT: Must be in (0, 1]
     */
    private final double tau;

    /**
     * Number of top patterns to return.
     * Mining finds the K patterns with highest support.
     * IMMUTABLE: Set at construction, never changes.
     * INVARIANT: Must be >= 1
     */
    private final int k;

    /**
     * Support calculator for computing expected support from probability distributions.
     * Uses the Generating Function (GF) approach for efficient probabilistic support calculation.
     * IMMUTABLE: Set at construction, never changes.
     */
    private final SupportCalculator calculator;

    /**
     * Vocabulary containing all unique items in the database.
     * Used to create itemsets and manage item identifiers.
     * IMMUTABLE: Derived from database at construction.
     */
    private final Vocabulary vocab;

    // ════════════════════════════════════════════════════════════════════════════
    // MINING STATE (protected - direct subclass access for simplicity)
    // These are working data structures that subclasses need to manipulate
    // ════════════════════════════════════════════════════════════════════════════

    /**
     * Top-K heap that maintains the k best patterns found so far.
     * Automatically updates the minimum support threshold as better patterns are discovered.
     */
    protected TopKHeap topK;

    /**
     * Cache storing computed patterns to avoid redundant calculations.
     * Maps itemsets to their CachedFrequentItemset (support, probability, tidset).
     * This is crucial for performance as it enables O(1) lookup for previously computed patterns.
     */
    protected Map<Itemset, CachedFrequentItemset> cache;

    /**
     * Pre-computed singleton itemsets for all items in the vocabulary.
     * Avoids creating new Itemset objects repeatedly during mining.
     * Index corresponds to item ID in the vocabulary.
     */
    protected Itemset[] singletonCache;

    /**
     * Number of frequent single items that meet the minimum support threshold.
     * Used to limit the search space in Phase 3.
     */
    protected int frequentItemCount;

    /**
     * Array of item IDs that are frequent (meet minimum support).
     * Sorted by support in descending order for efficient pruning.
     */
    protected int[] frequentItems;

    /**
     * Constructor with parameter validation.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold (0 < τ ≤ 1)
     * @param k number of top patterns to find (≥ 1)
     * @throws IllegalArgumentException if parameters are invalid
     */
    /**
     * Constructor with custom support calculator.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold (0 < τ ≤ 1)
     * @param k number of top patterns to find (≥ 1)
     * @param calculator custom support calculation strategy
     * @throws IllegalArgumentException if parameters are invalid
     */
    public AbstractFrequentItemsetMiner(UncertainDatabase database, double tau, int k,
                                       SupportCalculator calculator) {
        // Validate all parameters before storing
        validateParameters(database, tau, k);

        // Validate calculator
        if (calculator == null) {
            throw new IllegalArgumentException("SupportCalculator cannot be null");
        }

        this.database = database;
        this.tau = tau;
        this.k = k;
        this.calculator = calculator;
        this.vocab = database.getVocabulary();
        this.cache = new HashMap<>();
    }

    /**
     * Constructor without custom calculator (deprecated - prefer 4-parameter version).
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
        this.calculator = null;  // Must be set later by subclass
        this.vocab = database.getVocabulary();
        this.cache = new HashMap<>();
    }

    /**
     * Validate mining parameters.
     *
     * @throws IllegalArgumentException if any parameter is invalid
     */
    private void validateParameters(UncertainDatabase database, double tau, int k) {
        // Database must exist and not be empty
        if (database == null) {
            throw new IllegalArgumentException("Database cannot be null");
        }
        if (database.size() == 0) {
            throw new IllegalArgumentException("Database cannot be empty");
        }

        // Tau must be in (0, 1] - probability threshold
        // tau = 0 would accept any support (meaningless)
        // tau > 1 is impossible (probability cannot exceed 1)
        if (tau <= 0 || tau > 1) {
            throw new IllegalArgumentException(
                String.format("Tau must be in (0, 1], got: %.4f", tau));
        }

        // K must be at least 1 (find at least one pattern)
        if (k < 1) {
            throw new IllegalArgumentException(
                String.format("k must be at least 1, got: %d", k));
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    // PROTECTED GETTERS FOR IMMUTABLE CONFIGURATION
    // Provide controlled read-only access to private final fields
    // These are final - subclasses cannot override them
    // ════════════════════════════════════════════════════════════════════════════

    /**
     * Get the uncertain database being mined.
     * @return The database (never null after construction)
     */
    protected final UncertainDatabase getDatabase() {
        return database;
    }

    /**
     * Get the probability threshold (tau).
     * @return Tau value in (0, 1]
     */
    protected final double getTau() {
        return tau;
    }

    /**
     * Get the number of top patterns to find.
     * @return K value (always >= 1)
     */
    protected final int getK() {
        return k;
    }

    /**
     * Get the support calculator being used.
     * @return The calculator (may be null if using 3-parameter constructor)
     */
    protected final SupportCalculator getCalculator() {
        return calculator;
    }

    /**
     * Get the vocabulary.
     * @return The vocabulary (never null after construction)
     */
    protected final Vocabulary getVocabulary() {
        return vocab;
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
        executePhase3(frequent1Itemsets);

        long phase3Time = (System.nanoTime() - start3) / 1_000_000;

        // ═══════════════════════════════════════════════════════════════
        // RETURN: Get final top-K results
        // ═══════════════════════════════════════════════════════════════
        return getTopKResults();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SHARED CONCRETE METHODS - Implemented in base class (moved from subclasses)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Phase 1: Computes support and probability for all single-item patterns.
     *
     * <p><b>Algorithm Steps:</b></p>
     * <ol>
     *   <li>Create singleton itemsets for all items in the vocabulary</li>
     *   <li>For each item in parallel:
     *     <ul>
     *       <li>Get its tidset (transaction IDs and probabilities)</li>
     *       <li>Compute expected support and probability using the GF calculator</li>
     *       <li>Create Pattern for result and PatternInfo for caching</li>
     *     </ul>
     *   </li>
     *   <li>Sort results by support (descending) then probability (descending)</li>
     *   <li>Store PatternInfo objects in cache for Phase 2 and 3</li>
     * </ol>
     *
     * <p><b>Why Parallel Processing?</b> Single-item support calculations are independent,
     * so we can leverage multiple CPU cores to speed up this phase significantly.</p>
     *
     * <p><b>FrequentItemset vs CachedFrequentItemset:</b></p>
     * <ul>
     *   <li>FrequentItemset: For final results (no tidset to save memory)</li>
     *   <li>CachedFrequentItemset: Internal cache with tidset for efficient intersection in later phases</li>
     * </ul>
     *
     * @return List of all single-item patterns sorted by support (descending)
     */
    protected List<FrequentItemset> computeAllSingletonSupports() {
        int vocabSize = getVocabulary().size();

        // Array to store FrequentItemset results from parallel computation
        FrequentItemset[] resultArray = new FrequentItemset[vocabSize];

        // Thread-safe cache for concurrent writes during parallel processing
        ConcurrentHashMap<Itemset, CachedFrequentItemset> concurrentCache = new ConcurrentHashMap<>(vocabSize);

        // Pre-create all singleton itemsets to avoid repeated object creation
        this.singletonCache = new Itemset[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            singletonCache[i] = createSingletonItemset(i);
        }

        // Cache database and calculator references for lambda (avoid repeated getter calls)
        final UncertainDatabase db = getDatabase();
        final SupportCalculator calc = getCalculator();

        // Process each item in parallel for maximum performance
        java.util.stream.IntStream.range(0, vocabSize).parallel().forEach(item -> {
            // Get the pre-created singleton itemset for this item
            Itemset singleton = singletonCache[item];

            // Get tidset: list of (transaction_id, probability) pairs where this item appears
            Tidset tidset = db.getTidset(singleton);

            // Skip items that don't appear in any transaction
            if (tidset.isEmpty()) {
                resultArray[item] = null;
                return;
            }

            // Compute expected support and probability using Generating Function approach
            // supportResult[0] = expected support (number of transactions)
            // supportResult[1] = probability of appearing in at least one transaction
            double[] supportResult = calc.computeProbabilisticSupportFromTidset(tidset, db.size());

            int support = (int) supportResult[0];
            double probability = supportResult[1];

            /**
             * We create both FrequentItemset and CachedFrequentItemset here:
             *
             * - FrequentItemset is used for the final output of Phase 1 (doesn't store tidset)
             * - CachedFrequentItemset is cached for later phases (includes tidset for efficient intersections)
             *
             * While this creates some duplication, it allows us to:
             * 1. Process everything in parallel (both object creation)
             * 2. Keep final results memory-efficient (FrequentItemset without tidset)
             * 3. Enable fast intersection operations in Phase 3 (CachedFrequentItemset with tidset)
             */
            FrequentItemset fi = new FrequentItemset(singleton, support, probability);

            resultArray[item] = fi;
            concurrentCache.put(singleton, new CachedFrequentItemset(singleton, support, probability, tidset));
        });

        /**
         * Sort the results by:
         * 1. Support (descending) - higher support patterns are more important
         * 2. Probability (descending) - break ties using probability
         *
         * This ordering is crucial for Phase 2 efficiency, as it allows early termination
         * when the Top-K heap is full.
         */
        List<FrequentItemset> result = Arrays.stream(resultArray)
                .filter(Objects::nonNull)
                .sorted(FrequentItemset::compareBySupport)
                .collect(Collectors.toList());

        // Transfer concurrent cache to regular cache for Phase 2 and 3
        this.cache = concurrentCache;

        return result;
    }

    // ==================== Utility Methods ====================

    /**
     * Creates a singleton itemset containing only the specified item.
     *
     * @param item The item ID to include
     * @return A new Itemset containing only the specified item
     */
    protected Itemset createSingletonItemset(int item) {
        Itemset itemset = new Itemset(getVocabulary());
        itemset.add(item);
        return itemset;
    }

    /**
     * Retrieves the current minimum support threshold from the Top-K heap.
     *
     * <p>Returns 0 when the heap is not full (accepting any support).
     * Returns the minimum support in the heap when full (only better patterns can enter).</p>
     *
     * @return The current support threshold
     */
    protected int getThreshold() {
        return topK.getMinSupport();
    }

    /**
     * Gets the support value for a single item.
     *
     * <p>Looks up the item's PatternInfo from the cache (computed in Phase 1).</p>
     *
     * @param item The item ID
     * @return The support of the item, or 0 if not found
     */
    protected int getItemSupport(int item) {
        if (item < 0 || item >= singletonCache.length) return 0;
        Itemset singleton = singletonCache[item];
        CachedFrequentItemset cached = cache.get(singleton);
        return (cached != null) ? cached.getSupport() : 0;
    }

    /**
     * Gets the maximum item ID in an itemset.
     *
     * <p>Used to enforce canonical order: extensions must add items with ID
     * greater than this maximum.</p>
     *
     * @param itemset The itemset to examine
     * @return The maximum item ID, or -1 if empty
     */
    protected int getMaxItemIndex(Itemset itemset) {
        // Use primitive array to avoid boxing overhead
        int[] items = itemset.getItemsArray();
        if (items.length == 0) return -1;

        // Items are stored in sorted order, so last item is maximum
        return items[items.length - 1];
    }

    // ==================== Closure Checking Methods ====================

    /**
     * Checks if a 1-itemset is closed.
     *
     * <p><b>Closure Definition:</b> A 1-itemset {A} is closed if for all other items B,
     * support(A ∪ B) < support(A). This means no immediate superset has the same support.</p>
     *
     * <p><b>Algorithm:</b></p>
     * <ol>
     *   <li>For each other item B in support-descending order:
     *     <ul>
     *       <li>Skip if B is the same as A</li>
     *       <li>Stop if B's support < A's support (remaining items can't violate closure)</li>
     *       <li>Compute or retrieve support of A ∪ B</li>
     *       <li>If support(A ∪ B) == support(A), A is not closed</li>
     *       <li>Cache the 2-itemset if B's support >= minsup (for Phase 3)</li>
     *     </ul>
     *   </li>
     * </ol>
     *
     * @param oneItemFI The 1-itemset to check
     * @param supOneItem The support of the 1-itemset
     * @param frequent1Itemset The list of all 1-itemsets in support-descending order
     * @param minsup The current minimum support threshold
     * @return True if the 1-itemset is closed, false otherwise
     */
    protected boolean checkClosure1Itemset(FrequentItemset oneItemFI, int supOneItem,
                                         List<FrequentItemset> frequent1Itemset, int minsup) {
        // Extract the item ID from the singleton itemset
        int itemA = oneItemFI.getItems().get(0);

        // Check closure against all other items
        for (FrequentItemset otherFI : frequent1Itemset) {
            int itemB = otherFI.getItems().get(0);

            // Skip self-comparison
            if (itemA == itemB) continue;

            /**
             * Early termination: If other item's support < current item's support,
             * all remaining items also have lower support (due to sorted order).
             *
             * Items with lower support cannot violate closure property because:
             * support(A ∪ B) <= min(support(A), support(B)) < support(A)
             */
            if (otherFI.getSupport() < supOneItem) break;

            // Create the union itemset A ∪ B
            Itemset unionItemset = oneItemFI.union(otherFI);

            // Try to retrieve from cache first
            CachedFrequentItemset cached = cache.get(unionItemset);
            int supAB;
            double probAB;
            Tidset tidsetAB;

            if (cached != null) {
                // Cache hit - reuse previously computed values
                supAB = cached.getSupport();
                probAB = cached.getProbability();
                tidsetAB = cached.getTidset();
            } else {
                // Cache miss - compute support for A ∪ B

                // Intersect tidsets: transactions containing both A and B
                tidsetAB = cache.get(oneItemFI).getTidset().intersect(cache.get(otherFI).getTidset());

                if (!tidsetAB.isEmpty()) {
                    // Compute support using the Generating Function calculator
                    double[] result = getCalculator().computeProbabilisticSupportFromTidset(tidsetAB, getDatabase().size());
                    supAB = (int) result[0];
                    probAB = result[1];
                } else {
                    // Empty tidset means no common transactions
                    supAB = 0;
                    probAB = 0.0;
                }

                /**
                 * Cache the 2-itemset if the other item meets minsup.
                 *
                 * We only cache itemsets that might be useful in Phase 3.
                 * If otherItem's support < minsup, it won't be used for extensions.
                 */
                if (otherFI.getSupport() >= minsup) {
                    cache.put(unionItemset, new CachedFrequentItemset(unionItemset, supAB, probAB, tidsetAB));
                }
            }

            /**
             * Closure violation check:
             * If support(A ∪ B) == support(A), then A is not closed because
             * there exists a superset with the same support.
             */
            if (supAB == supOneItem) {
                return false;  // Not closed
            }
        }

        // No closure violation found
        return true;
    }

    /**
     * Checks closure and generates extensions for a candidate pattern.
     *
     * <p>This is the core method of Phase 3, implementing multiple optimizations:</p>
     *
     * <p><b>Algorithm:</b></p>
     * <ol>
     *   <li>For each frequent item i (in support-descending order):
     *     <ul>
     *       <li>Skip if i is already in candidate X</li>
     *       <li>Apply pruning strategies P3-P7 to avoid unnecessary computations</li>
     *       <li>Compute support of X ∪ {i} if needed for closure check or extension</li>
     *       <li>Check closure: if support(X ∪ {i}) == support(X), mark X as not closed</li>
     *       <li>Generate extension X ∪ {i} if i > max(X) (canonical order)</li>
     *     </ul>
     *   </li>
     * </ol>
     *
     * @param candidate The candidate pattern to check
     * @param threshold The current minimum support threshold (cached from caller to avoid synchronized access)
     * @return ClosureCheckResult containing closure status and valid extensions
     */
    protected ClosureCheckResult checkClosureAndGenerateExtensions(FrequentItemset candidate, int threshold) {
        // candidate IS-A Itemset, so we can use it directly as X
        int supX = candidate.getSupport();
        boolean isClosed = true;  // Assume closed until proven otherwise

        List<FrequentItemset> extensions = new ArrayList<>();

        // Get maximum item in X for canonical order enforcement
        int maxItemInX = getMaxItemIndex(candidate);

        /**
         * Closure checking optimization flag.
         *
         * Once we encounter an item with support < supX, we know all remaining items
         * (due to sorted order) also have lower support, so they cannot violate closure.
         * This allows us to skip closure checks for remaining items.
         */
        boolean closureCheckingDone = false;

        // Iterate through all frequent items in support-descending order
        for (int idx = 0; idx < frequentItemCount; idx++) {
            int item = frequentItems[idx];

            // Skip if item already in candidate itemset
            if (candidate.contains(item)) continue;

            int itemSupport = getItemSupport(item);

            /**
             * Pruning Strategy 3: Item Support Threshold Pruning
             *
             * If item's support < threshold, skip it and all remaining items because:
             * 1. support(X ∪ {item}) <= min(supX, supItem) < threshold
             * 2. Extension cannot enter Top-K
             * 3. All remaining items have even lower support (sorted order)
             */
            if (itemSupport < threshold) {
                break;  // All remaining items also fail this test
            }

            /**
             * Update closureCheckingDone flag.
             *
             * Once item support < supX, all remaining items also have lower support.
             * By anti-monotonicity: support(X ∪ {item}) <= supItem < supX
             * Therefore, remaining items cannot violate closure (supXe < supX guaranteed)
             */
            if (!closureCheckingDone && itemSupport < supX) {
                closureCheckingDone = true;
            }

            /**
             * Determine what operations we need to perform:
             *
             * - needClosureCheck: Only if item support >= supX and we haven't found violation yet
             * - needExtension: Only if item > maxItemInX (canonical order)
             */
            boolean needClosureCheck = !closureCheckingDone && isClosed;
            boolean needExtension = (item > maxItemInX);

            /**
             * Upper bound calculation for support(X ∪ {item}).
             *
             * Basic upper bound: min(supX, supItem)
             * This comes from anti-monotonicity property.
             */
            int upperBound = Math.min(supX, itemSupport);
            int standardUpperBound = upperBound;

            /**
             * Pruning Strategy 4: Subset-Based Upper Bound Tightening
             *
             * We can tighten the upper bound using 2-itemset supports:
             * For each item y in X:
             *   support(X ∪ {item}) <= support({y, item})
             *
             * Take minimum over all such 2-itemsets to get tighter bound.
             *
             * Only applies when generating extensions and Top-K is full.
             */
            if (topK.isFull() && needExtension) {
                // Use primitive array to avoid Integer boxing/unboxing overhead
                for (int existingItem : candidate.getItemsArray()) {
                    // Create 2-itemset {existingItem, item} using factory method (optimized)
                    Itemset twoItemset = Itemset.of(vocab,
                        Math.min(existingItem, item),
                        Math.max(existingItem, item));

                    // Look up cached support of this 2-itemset
                    CachedFrequentItemset cachedSubset = cache.get(twoItemset);
                    if (cachedSubset != null) {
                        // Tighten upper bound
                        upperBound = Math.min(upperBound, cachedSubset.getSupport());

                        // Early exit if upper bound already too low
                        if (upperBound < threshold) {
                            break;
                        }
                    }
                }
            }

            /**
             * Pruning Strategy 5: Upper Bound Filtering
             *
             * If upper bound < threshold, the extension cannot enter Top-K.
             * Skip computing actual support if we know it won't be good enough.
             */
            boolean canEnterTopK = (upperBound >= threshold);
            boolean shouldGenerateExtension = needExtension && canEnterTopK;

            /**
             * Optimization: Skip if neither closure check nor extension needed.
             *
             * This happens when:
             * 1. Closure checking is done (item support < supX)
             * 2. Extension not needed (item <= maxItemInX) or filtered out
             */
            if (!needClosureCheck && !shouldGenerateExtension) {
                continue;
            }

            // At this point, we need to compute support of candidate ∪ {item}

            Itemset itemItemset = singletonCache[item];
            Itemset Xe = candidate.union(itemItemset);  // Extension: candidate ∪ {item}
            int supXe;
            double probXe;
            Tidset tidsetXe;

            // Try to retrieve from cache first
            CachedFrequentItemset cached = cache.get(Xe);
            if (cached != null) {
                // Cache hit - reuse values
                supXe = cached.getSupport();
                probXe = cached.getProbability();
                tidsetXe = cached.getTidset();
            } else {
                // Cache miss - need to compute support

                // Get cached tidsets for candidate and item
                CachedFrequentItemset xInfo = cache.get(candidate);
                CachedFrequentItemset itemInfo = cache.get(itemItemset);

                // Compute intersection of tidsets
                if (xInfo == null || itemInfo == null) {
                    // Fallback: retrieve from database (should rarely happen)
                    Tidset tidsetX = getDatabase().getTidset(candidate);
                    Tidset tidsetItem = getDatabase().getTidset(itemItemset);
                    tidsetXe = tidsetX.intersect(tidsetItem);
                } else {
                    // Normal case: intersect cached tidsets
                    tidsetXe = xInfo.getTidset().intersect(itemInfo.getTidset());
                }

                int tidsetSize = tidsetXe.size();

                /**
                 * Pruning Strategy 6: Tidset Size Pruning
                 *
                 * The tidset size provides a very cheap upper bound on support:
                 * support(Xe) <= tidsetSize
                 *
                 * If tidsetSize < threshold and we don't need closure check,
                 * we can skip the expensive support calculation.
                 *
                 * Note: We still need to compute support if closure check is needed,
                 * even if tidset size < threshold.
                 */
                if (tidsetSize < threshold && !needClosureCheck) {
                    supXe = 0;
                    probXe = 0.0;
                    // Cache the zero-support result to avoid recomputing
                    cache.put(Xe, new CachedFrequentItemset(Xe, 0, 0.0, tidsetXe));
                    continue;
                }

                /**
                 * Pruning Strategy 7: Tidset-Based Early Closure Detection
                 *
                 * If tidsetSize < supX, then definitely support(Xe) < supX
                 * (since support cannot exceed tidset size).
                 *
                 * This means Xe cannot violate closure. If we also don't need
                 * to generate extension, we can skip support calculation entirely.
                 */
                if (needClosureCheck && tidsetSize < supX) {
                    if (!shouldGenerateExtension) {
                        supXe = 0;
                        probXe = 0.0;
                        // Cache the zero-support result
                        cache.put(Xe, new CachedFrequentItemset(Xe, 0, 0.0, tidsetXe));
                        continue;
                    }
                    // We still need extension, but can skip closure check
                    needClosureCheck = false;
                }

                /**
                 * Compute actual support using the Generating Function calculator.
                 *
                 * This is the most expensive operation in the algorithm.
                 * All the pruning strategies above aim to avoid this computation
                 * when we know the result won't be useful.
                 */
                double[] result = getCalculator().computeProbabilisticSupportFromTidset(
                        tidsetXe, getDatabase().size());
                supXe = (int) result[0];
                probXe = result[1];

                // Cache the computed result for potential reuse
                cache.put(Xe, new CachedFrequentItemset(Xe, supXe, probXe, tidsetXe));
            }

            /**
             * Closure check: If support(Xe) == support(X), then X is not closed
             * because there exists a superset with the same support.
             */
            if (needClosureCheck && supXe == supX) {
                isClosed = false;
            }

            /**
             * Generate extension if needed.
             *
             * Extensions are added to the result list for later exploration.
             * They will be inserted into the priority queue if their support
             * meets the threshold.
             */
            if (shouldGenerateExtension) {
                extensions.add(new FrequentItemset(Xe, supXe, probXe));
            }
        }

        return new ClosureCheckResult(isClosed, extensions);
    }

    // ==================== ABSTRACT METHODS - Subclasses must implement these ====================

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
     * Phase 3: Execute pattern space enumeration to discover closed itemsets.
     *
     * <p>This phase explores the pattern search space using a specific traversal
     * strategy to find the top-K frequent closed itemsets. The search continues
     * until the search frontier is exhausted or early termination conditions are met.</p>
     *
     * <p><b>Phase 3 Objectives:</b></p>
     * <ul>
     *   <li>Enumerate candidate patterns in a strategy-specific order</li>
     *   <li>Check closure for each candidate using inherited closure checking methods</li>
     *   <li>Insert closed patterns into Top-K heap</li>
     *   <li>Generate canonical extensions to grow the search frontier</li>
     *   <li>Apply pruning strategies to avoid unnecessary computation</li>
     * </ul>
     *
     * <p><b>Search Strategy (Subclass-Defined):</b></p>
     * <p>Subclasses determine the traversal order by choosing appropriate data structures:</p>
     * <ul>
     *   <li><b>Best-First:</b> Priority queue (highest support first) - optimal for Top-K</li>
     *   <li><b>Depth-First:</b> Stack (LIFO) - memory-efficient for deep trees</li>
     *   <li><b>Breadth-First:</b> Queue (FIFO) - level-by-level exploration</li>
     *   <li><b>Other:</b> A*, random sampling, beam search, etc.</li>
     * </ul>
     *
     * <p><b>Generic Algorithm Pattern:</b></p>
     * <pre>
     * while (searchFrontier is not empty) {
     *     candidate = getNextCandidate()           // Strategy-specific ordering
     *
     *     if (canTerminate(candidate, threshold))  // Optional early termination
     *         break
     *
     *     result = checkClosureAndGenerateExtensions(candidate, threshold)
     *
     *     if (result.isClosed())
     *         topK.insert(candidate)               // Update Top-K heap
     *
     *     addToFrontier(result.getExtensions())    // Strategy-specific insertion
     * }
     * </pre>
     *
     * <p><b>Subclass Responsibilities:</b></p>
     * <ul>
     *   <li>Choose and initialize search data structure (queue, stack, priority queue, etc.)</li>
     *   <li>Determine candidate processing order (best-first, DFS, BFS, etc.)</li>
     *   <li>Implement main enumeration loop</li>
     *   <li>Call inherited closure checking methods (checkClosureAndGenerateExtensions)</li>
     *   <li>Insert closed patterns into Top-K heap</li>
     *   <li>Manage search frontier (add extensions back to data structure)</li>
     *   <li>Optionally implement early termination for efficiency</li>
     * </ul>
     *
     * @param singletonPatterns singleton patterns from Phase 1 (may be used for initialization)
     */
    protected abstract void executePhase3(List<FrequentItemset> singletonPatterns);

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