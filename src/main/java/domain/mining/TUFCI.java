package domain.mining;

import domain.support.SupportCalculator;
import infrastructure.persistence.UncertainDatabase;
import infrastructure.persistence.Vocabulary;
import domain.model.*;
import domain.support.DirectConvolutionSupportCalculator;
import infrastructure.topK.TopKHeap;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * TUFCI: Top-K Uncertain Frequent Closed Itemset Mining Algorithm
 *
 * <p>This class implements an efficient algorithm for mining the top-k frequent closed itemsets
 * from uncertain databases. An uncertain database contains transactions where each item has an
 * existential probability, making support calculation probabilistic rather than deterministic.</p>
 *
 * <p><b>Key Concepts:</b></p>
 * <ul>
 *   <li><b>Itemset:</b> A set of items (e.g., {bread, milk})</li>
 *   <li><b>Support:</b> The expected number of transactions containing an itemset</li>
 *   <li><b>Probability:</b> The likelihood that an itemset appears in at least one transaction</li>
 *   <li><b>Closed Itemset:</b> An itemset where no superset has the same support</li>
 *   <li><b>Top-K:</b> The k itemsets with highest support values</li>
 * </ul>
 *
 * <p><b>Algorithm Overview - Three Phases:</b></p>
 * <ol>
 *   <li><b>Phase 1 (computeAllSingletonSupports):</b> Compute probabilistic support for ALL singleton patterns (no filtering)</li>
 *   <li><b>Phase 2 (initializeTopKWithClosedSingletons):</b> Check closure for singletons, populate Top-K heap, seed priority queue</li>
 *   <li><b>Phase 3 (performBestFirstMining):</b> Best-first search using priority queue to discover larger closed patterns</li>
 * </ol>
 *
 * <p><b>Pruning Strategies:</b> The algorithm uses multiple pruning techniques to improve efficiency:
 * early termination, upper bound filtering, subset-based pruning, and tidset-based pruning.</p>
 *
 * @author Your Name
 * @version 1.0
 */
public class TUFCI extends AbstractFrequentItemsetMiner {

    // ==================== Instance Variables ====================

    /**
     * Vocabulary containing all unique items in the database.
     * Used to create itemsets and manage item identifiers.
     */
    private final Vocabulary vocab;

    /**
     * Top-K heap that maintains the k best patterns found so far.
     * Automatically updates the minimum support threshold as better patterns are discovered.
     */
    private TopKHeap topK;

    /**
     * Priority queue for canonical mining in Phase 3.
     * Orders candidates by support (descending), then by itemset size (ascending),
     * then by probability (descending). This ordering ensures we explore the most
     * promising candidates first.
     */
    private PriorityQueue<FrequentItemset> pq;

    /**
     * Cache storing computed patterns to avoid redundant calculations.
     * Maps itemsets to their CachedFrequentItemset (support, probability, tidset).
     * This is crucial for performance as it enables O(1) lookup for previously computed patterns.
     */
    private Map<Itemset, CachedFrequentItemset> cache;

    /**
     * Support calculator for computing expected support from probability distributions.
     * Uses the Generating Function (GF) approach for efficient probabilistic support calculation.
     */
    private SupportCalculator calculator;

    /**
     * Pre-computed singleton itemsets for all items in the vocabulary.
     * Avoids creating new Itemset objects repeatedly during mining.
     * Index corresponds to item ID in the vocabulary.
     */
    private Itemset[] singletonCache;

    /**
     * Number of frequent single items that meet the minimum support threshold.
     * Used to limit the search space in Phase 3.
     */
    private int frequentItemCount;

    /**
     * Array of item IDs that are frequent (meet minimum support).
     * Sorted by support in descending order for efficient pruning.
     */
    private int[] frequentItems;

    // ==================== Constructors ====================

    /**
     * Constructs a new TUFCI miner.
     *
     * @param database The uncertain database to mine
     * @param tau The probability threshold (not used in top-k mining but kept for compatibility)
     * @param k The number of top patterns to find
     */
    public TUFCI(UncertainDatabase database, double tau, int k) {
        super(database, tau, k);
        this.vocab = database.getVocabulary();
        this.calculator = new DirectConvolutionSupportCalculator(tau);
        this.cache = new HashMap<>();
    }

    /**
     * Constructor with custom support calculator.
     *
     * @param database uncertain database to mine
     * @param tau probability threshold
     * @param k number of top patterns
     * @param calculator custom support calculation strategy
     */
    public TUFCI(UncertainDatabase database, double tau, int k,
                                  SupportCalculator calculator) {
        super(database, tau, k);
        this.vocab = database.getVocabulary();
        this.calculator = calculator;
        this.cache = new HashMap<>();
    }

    // ==================== Phase 1: Compute Frequent 1-Itemsets ====================

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
    @Override
    protected List<FrequentItemset> computeAllSingletonSupports() {
        int vocabSize = vocab.size();

        // Array to store FrequentItemset results from parallel computation
        FrequentItemset[] resultArray = new FrequentItemset[vocabSize];

        // Thread-safe cache for concurrent writes during parallel processing
        ConcurrentHashMap<Itemset, CachedFrequentItemset> concurrentCache = new ConcurrentHashMap<>(vocabSize);

        // Pre-create all singleton itemsets to avoid repeated object creation
        this.singletonCache = new Itemset[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            singletonCache[i] = createSingletonItemset(i);
        }

        // Process each item in parallel for maximum performance
        java.util.stream.IntStream.range(0, vocabSize).parallel().forEach(item -> {
            // Get the pre-created singleton itemset for this item
            Itemset singleton = singletonCache[item];

            // Get tidset: list of (transaction_id, probability) pairs where this item appears
            Tidset tidset = database.getTidset(singleton);

            // Skip items that don't appear in any transaction
            if (tidset.isEmpty()) {
                resultArray[item] = null;
                return;
            }

            // Compute expected support and probability using Generating Function approach
            // supportResult[0] = expected support (number of transactions)
            // supportResult[1] = probability of appearing in at least one transaction
            double[] supportResult = calculator.computeProbabilisticSupportFromTidset(tidset, database.size());

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

    // ==================== Phase 2: Initialize Data Structures ====================

    /**
     * Phase 2: Initializes Top-K heap, performs closure checking on 1-itemsets,
     * and prepares the priority queue for Phase 3.
     *
     * <p><b>Algorithm Steps:</b></p>
     * <ol>
     *   <li>Initialize Top-K heap to store the best k patterns</li>
     *   <li>Initialize priority queue for canonical mining</li>
     *   <li>For each 1-itemset (in support-descending order):
     *     <ul>
     *       <li>Apply early termination pruning if Top-K is full and support < minsup</li>
     *       <li>Check closure: verify no other single item has same support when combined</li>
     *       <li>If closed, insert into Top-K heap</li>
     *       <li>Update minsup threshold if Top-K becomes full</li>
     *     </ul>
     *   </li>
     *   <li>Build list of frequent items (support >= minsup)</li>
     *   <li>Add all 2-itemsets with support >= minsup to priority queue</li>
     * </ol>
     *
     * <p><b>Closure Check for 1-itemsets:</b></p>
     * <p>A single item A is closed if for all other items B:
     * support(A ∪ B) < support(A). This means no superset has the same support.</p>
     *
     * <p><b>Why Check 2-itemsets?</b> During closure checking, we compute support for
     * all 2-itemsets. We cache these and add promising ones to the priority queue
     * to start Phase 3.</p>
     *
     * @param frequent1Itemsets The sorted list of 1-itemsets from Phase 1
     */
    @Override
    protected void initializeTopKWithClosedSingletons(List<FrequentItemset> frequent1Itemsets) {
        // Initialize Top-K heap to track the k best patterns
        this.topK = new TopKHeap(k);

        /**
         * Initialize priority queue with custom comparator using static method.
         * Ordering: support (DESC) → size (ASC) → probability (DESC)
         */
        this.pq = new PriorityQueue<>(FrequentItemset::compareForPriorityQueue);

        // Minimum support threshold - starts at 0, increases as Top-K fills up
        int minsup = 0;

        // Track how many items we process before early termination
        int processedItemCount = 0;

        // Process each 1-itemset in support-descending order
        for (FrequentItemset fi : frequent1Itemsets) {
            int support = fi.getSupport();
            double probability = fi.getProbability();

            /**
             * Pruning Strategy 1: Phase 1 Early Termination
             *
             * If Top-K is full and current support < minsup, we can stop because:
             * 1. All remaining 1-itemsets have even lower support (sorted order)
             * 2. By anti-monotonicity, their supersets also have lower support
             * 3. Therefore, they cannot enter the Top-K
             */
            if (topK.isFull() && support < minsup) {
                break;
            }

            processedItemCount++;

            // Check if this 1-itemset is closed
            boolean isClosed = checkClosure1Itemset(fi, support, frequent1Itemsets, minsup);

            // If closed, try to insert into Top-K
            if (isClosed) {
                boolean inserted = topK.insert(fi);

                // Update minimum support threshold when Top-K becomes full
                if (inserted && topK.isFull()) {
                    minsup = topK.getMinSupport();
                }
            }
        }

        /**
         * Build the list of frequent items (those meeting minsup threshold).
         * These are the only items we'll consider for extension in Phase 3.
         */
        List<Integer> frequentItemIndices = new ArrayList<>();

        for (int i = 0; i < processedItemCount; i++) {
            FrequentItemset fi = frequent1Itemsets.get(i);

            if (fi.getSupport() >= minsup) {
                // Extract the item ID from the singleton itemset
                frequentItemIndices.add(fi.getItems().get(0));
            }
        }

        // Store frequent items as array for fast iteration in Phase 3
        this.frequentItemCount = frequentItemIndices.size();
        this.frequentItems = new int[frequentItemCount];
        for (int i = 0; i < frequentItemCount; i++) {
            frequentItems[i] = frequentItemIndices.get(i);
        }

        /**
         * Seed the priority queue with 2-itemsets.
         *
         * During closure checking, we already computed support for all 2-itemsets.
         * Add those with support >= minsup to the priority queue to start Phase 3.
         * This bootstraps the canonical mining process.
         */
        for (Map.Entry<Itemset, CachedFrequentItemset> entry : cache.entrySet()) {
            Itemset itemset = entry.getKey();

            if (itemset.size() == 2) {
                CachedFrequentItemset cached = entry.getValue();

                if (cached.getSupport() >= minsup) {
                    // Add FrequentItemset to PQ (tidset not needed in PQ)
                    pq.add(cached.toFrequentItemset());
                }
            }
        }
    }

    // ==================== Phase 3: Recursive Mining ====================

    /**
     * Phase 3: Performs canonical mining to discover all top-k closed itemsets.
     *
     * <p><b>Algorithm Overview:</b></p>
     * <p>Uses a best-first search strategy with a priority queue. For each candidate pattern X:
     * <ol>
     *   <li>Check if X can potentially enter Top-K (threshold pruning)</li>
     *   <li>Check if X is closed (no superset has the same support)</li>
     *   <li>If closed, insert into Top-K heap</li>
     *   <li>Generate extensions X ∪ {item} for all valid items</li>
     *   <li>Add promising extensions to priority queue</li>
     * </ol>
     * </p>
     *
     * <p><b>Canonical Order:</b></p>
     * <p>Extensions are generated in canonical order: only add items with ID greater than
     * the maximum item in X. This ensures each pattern is generated exactly once,
     * avoiding redundant computation.</p>
     *
     * <p><b>Dynamic Threshold:</b></p>
     * <p>As better patterns are found and enter Top-K, the minimum support threshold
     * increases, allowing more aggressive pruning of unpromising candidates.</p>
     *
     * @param frequent1itemsets The list of 1-itemsets (not used in this implementation)
     */
    @Override
    protected void performBestFirstMining(List<FrequentItemset> frequent1itemsets) {
        // Process candidates in priority order until queue is empty
        while (!pq.isEmpty()) {
            // Get the most promising candidate (highest support)
            FrequentItemset candidate = pq.poll();

            int threshold = getThreshold();

            /**
             * Pruning Strategy 2: Main Loop Threshold Pruning
             *
             * If candidate support < current threshold, skip it because:
             * 1. It cannot enter the Top-K (not good enough)
             * 2. All remaining candidates in PQ also have support <= this candidate (PQ ordering)
             * 3. Therefore, we can terminate the entire search
             */
            if (candidate.getSupport() < threshold) {
                break;
            }

            /**
             * Check closure and generate extensions.
             *
             * This method:
             * 1. Verifies if the candidate is closed
             * 2. Generates all valid extensions in canonical order
             * 3. Applies multiple pruning strategies to reduce computation
             *
             * Pass cached threshold to avoid redundant synchronized getThreshold() call
             */
            ClosureCheckResult result = checkClosureAndGenerateExtensions(candidate, threshold);

            // If closed, add to Top-K (which will update threshold if needed)
            if (result.isClosed()) {
                topK.insert(candidate);
            }

            /**
             * Add extensions to priority queue for future exploration.
             *
             * Get updated threshold (may have changed after insertion to Top-K)
             * and only add extensions that meet the threshold.
             */
            int newThreshold = getThreshold();
            for (FrequentItemset ext : result.getExtensions()) {
                if (ext.getSupport() >= newThreshold) {
                    pq.add(ext);
                }
            }
        }
    }

    // ==================== Result Retrieval ====================

    /**
     * Retrieves the final top-k patterns from the heap.
     *
     * <p>Extracts all patterns from the Top-K heap and sorts them by:
     * <ol>
     *   <li>Support (descending) - highest support first</li>
     *   <li>Probability (descending) - break ties with probability</li>
     * </ol>
     * </p>
     *
     * @return List of top-k patterns sorted by support then probability
     */
    @Override
    protected List<FrequentItemset> getTopKResults() {
        // Get all patterns from heap
        List<FrequentItemset> results = topK.getAll();

        // Sort by support DESC, then probability DESC using static comparator
        results.sort(FrequentItemset::compareBySupport);

        return results;
    }

    // ==================== Utility Methods ====================

    /**
     * Creates a singleton itemset containing a single item.
     *
     * @param item The item ID to include
     * @return A new Itemset containing only the specified item
     */
    private Itemset createSingletonItemset(int item) {
        Itemset itemset = new Itemset(vocab);
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
    private int getThreshold() {
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
    private int getItemSupport(int item) {
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
    private int getMaxItemIndex(Itemset itemset) {
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
     * @param oneItem The 1-itemset to check
     * @param supOneItem The support of the 1-itemset
     * @param frequent1Itemset The list of all 1-itemsets in support-descending order
     * @param minsup The current minimum support threshold
     * @return True if the 1-itemset is closed, false otherwise
     */
    private boolean checkClosure1Itemset(FrequentItemset oneItemFI, int supOneItem,
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
                    double[] result = calculator.computeProbabilisticSupportFromTidset(tidsetAB, database.size());
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
     *   <li>For each frequent item not in X:
     *     <ul>
     *       <li>Apply various pruning strategies to avoid unnecessary computation</li>
     *       <li>Determine if we need closure check (item support >= X support)</li>
     *       <li>Determine if we need extension (canonical order)</li>
     *       <li>Compute support of X ∪ {item} if needed</li>
     *       <li>Update closure status if needed</li>
     *       <li>Add to extensions list if promising</li>
     *     </ul>
     *   </li>
     * </ol>
     *
     * <p><b>Pruning Strategies Applied:</b></p>
     * <ul>
     *   <li><b>Item Support Threshold Pruning:</b> Skip items with support < threshold</li>
     *   <li><b>Upper Bound Filtering:</b> Skip if min(supX, supItem) < threshold</li>
     *   <li><b>Subset-Based Upper Bound:</b> Use 2-itemset supports to tighten upper bound</li>
     *   <li><b>Tidset Size Pruning:</b> Skip if tidset size < threshold (before support calculation)</li>
     *   <li><b>Tidset-Based Early Closure:</b> Skip closure check if tidset size < supX</li>
     * </ul>
     *
     * @param candidate The candidate pattern to check
     * @param threshold The current minimum support threshold (cached from caller to avoid synchronized access)
     * @return ClosureCheckResult containing closure status and valid extensions
     */
    private ClosureCheckResult checkClosureAndGenerateExtensions(FrequentItemset candidate, int threshold) {
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
                    Tidset tidsetX = database.getTidset(candidate);
                    Tidset tidsetItem = database.getTidset(itemItemset);
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
                double[] result = calculator.computeProbabilisticSupportFromTidset(
                        tidsetXe, database.size());
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
}
