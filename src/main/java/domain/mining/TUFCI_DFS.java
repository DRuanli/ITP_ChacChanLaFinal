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
 * TUFCI_DFS: Top-K Uncertain Frequent Closed Itemset Mining using Depth-First Search
 *
 * <p>This class implements the TUFCI algorithm using DFS traversal strategy instead of
 * Best-First Search. Created for comparison purposes to analyze the performance difference
 * between DFS and Best-First Search strategies.</p>
 *
 * <h2>Algorithm Comparison:</h2>
 * <table border="1">
 *   <tr><th>Aspect</th><th>Best-First (TUFCI)</th><th>DFS (TUFCI_DFS)</th></tr>
 *   <tr><td>Data Structure</td><td>PriorityQueue</td><td>Stack (Deque)</td></tr>
 *   <tr><td>Selection Order</td><td>Highest support first</td><td>Most recent (LIFO)</td></tr>
 *   <tr><td>Traversal</td><td>Greedy by support</td><td>Depth-first exploration</td></tr>
 *   <tr><td>Threshold Rise</td><td>Fast (high support first)</td><td>Slower (depends on path)</td></tr>
 *   <tr><td>Early Termination</td><td>Strong (when best < threshold)</td><td>Weak (must explore all)</td></tr>
 *   <tr><td>Memory</td><td>O(candidates)</td><td>O(max depth × branching)</td></tr>
 * </table>
 *
 * <h2>DFS Traversal Pattern:</h2>
 * <pre>
 *                    Root
 *                   /    \
 *                  A      B
 *                 / \    / \
 *               A,C A,D B,C B,D
 *
 * DFS Order:  A → A,C → A,C,D → backtrack → A,D → backtrack → B → B,C → ...
 * BFS Order:  A → B → A,C → A,D → B,C → B,D → ...
 * Best-First: Ordered by support (e.g., A:100 → B:80 → A,C:70 → ...)
 * </pre>
 *
 * <h2>Expected Performance Difference:</h2>
 * <ul>
 *   <li>DFS may explore more candidates before finding good patterns</li>
 *   <li>DFS threshold rises slower → less effective pruning</li>
 *   <li>DFS memory usage can be lower for deep, narrow search trees</li>
 *   <li>DFS cannot use early termination as effectively as Best-First</li>
 * </ul>
 *
 * @author Dang Nguyen Le
 * @version 1.0
 * @see TUFCI The Best-First Search version for comparison
 */
public class TUFCI_DFS extends AbstractFrequentItemsetMiner {

    // ==================== Instance Variables ====================

    /**
     * Vocabulary containing all unique items in the database.
     */
    private final Vocabulary vocab;

    /**
     * Top-K heap that maintains the k best patterns found so far.
     */
    private TopKHeap topK;

    /**
     * Stack for DFS traversal (LIFO - Last In, First Out).
     *
     * KEY DIFFERENCE FROM TUFCI:
     * - TUFCI uses PriorityQueue (orders by support)
     * - TUFCI_DFS uses ArrayDeque as Stack (LIFO order)
     *
     * DFS processes the most recently added candidate first,
     * exploring deeply into one branch before backtracking.
     */
    private Deque<FrequentItemset> stack;

    /**
     * Cache storing computed patterns to avoid redundant calculations.
     */
    private Map<Itemset, CachedFrequentItemset> cache;

    /**
     * Support calculator for computing expected support.
     */
    private SupportCalculator calculator;

    /**
     * Pre-computed singleton itemsets for all items.
     */
    private Itemset[] singletonCache;

    /**
     * Number of frequent single items.
     */
    private int frequentItemCount;

    /**
     * Array of frequent item IDs sorted by support descending.
     */
    private int[] frequentItems;

    // ==================== Statistics for Comparison ====================

    /**
     * Count of candidates explored (for comparison with Best-First).
     */
    private long candidatesExplored = 0;

    /**
     * Count of candidates pruned (for comparison with Best-First).
     */
    private long candidatesPruned = 0;

    // ==================== Constructors ====================

    /**
     * Constructs a new TUFCI_DFS miner.
     *
     * @param database The uncertain database to mine
     * @param tau The probability threshold
     * @param k The number of top patterns to find
     */
    public TUFCI_DFS(UncertainDatabase database, double tau, int k) {
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
    public TUFCI_DFS(UncertainDatabase database, double tau, int k,
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
     * This phase is IDENTICAL to TUFCI - no change needed for DFS.
     * The difference in search strategy only affects Phase 3.
     *
     * @return List of all single-item patterns sorted by support (descending)
     */
    @Override
    protected List<FrequentItemset> computeAllSingletonSupports() {
        int vocabSize = vocab.size();

        FrequentItemset[] resultArray = new FrequentItemset[vocabSize];
        ConcurrentHashMap<Itemset, CachedFrequentItemset> concurrentCache = new ConcurrentHashMap<>(vocabSize);

        // Pre-create all singleton itemsets
        this.singletonCache = new Itemset[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            singletonCache[i] = createSingletonItemset(i);
        }

        // Process each item in parallel
        java.util.stream.IntStream.range(0, vocabSize).parallel().forEach(item -> {
            Itemset singleton = singletonCache[item];
            Tidset tidset = database.getTidset(singleton);

            if (tidset.isEmpty()) {
                resultArray[item] = null;
                return;
            }

            double[] supportResult = calculator.computeProbabilisticSupportFromTidset(tidset, database.size());
            int support = (int) supportResult[0];
            double probability = supportResult[1];

            FrequentItemset fi = new FrequentItemset(singleton, support, probability);
            resultArray[item] = fi;
            concurrentCache.put(singleton, new CachedFrequentItemset(singleton, support, probability, tidset));
        });

        // Sort by support descending
        List<FrequentItemset> result = Arrays.stream(resultArray)
                .filter(Objects::nonNull)
                .sorted(FrequentItemset::compareBySupport)
                .collect(Collectors.toList());

        this.cache = concurrentCache;
        return result;
    }

    // ==================== Phase 2: Initialize Data Structures ====================

    /**
     * Phase 2: Initializes Top-K heap and prepares the STACK for DFS.
     *
     * KEY DIFFERENCE FROM TUFCI:
     * - Uses Stack (Deque) instead of PriorityQueue
     * - Seeds stack with 2-itemsets in REVERSE order so that
     *   higher-support items are processed first within each level
     *
     * @param frequent1Itemsets The sorted list of 1-itemsets from Phase 1
     */
    @Override
    protected void initializeTopKWithClosedSingletons(List<FrequentItemset> frequent1Itemsets) {
        // Initialize Top-K heap
        this.topK = new TopKHeap(k);

        /**
         * Initialize STACK for DFS traversal.
         *
         * KEY DIFFERENCE: ArrayDeque used as STACK (LIFO)
         * - push() adds to front
         * - pop() removes from front
         * - Last pushed = First processed (depth-first)
         */
        this.stack = new ArrayDeque<>();

        int minsup = 0;
        int processedItemCount = 0;

        // Process each 1-itemset (same as TUFCI)
        for (FrequentItemset fi : frequent1Itemsets) {
            int support = fi.getSupport();

            // Early termination pruning
            if (topK.isFull() && support < minsup) {
                break;
            }

            processedItemCount++;

            // Check closure
            boolean isClosed = checkClosure1Itemset(fi, support, frequent1Itemsets, minsup);

            if (isClosed) {
                boolean inserted = topK.insert(fi);
                if (inserted && topK.isFull()) {
                    minsup = topK.getMinSupport();
                }
            }
        }

        // Build frequent items array
        List<Integer> frequentItemIndices = new ArrayList<>();
        for (int i = 0; i < processedItemCount; i++) {
            FrequentItemset fi = frequent1Itemsets.get(i);
            if (fi.getSupport() >= minsup) {
                frequentItemIndices.add(fi.getItems().get(0));
            }
        }

        this.frequentItemCount = frequentItemIndices.size();
        this.frequentItems = new int[frequentItemCount];
        for (int i = 0; i < frequentItemCount; i++) {
            frequentItems[i] = frequentItemIndices.get(i);
        }

        /**
         * Seed the STACK with 2-itemsets.
         *
         * DFS OPTIMIZATION: Add in REVERSE support order so that
         * higher-support 2-itemsets are on TOP of stack and processed first.
         * This helps find good patterns earlier, raising threshold faster.
         */
        List<FrequentItemset> twoItemsets = new ArrayList<>();

        for (Map.Entry<Itemset, CachedFrequentItemset> entry : cache.entrySet()) {
            Itemset itemset = entry.getKey();
            if (itemset.size() == 2) {
                CachedFrequentItemset cached = entry.getValue();
                if (cached.getSupport() >= minsup) {
                    twoItemsets.add(cached.toFrequentItemset());
                }
            }
        }

        // Sort by support ASCENDING so higher support ends up on top of stack
        twoItemsets.sort((a, b) -> Integer.compare(a.getSupport(), b.getSupport()));

        // Push all 2-itemsets onto stack (higher support will be on top)
        for (FrequentItemset fi : twoItemsets) {
            stack.push(fi);
        }
    }

    // ==================== Phase 3: DFS Mining ====================

    /**
     * Phase 3: Performs Depth-First Search to discover closed itemsets.
     *
     * <h2>KEY DIFFERENCE FROM TUFCI (Best-First):</h2>
     * <ul>
     *   <li>Uses stack.pop() instead of pq.poll()</li>
     *   <li>Processes most recently added candidate (LIFO)</li>
     *   <li>Goes deep into one branch before backtracking</li>
     *   <li>Cannot use early termination as effectively</li>
     * </ul>
     *
     * <h2>DFS Algorithm:</h2>
     * <pre>
     * while (stack not empty):
     *     candidate = stack.pop()           // LIFO - most recent first
     *     if (candidate.support < threshold):
     *         continue                      // Skip but DON'T break (unlike Best-First)
     *
     *     check closure and generate extensions
     *     if (closed): insert into Top-K
     *
     *     push extensions onto stack        // Extensions explored BEFORE siblings
     * </pre>
     *
     * <h2>Why No Early Termination in DFS?</h2>
     * <p>In Best-First, when the best candidate fails the threshold, ALL remaining
     * candidates also fail (due to priority queue ordering). In DFS, the stack
     * contains candidates with mixed support values - a low-support candidate
     * might be followed by a high-support one. We must check all.</p>
     *
     * @param frequent1itemsets The list of 1-itemsets (not used directly)
     */
    @Override
    protected void performBestFirstMining(List<FrequentItemset> frequent1itemsets) {
        // Reset statistics
        candidatesExplored = 0;
        candidatesPruned = 0;

        /**
         * DFS MAIN LOOP
         *
         * Key differences from Best-First:
         * 1. stack.pop() vs pq.poll() - LIFO vs priority ordering
         * 2. continue vs break - cannot early terminate entire search
         * 3. Extensions pushed to stack - explored before siblings
         */
        while (!stack.isEmpty()) {
            // Pop from stack (LIFO - Last In, First Out)
            FrequentItemset candidate = stack.pop();
            candidatesExplored++;

            int threshold = getThreshold();

            /**
             * PRUNING: Skip candidates below threshold
             *
             * IMPORTANT DIFFERENCE FROM BEST-FIRST:
             * - Best-First can BREAK here (all remaining are worse)
             * - DFS must CONTINUE (stack has mixed support values)
             *
             * This is a major reason why DFS is less efficient for Top-K mining.
             */
            if (candidate.getSupport() < threshold) {
                candidatesPruned++;
                continue;  // Skip this candidate, but check remaining
            }

            // Check closure and generate extensions
            ClosureCheckResult result = checkClosureAndGenerateExtensions(candidate, threshold);

            // Insert closed patterns into Top-K
            if (result.isClosed()) {
                topK.insert(candidate);
            }

            /**
             * PUSH EXTENSIONS ONTO STACK
             *
             * DFS behavior: Extensions are pushed and will be processed
             * BEFORE any siblings of the current candidate.
             *
             * This creates the "depth-first" behavior:
             *   Process {A} → push {A,B}, {A,C}
             *   Process {A,C} (top of stack) → push {A,C,D}
             *   Process {A,C,D} → ...
             *   Eventually backtrack to {A,B}
             *
             * Optimization: Push in reverse support order so higher-support
             * extensions are on top and processed first.
             */
            int newThreshold = getThreshold();
            List<FrequentItemset> validExtensions = new ArrayList<>();

            for (FrequentItemset ext : result.getExtensions()) {
                if (ext.getSupport() >= newThreshold) {
                    validExtensions.add(ext);
                } else {
                    candidatesPruned++;
                }
            }

            // Sort by support ASCENDING so higher support ends up on top
            validExtensions.sort((a, b) -> Integer.compare(a.getSupport(), b.getSupport()));

            // Push all extensions (higher support will be processed first)
            for (FrequentItemset ext : validExtensions) {
                stack.push(ext);
            }
        }
    }

    // ==================== Result Retrieval ====================

    /**
     * Retrieves the final top-k patterns from the heap.
     *
     * @return List of top-k patterns sorted by support then probability
     */
    @Override
    protected List<FrequentItemset> getTopKResults() {
        List<FrequentItemset> results = topK.getAll();
        results.sort(FrequentItemset::compareBySupport);
        return results;
    }

    // ==================== Statistics Methods ====================

    /**
     * Get the number of candidates explored during mining.
     * Useful for comparing with Best-First Search.
     *
     * @return number of candidates popped from stack
     */
    public long getCandidatesExplored() {
        return candidatesExplored;
    }

    /**
     * Get the number of candidates pruned during mining.
     * Useful for comparing with Best-First Search.
     *
     * @return number of candidates skipped due to threshold
     */
    public long getCandidatesPruned() {
        return candidatesPruned;
    }

    // ==================== Utility Methods ====================

    private Itemset createSingletonItemset(int item) {
        Itemset itemset = new Itemset(vocab);
        itemset.add(item);
        return itemset;
    }

    private int getThreshold() {
        return topK.getMinSupport();
    }

    private int getItemSupport(int item) {
        if (item < 0 || item >= singletonCache.length) return 0;
        Itemset singleton = singletonCache[item];
        CachedFrequentItemset cached = cache.get(singleton);
        return (cached != null) ? cached.getSupport() : 0;
    }

    private int getMaxItemIndex(Itemset itemset) {
        int[] items = itemset.getItemsArray();
        if (items.length == 0) return -1;
        return items[items.length - 1];
    }

    // ==================== Closure Checking Methods ====================

    /**
     * Checks if a 1-itemset is closed.
     * (Identical to TUFCI implementation)
     */
    private boolean checkClosure1Itemset(FrequentItemset oneItemFI, int supOneItem,
                                         List<FrequentItemset> frequent1Itemset, int minsup) {
        int itemA = oneItemFI.getItems().get(0);

        for (FrequentItemset otherFI : frequent1Itemset) {
            int itemB = otherFI.getItems().get(0);

            if (itemA == itemB) continue;

            if (otherFI.getSupport() < supOneItem) break;

            Itemset unionItemset = oneItemFI.union(otherFI);

            CachedFrequentItemset cached = cache.get(unionItemset);
            int supAB;
            double probAB;
            Tidset tidsetAB;

            if (cached != null) {
                supAB = cached.getSupport();
                probAB = cached.getProbability();
                tidsetAB = cached.getTidset();
            } else {
                tidsetAB = cache.get(oneItemFI).getTidset().intersect(cache.get(otherFI).getTidset());

                if (!tidsetAB.isEmpty()) {
                    double[] result = calculator.computeProbabilisticSupportFromTidset(tidsetAB, database.size());
                    supAB = (int) result[0];
                    probAB = result[1];
                } else {
                    supAB = 0;
                    probAB = 0.0;
                }

                if (otherFI.getSupport() >= minsup) {
                    cache.put(unionItemset, new CachedFrequentItemset(unionItemset, supAB, probAB, tidsetAB));
                }
            }

            if (supAB == supOneItem) {
                return false;
            }
        }

        return true;
    }

    /**
     * Checks closure and generates extensions for a candidate pattern.
     * (Identical to TUFCI implementation - pruning strategies still apply)
     */
    private ClosureCheckResult checkClosureAndGenerateExtensions(FrequentItemset candidate, int threshold) {
        int supX = candidate.getSupport();
        boolean isClosed = true;

        List<FrequentItemset> extensions = new ArrayList<>();

        int maxItemInX = getMaxItemIndex(candidate);
        boolean closureCheckingDone = false;

        for (int idx = 0; idx < frequentItemCount; idx++) {
            int item = frequentItems[idx];

            if (candidate.contains(item)) continue;

            int itemSupport = getItemSupport(item);

            // Pruning: Item support threshold
            if (itemSupport < threshold) {
                break;
            }

            if (!closureCheckingDone && itemSupport < supX) {
                closureCheckingDone = true;
            }

            boolean needClosureCheck = !closureCheckingDone && isClosed;
            boolean needExtension = (item > maxItemInX);

            int upperBound = Math.min(supX, itemSupport);

            // Pruning: Subset-based upper bound
            if (topK.isFull() && needExtension) {
                for (int existingItem : candidate.getItemsArray()) {
                    Itemset twoItemset = Itemset.of(vocab,
                        Math.min(existingItem, item),
                        Math.max(existingItem, item));

                    CachedFrequentItemset cachedSubset = cache.get(twoItemset);
                    if (cachedSubset != null) {
                        upperBound = Math.min(upperBound, cachedSubset.getSupport());
                        if (upperBound < threshold) {
                            break;
                        }
                    }
                }
            }

            boolean canEnterTopK = (upperBound >= threshold);
            boolean shouldGenerateExtension = needExtension && canEnterTopK;

            if (!needClosureCheck && !shouldGenerateExtension) {
                continue;
            }

            Itemset itemItemset = singletonCache[item];
            Itemset Xe = candidate.union(itemItemset);
            int supXe;
            double probXe;
            Tidset tidsetXe;

            CachedFrequentItemset cached = cache.get(Xe);
            if (cached != null) {
                supXe = cached.getSupport();
                probXe = cached.getProbability();
                tidsetXe = cached.getTidset();
            } else {
                CachedFrequentItemset xInfo = cache.get(candidate);
                CachedFrequentItemset itemInfo = cache.get(itemItemset);

                if (xInfo == null || itemInfo == null) {
                    Tidset tidsetX = database.getTidset(candidate);
                    Tidset tidsetItem = database.getTidset(itemItemset);
                    tidsetXe = tidsetX.intersect(tidsetItem);
                } else {
                    tidsetXe = xInfo.getTidset().intersect(itemInfo.getTidset());
                }

                int tidsetSize = tidsetXe.size();

                // Pruning: Tidset size
                if (tidsetSize < threshold && !needClosureCheck) {
                    supXe = 0;
                    probXe = 0.0;
                    cache.put(Xe, new CachedFrequentItemset(Xe, 0, 0.0, tidsetXe));
                    continue;
                }

                // Pruning: Tidset-based early closure
                if (needClosureCheck && tidsetSize < supX) {
                    if (!shouldGenerateExtension) {
                        supXe = 0;
                        probXe = 0.0;
                        cache.put(Xe, new CachedFrequentItemset(Xe, 0, 0.0, tidsetXe));
                        continue;
                    }
                    needClosureCheck = false;
                }

                double[] result = calculator.computeProbabilisticSupportFromTidset(
                        tidsetXe, database.size());
                supXe = (int) result[0];
                probXe = result[1];

                cache.put(Xe, new CachedFrequentItemset(Xe, supXe, probXe, tidsetXe));
            }

            // Closure check
            if (needClosureCheck && supXe == supX) {
                isClosed = false;
            }

            // Generate extension
            if (shouldGenerateExtension) {
                extensions.add(new FrequentItemset(Xe, supXe, probXe));
            }
        }

        return new ClosureCheckResult(isClosed, extensions);
    }
}