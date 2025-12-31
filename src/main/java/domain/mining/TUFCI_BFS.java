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
 * TUFCI_BFS: Top-K Uncertain Frequent Closed Itemset Mining using Breadth-First Search
 *
 * <p>This class implements the TUFCI algorithm using BFS traversal strategy instead of
 * Best-First Search. Created for comparison purposes to analyze the performance difference
 * between BFS, DFS, and Best-First Search strategies.</p>
 *
 * <h2>Algorithm Comparison:</h2>
 * <table border="1">
 *   <tr><th>Aspect</th><th>Best-First</th><th>DFS</th><th>BFS</th></tr>
 *   <tr><td>Data Structure</td><td>PriorityQueue</td><td>Stack (LIFO)</td><td>Queue (FIFO)</td></tr>
 *   <tr><td>Selection Order</td><td>Highest support</td><td>Most recent</td><td>Oldest first</td></tr>
 *   <tr><td>Traversal</td><td>Greedy by support</td><td>Depth-first</td><td>Level-by-level</td></tr>
 *   <tr><td>Threshold Rise</td><td>Fast</td><td>Depends on path</td><td>Slow (all levels)</td></tr>
 *   <tr><td>Memory</td><td>O(candidates)</td><td>O(depth × branch)</td><td>O(level width)</td></tr>
 * </table>
 *
 * <h2>BFS Traversal Pattern:</h2>
 * <pre>
 *                    Root
 *                   /    \
 *                  A      B        ← Level 1: Process A, B first
 *                 / \    / \
 *               A,C A,D B,C B,D    ← Level 2: Then all 2-extensions
 *               /
 *            A,C,D                 ← Level 3: Then all 3-extensions
 *
 * BFS Order:   A → B → A,C → A,D → B,C → B,D → A,C,D → ...
 * DFS Order:   A → A,C → A,C,D → backtrack → A,D → backtrack → B → ...
 * Best-First:  Ordered by support (e.g., A:100 → B:80 → A,C:70 → ...)
 * </pre>
 *
 * <h2>BFS Characteristics:</h2>
 * <ul>
 *   <li>Explores ALL patterns at level k before any at level k+1</li>
 *   <li>Guarantees shortest path (smallest itemset) first</li>
 *   <li>Memory can be high for wide search trees</li>
 *   <li>Threshold rises slowly (processes many low-support patterns first)</li>
 * </ul>
 *
 * @author Dang Nguyen Le
 * @version 1.0
 * @see TUFCI The Best-First Search version
 * @see TUFCI_DFS The Depth-First Search version
 */
public class TUFCI_BFS extends AbstractFrequentItemsetMiner {

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
     * Queue for BFS traversal (FIFO - First In, First Out).
     *
     * KEY DIFFERENCE FROM OTHER VERSIONS:
     * - TUFCI uses PriorityQueue (orders by support)
     * - TUFCI_DFS uses ArrayDeque as Stack (LIFO)
     * - TUFCI_BFS uses ArrayDeque as Queue (FIFO)
     *
     * BFS processes candidates in the order they were added,
     * exploring all patterns at one level before moving to the next.
     */
    private Queue<FrequentItemset> queue;

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
     * Count of candidates explored (for comparison).
     */
    private long candidatesExplored = 0;

    /**
     * Count of candidates pruned (for comparison).
     */
    private long candidatesPruned = 0;

    /**
     * Maximum queue size reached (for memory analysis).
     */
    private long maxQueueSize = 0;

    /**
     * Count of candidates at each level (for level analysis).
     */
    private Map<Integer, Long> candidatesPerLevel = new HashMap<>();

    // ==================== Constructors ====================

    /**
     * Constructs a new TUFCI_BFS miner.
     *
     * @param database The uncertain database to mine
     * @param tau The probability threshold
     * @param k The number of top patterns to find
     */
    public TUFCI_BFS(UncertainDatabase database, double tau, int k) {
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
    public TUFCI_BFS(UncertainDatabase database, double tau, int k,
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
     * This phase is IDENTICAL to TUFCI - no change needed for BFS.
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
     * Phase 2: Initializes Top-K heap and prepares the QUEUE for BFS.
     *
     * KEY DIFFERENCE FROM OTHER VERSIONS:
     * - Uses Queue (FIFO) instead of PriorityQueue or Stack
     * - Candidates are processed in the order they were added
     * - All 2-itemsets processed before any 3-itemsets, etc.
     *
     * @param frequent1Itemsets The sorted list of 1-itemsets from Phase 1
     */
    @Override
    protected void initializeTopKWithClosedSingletons(List<FrequentItemset> frequent1Itemsets) {
        // Initialize Top-K heap
        this.topK = new TopKHeap(k);

        /**
         * Initialize QUEUE for BFS traversal.
         *
         * KEY DIFFERENCE: LinkedList used as QUEUE (FIFO)
         * - offer() adds to back
         * - poll() removes from front
         * - First added = First processed (breadth-first)
         */
        this.queue = new LinkedList<>();

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
         * Seed the QUEUE with 2-itemsets.
         *
         * BFS: Add in support-descending order.
         * Since FIFO, the order we add is the order they'll be processed.
         * Higher-support 2-itemsets added first → processed first.
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

        // Sort by support DESCENDING for better threshold rise within level
        twoItemsets.sort(FrequentItemset::compareBySupport);

        // Add all 2-itemsets to queue (will be processed before any 3-itemsets)
        for (FrequentItemset fi : twoItemsets) {
            queue.offer(fi);
        }

        // Track initial queue size
        maxQueueSize = queue.size();
    }

    // ==================== Phase 3: BFS Mining ====================

    /**
     * Phase 3: Performs Breadth-First Search to discover closed itemsets.
     *
     * <h2>KEY DIFFERENCE FROM OTHER VERSIONS:</h2>
     * <ul>
     *   <li>Uses queue.poll() instead of pq.poll() or stack.pop()</li>
     *   <li>Processes candidates in FIFO order</li>
     *   <li>Explores all patterns at level k before any at level k+1</li>
     *   <li>Cannot use early termination (queue has mixed support values)</li>
     * </ul>
     *
     * <h2>BFS Algorithm:</h2>
     * <pre>
     * while (queue not empty):
     *     candidate = queue.poll()           // FIFO - oldest first
     *     if (candidate.support < threshold):
     *         continue                       // Skip but DON'T break
     *
     *     check closure and generate extensions
     *     if (closed): insert into Top-K
     *
     *     offer extensions to queue          // Extensions processed AFTER all siblings
     * </pre>
     *
     * <h2>BFS Level-by-Level Processing:</h2>
     * <pre>
     * Level 2: {A,B}, {A,C}, {A,D}, {B,C}, {B,D}, {C,D}  ← Process ALL first
     * Level 3: {A,B,C}, {A,B,D}, {A,C,D}, {B,C,D}        ← Then ALL of these
     * Level 4: {A,B,C,D}                                  ← Finally this
     * </pre>
     *
     * @param frequent1itemsets The list of 1-itemsets (not used directly)
     */
    @Override
    protected void performBestFirstMining(List<FrequentItemset> frequent1itemsets) {
        // Reset statistics
        candidatesExplored = 0;
        candidatesPruned = 0;
        candidatesPerLevel.clear();

        /**
         * BFS MAIN LOOP
         *
         * Key differences from other versions:
         * 1. queue.poll() - FIFO ordering (oldest first)
         * 2. continue vs break - cannot early terminate
         * 3. Extensions added to BACK of queue - processed after all siblings
         */
        while (!queue.isEmpty()) {
            // Track max queue size for memory analysis
            if (queue.size() > maxQueueSize) {
                maxQueueSize = queue.size();
            }

            // Poll from queue (FIFO - First In, First Out)
            FrequentItemset candidate = queue.poll();
            candidatesExplored++;

            // Track candidates per level
            int level = candidate.size();
            candidatesPerLevel.merge(level, 1L, Long::sum);

            int threshold = getThreshold();

            /**
             * PRUNING: Skip candidates below threshold
             *
             * SAME AS DFS - Cannot use early termination like Best-First
             * because queue has candidates from different levels with
             * mixed support values.
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
             * ADD EXTENSIONS TO QUEUE
             *
             * BFS behavior: Extensions are added to the BACK of the queue
             * and will be processed AFTER all candidates currently in queue.
             *
             * This creates the "breadth-first" behavior:
             *   Queue: [{A,B}, {A,C}, {B,C}]          Level 2
             *   Process {A,B} → add {A,B,C}, {A,B,D}
             *   Queue: [{A,C}, {B,C}, {A,B,C}, {A,B,D}]
             *                  ↑ Level 2   ↑ Level 3
             *   Process {A,C} next (not {A,B,C})
             */
            int newThreshold = getThreshold();

            for (FrequentItemset ext : result.getExtensions()) {
                if (ext.getSupport() >= newThreshold) {
                    queue.offer(ext);  // Add to back of queue
                } else {
                    candidatesPruned++;
                }
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
     *
     * @return number of candidates polled from queue
     */
    public long getCandidatesExplored() {
        return candidatesExplored;
    }

    /**
     * Get the number of candidates pruned during mining.
     *
     * @return number of candidates skipped due to threshold
     */
    public long getCandidatesPruned() {
        return candidatesPruned;
    }

    /**
     * Get the maximum queue size reached during mining.
     * Useful for memory analysis - BFS can have large queues.
     *
     * @return maximum number of candidates in queue at any point
     */
    public long getMaxQueueSize() {
        return maxQueueSize;
    }

    /**
     * Get the number of candidates processed at each level.
     * Useful for understanding BFS level-by-level behavior.
     *
     * @return map from itemset size to count of candidates at that size
     */
    public Map<Integer, Long> getCandidatesPerLevel() {
        return new HashMap<>(candidatesPerLevel);
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