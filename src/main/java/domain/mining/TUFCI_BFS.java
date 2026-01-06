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
 * @author Dang Nguyen Le, Gia Huy Vo
 * @see TUFCI The Best-First Search version
 * @see TUFCI_DFS The Depth-First Search version
 */
public class TUFCI_BFS extends AbstractMiner {

    // ==================== Instance Variables ====================

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
        super(database, tau, k, new DirectConvolutionSupportCalculator(tau));
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
        super(database, tau, k, calculator);
    }

    // ==================== Phase 1: Compute Frequent 1-Itemsets ====================
    // Implemented in AbstractFrequentItemsetMiner - no override needed

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
        this.topK = new TopKHeap(getK());

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
    protected void executePhase3(List<FrequentItemset> frequent1itemsets) {
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

    // ==================== Utility Methods & Closure Checking ====================
    // All utility methods and closure checking methods are now implemented in
    // AbstractFrequentItemsetMiner to eliminate code duplication.
    // Available methods: createSingletonItemset, getThreshold, getItemSupport,
    // getMaxItemIndex, checkClosure1Itemset, checkClosureAndGenerateExtensions
}