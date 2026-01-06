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
 * @author Dang Nguyen Le, Gia Huy Vo
 * @see TUFCI The Best-First Search version for comparison
 */
public class TUFCI_DFS extends AbstractMiner {

    // ==================== Instance Variables ====================

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
    public TUFCI_DFS(UncertainDatabase database, double tau, int k,
                     SupportCalculator calculator) {
        super(database, tau, k, calculator);
    }

    // ==================== Phase 1: Compute Frequent 1-Itemsets ====================
    // Implemented in AbstractFrequentItemsetMiner - no override needed
    // This phase is IDENTICAL across all search strategies

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
        this.topK = new TopKHeap(getK());

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
    protected void executePhase3(List<FrequentItemset> frequent1itemsets) {
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

    // ==================== Utility Methods & Closure Checking ====================
    // All utility methods and closure checking methods are now implemented in
    // AbstractFrequentItemsetMiner to eliminate code duplication.
    // Available methods: createSingletonItemset, getThreshold, getItemSupport,
    // getMaxItemIndex, checkClosure1Itemset, checkClosureAndGenerateExtensions
}