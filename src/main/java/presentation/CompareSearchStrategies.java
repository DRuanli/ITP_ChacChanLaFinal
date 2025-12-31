package presentation;

import domain.mining.TUFCI;
import domain.mining.TUFCI_DFS;
import domain.mining.TUFCI_BFS;
import domain.model.FrequentItemset;
import infrastructure.persistence.UncertainDatabase;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * CompareSearchStrategies - Compares Best-First Search vs DFS vs BFS for TUFCI algorithm.
 *
 * <h2>Purpose:</h2>
 * <p>This class runs TUFCI with three different search strategies on the same
 * database and compares their performance metrics.</p>
 *
 * <h2>Search Strategies:</h2>
 * <ul>
 *   <li><b>Best-First:</b> PriorityQueue - always processes highest support first</li>
 *   <li><b>DFS:</b> Stack (LIFO) - explores deeply before backtracking</li>
 *   <li><b>BFS:</b> Queue (FIFO) - explores level-by-level</li>
 * </ul>
 *
 * <h2>Expected Observations:</h2>
 * <ul>
 *   <li>All three should produce IDENTICAL results (same Top-K patterns)</li>
 *   <li>Best-First should be FASTEST (most effective pruning)</li>
 *   <li>DFS and BFS explore more candidates due to less effective early termination</li>
 *   <li>BFS may have higher memory usage (wide search tree)</li>
 * </ul>
 *
 * @author Dang Nguyen Le
 */
public class CompareSearchStrategies {

    // Store results for comparison
    private static List<FrequentItemset> resultsBestFirst;
    private static List<FrequentItemset> resultsDFS;
    private static List<FrequentItemset> resultsBFS;

    private static long timeBestFirst;
    private static long timeDFS;
    private static long timeBFS;

    private static long memoryBestFirst;
    private static long memoryDFS;
    private static long memoryBFS;

    private static TUFCI_DFS minerDFS;
    private static TUFCI_BFS minerBFS;

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            printUsage();
            return;
        }

        String dbFile = args[0];
        double tau = args.length > 1 ? Double.parseDouble(args[1]) : 0.7;
        int k = args.length > 2 ? Integer.parseInt(args[2]) : 5;

        // Print header
        System.out.println("╔════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║       TUFCI Search Strategy Comparison: Best-First vs DFS vs BFS      ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        // Print configuration
        System.out.println("Configuration:");
        System.out.println("  Database file     : " + dbFile);
        System.out.println("  Tau (threshold)   : " + tau);
        System.out.println("  K (top patterns)  : " + k);
        System.out.println();

        // Load database to get info
        System.out.println("Loading database...");
        UncertainDatabase database = UncertainDatabase.loadFromFile(dbFile);
        System.out.println("  Transactions : " + database.size());
        System.out.println("  Vocabulary   : " + database.getVocabulary().size() + " unique items");
        System.out.println();

        Runtime runtime = Runtime.getRuntime();

        // ==================== Run Best-First Search (TUFCI) ====================
        System.out.println("═".repeat(75));
        System.out.println("  [1/3] Running TUFCI with BEST-FIRST SEARCH (PriorityQueue)...");
        System.out.println("═".repeat(75));

        database = UncertainDatabase.loadFromFile(dbFile);
        runtime.gc();
        long memoryBeforeBestFirst = runtime.totalMemory() - runtime.freeMemory();

        TUFCI minerBestFirst = new TUFCI(database, tau, k);
        long startBestFirst = System.nanoTime();
        resultsBestFirst = minerBestFirst.mine();
        long endBestFirst = System.nanoTime();
        timeBestFirst = (endBestFirst - startBestFirst) / 1_000_000;

        long memoryAfterBestFirst = runtime.totalMemory() - runtime.freeMemory();
        memoryBestFirst = memoryAfterBestFirst - memoryBeforeBestFirst;

        System.out.println("  ✓ Completed in " + timeBestFirst + " ms");
        System.out.println("  ✓ Found " + resultsBestFirst.size() + " patterns");
        System.out.println();

        // ==================== Run Depth-First Search (TUFCI_DFS) ====================
        System.out.println("═".repeat(75));
        System.out.println("  [2/3] Running TUFCI with DEPTH-FIRST SEARCH (Stack - LIFO)...");
        System.out.println("═".repeat(75));

        database = UncertainDatabase.loadFromFile(dbFile);
        runtime.gc();
        long memoryBeforeDFS = runtime.totalMemory() - runtime.freeMemory();

        minerDFS = new TUFCI_DFS(database, tau, k);
        long startDFS = System.nanoTime();
        resultsDFS = minerDFS.mine();
        long endDFS = System.nanoTime();
        timeDFS = (endDFS - startDFS) / 1_000_000;

        long memoryAfterDFS = runtime.totalMemory() - runtime.freeMemory();
        memoryDFS = memoryAfterDFS - memoryBeforeDFS;

        System.out.println("  ✓ Completed in " + timeDFS + " ms");
        System.out.println("  ✓ Found " + resultsDFS.size() + " patterns");
        System.out.println("  ✓ Candidates explored: " + minerDFS.getCandidatesExplored());
        System.out.println("  ✓ Candidates pruned:   " + minerDFS.getCandidatesPruned());
        System.out.println();

        // ==================== Run Breadth-First Search (TUFCI_BFS) ====================
        System.out.println("═".repeat(75));
        System.out.println("  [3/3] Running TUFCI with BREADTH-FIRST SEARCH (Queue - FIFO)...");
        System.out.println("═".repeat(75));

        database = UncertainDatabase.loadFromFile(dbFile);
        runtime.gc();
        long memoryBeforeBFS = runtime.totalMemory() - runtime.freeMemory();

        minerBFS = new TUFCI_BFS(database, tau, k);
        long startBFS = System.nanoTime();
        resultsBFS = minerBFS.mine();
        long endBFS = System.nanoTime();
        timeBFS = (endBFS - startBFS) / 1_000_000;

        long memoryAfterBFS = runtime.totalMemory() - runtime.freeMemory();
        memoryBFS = memoryAfterBFS - memoryBeforeBFS;

        System.out.println("  ✓ Completed in " + timeBFS + " ms");
        System.out.println("  ✓ Found " + resultsBFS.size() + " patterns");
        System.out.println("  ✓ Candidates explored: " + minerBFS.getCandidatesExplored());
        System.out.println("  ✓ Candidates pruned:   " + minerBFS.getCandidatesPruned());
        System.out.println("  ✓ Max queue size:      " + minerBFS.getMaxQueueSize());
        System.out.println();

        // ==================== Print Comparison Results ====================
        printComparisonResults();
        printTraversalVisualization();
        printTopPatterns(Math.min(k, 10));
        printAnalysis();
    }

    /**
     * Print the main comparison results table.
     */
    private static void printComparisonResults() {
        System.out.println("╔════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                        COMPARISON RESULTS                              ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        // Timing comparison
        System.out.println("┌────────────────────────────────────────────────────────────────────────┐");
        System.out.println("│                         EXECUTION TIME                                 │");
        System.out.println("├─────────────────────────────┬──────────────┬───────────────────────────┤");
        System.out.println("│ Strategy                    │ Time (ms)    │ Relative to Best-First    │");
        System.out.println("├─────────────────────────────┼──────────────┼───────────────────────────┤");

        System.out.printf("│ %-27s │ %10d   │ %23s   │%n",
            "Best-First (PriorityQueue)", timeBestFirst, "1.00x (baseline)");

        String dfsRatio = timeBestFirst > 0 ?
            String.format("%.2fx", (double) timeDFS / timeBestFirst) : "N/A";
        System.out.printf("│ %-27s │ %10d   │ %23s   │%n",
            "DFS (Stack - LIFO)", timeDFS, dfsRatio);

        String bfsRatio = timeBestFirst > 0 ?
            String.format("%.2fx", (double) timeBFS / timeBestFirst) : "N/A";
        System.out.printf("│ %-27s │ %10d   │ %23s   │%n",
            "BFS (Queue - FIFO)", timeBFS, bfsRatio);

        System.out.println("└─────────────────────────────┴──────────────┴───────────────────────────┘");
        System.out.println();

        // Memory comparison
        System.out.println("┌────────────────────────────────────────────────────────────────────────┐");
        System.out.println("│                          MEMORY USAGE                                  │");
        System.out.println("├─────────────────────────────┬──────────────────────────────────────────┤");
        System.out.printf("│ %-27s │ %38.2f MB │%n",
            "Best-First (PriorityQueue)", memoryBestFirst / (1024.0 * 1024.0));
        System.out.printf("│ %-27s │ %38.2f MB │%n",
            "DFS (Stack - LIFO)", memoryDFS / (1024.0 * 1024.0));
        System.out.printf("│ %-27s │ %38.2f MB │%n",
            "BFS (Queue - FIFO)", memoryBFS / (1024.0 * 1024.0));
        System.out.println("└─────────────────────────────┴──────────────────────────────────────────┘");
        System.out.println();

        // Exploration statistics
        System.out.println("┌────────────────────────────────────────────────────────────────────────┐");
        System.out.println("│                      EXPLORATION STATISTICS                            │");
        System.out.println("├─────────────────────────────┬──────────────┬──────────────┬────────────┤");
        System.out.println("│ Strategy                    │ Explored     │ Pruned       │ Prune Rate │");
        System.out.println("├─────────────────────────────┼──────────────┼──────────────┼────────────┤");

        System.out.printf("│ %-27s │ %12s │ %12s │ %10s │%n",
            "Best-First (PriorityQueue)", "N/A*", "N/A*", "N/A*");

        long totalDFS = minerDFS.getCandidatesExplored() + minerDFS.getCandidatesPruned();
        double pruneRateDFS = totalDFS > 0 ? (minerDFS.getCandidatesPruned() * 100.0 / totalDFS) : 0;
        System.out.printf("│ %-27s │ %,12d │ %,12d │ %9.1f%% │%n",
            "DFS (Stack - LIFO)", minerDFS.getCandidatesExplored(),
            minerDFS.getCandidatesPruned(), pruneRateDFS);

        long totalBFS = minerBFS.getCandidatesExplored() + minerBFS.getCandidatesPruned();
        double pruneRateBFS = totalBFS > 0 ? (minerBFS.getCandidatesPruned() * 100.0 / totalBFS) : 0;
        System.out.printf("│ %-27s │ %,12d │ %,12d │ %9.1f%% │%n",
            "BFS (Queue - FIFO)", minerBFS.getCandidatesExplored(),
            minerBFS.getCandidatesPruned(), pruneRateBFS);

        System.out.println("└─────────────────────────────┴──────────────┴──────────────┴────────────┘");
        System.out.println("  * Best-First uses early termination - statistics not directly comparable");
        System.out.println();

        // BFS level statistics
        System.out.println("┌────────────────────────────────────────────────────────────────────────┐");
        System.out.println("│                    BFS LEVEL-BY-LEVEL BREAKDOWN                        │");
        System.out.println("├─────────────────────────────┬──────────────────────────────────────────┤");
        System.out.printf("│ %-27s │ %40d │%n", "Max Queue Size", minerBFS.getMaxQueueSize());
        System.out.println("├─────────────────────────────┴──────────────────────────────────────────┤");

        Map<Integer, Long> levelStats = minerBFS.getCandidatesPerLevel();
        if (!levelStats.isEmpty()) {
            System.out.println("│ Candidates processed per level (itemset size):                         │");
            for (int level = 2; level <= levelStats.keySet().stream().max(Integer::compareTo).orElse(2); level++) {
                Long count = levelStats.getOrDefault(level, 0L);
                if (count > 0) {
                    System.out.printf("│   Level %d (size-%d itemsets): %,45d │%n", level, level, count);
                }
            }
        }
        System.out.println("└────────────────────────────────────────────────────────────────────────┘");
        System.out.println();

        // Result verification
        System.out.println("┌────────────────────────────────────────────────────────────────────────┐");
        System.out.println("│                       RESULT VERIFICATION                              │");
        System.out.println("├────────────────────────────────────────────────────────────────────────┤");

        boolean bfVsDfs = verifyResults(resultsBestFirst, resultsDFS);
        boolean bfVsBfs = verifyResults(resultsBestFirst, resultsBFS);
        boolean dfsVsBfs = verifyResults(resultsDFS, resultsBFS);

        if (bfVsDfs && bfVsBfs && dfsVsBfs) {
            System.out.println("│  ✓ All three strategies produced IDENTICAL results                     │");
        } else {
            System.out.println("│  ✗ WARNING: Results differ! This may indicate a bug.                   │");
            System.out.printf("│    Best-First vs DFS: %s                                              │%n",
                bfVsDfs ? "MATCH" : "DIFFER");
            System.out.printf("│    Best-First vs BFS: %s                                              │%n",
                bfVsBfs ? "MATCH" : "DIFFER");
            System.out.printf("│    DFS vs BFS:        %s                                              │%n",
                dfsVsBfs ? "MATCH" : "DIFFER");
        }

        System.out.printf("│  Patterns found: Best-First=%d, DFS=%d, BFS=%d                      │%n",
            resultsBestFirst.size(), resultsDFS.size(), resultsBFS.size());
        System.out.println("└────────────────────────────────────────────────────────────────────────┘");
        System.out.println();
    }

    /**
     * Print visualization of different traversal strategies.
     */
    private static void printTraversalVisualization() {
        System.out.println("╔════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                    TRAVERSAL ORDER VISUALIZATION                       ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("  Given itemset lattice with supports:");
        System.out.println();
        System.out.println("                         {}");
        System.out.println("                    _____|_____");
        System.out.println("                   |     |     |");
        System.out.println("                 {A}   {B}   {C}          ← Level 1 (1-itemsets)");
        System.out.println("                sup:100 sup:80 sup:60");
        System.out.println("                 / \\   / \\   / ");
        System.out.println("              {A,B} {A,C} {B,C}           ← Level 2 (2-itemsets)");
        System.out.println("              sup:70 sup:50 sup:40");
        System.out.println("                  \\   |   /");
        System.out.println("                   {A,B,C}                ← Level 3 (3-itemsets)");
        System.out.println("                    sup:30");
        System.out.println();
        System.out.println("  ┌─────────────────────────────────────────────────────────────────┐");
        System.out.println("  │ BEST-FIRST (PriorityQueue - by support):                        │");
        System.out.println("  │   {A}:100 → {B}:80 → {A,B}:70 → {C}:60 → {A,C}:50 → ...        │");
        System.out.println("  │   Always picks highest support next                             │");
        System.out.println("  │   ✓ Early termination when best < threshold                     │");
        System.out.println("  └─────────────────────────────────────────────────────────────────┘");
        System.out.println();
        System.out.println("  ┌─────────────────────────────────────────────────────────────────┐");
        System.out.println("  │ DFS (Stack - LIFO):                                             │");
        System.out.println("  │   {A} → {A,B} → {A,B,C} → backtrack → {A,C} → backtrack → {B}  │");
        System.out.println("  │   Goes deep before exploring siblings                           │");
        System.out.println("  │   ✗ Cannot early terminate (mixed support in stack)             │");
        System.out.println("  └─────────────────────────────────────────────────────────────────┘");
        System.out.println();
        System.out.println("  ┌─────────────────────────────────────────────────────────────────┐");
        System.out.println("  │ BFS (Queue - FIFO):                                             │");
        System.out.println("  │   {A} → {B} → {C} → {A,B} → {A,C} → {B,C} → {A,B,C}            │");
        System.out.println("  │   Completes each level before going deeper                      │");
        System.out.println("  │   ✗ Cannot early terminate (mixed support in queue)             │");
        System.out.println("  └─────────────────────────────────────────────────────────────────┘");
        System.out.println();
    }

    /**
     * Print top patterns comparison.
     */
    private static void printTopPatterns(int count) {
        System.out.println("╔════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                        TOP PATTERNS COMPARISON                         ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        int displayCount = Math.min(count,
            Math.min(resultsBestFirst.size(),
                Math.min(resultsDFS.size(), resultsBFS.size())));

        if (displayCount == 0) {
            System.out.println("  No patterns to display.");
            System.out.println();
            return;
        }

        System.out.printf("  %-4s │ %-25s %6s │ %-25s %6s │ %-25s %6s%n",
            "Rank", "Best-First", "Sup", "DFS", "Sup", "BFS", "Sup");
        System.out.println("  " + "─".repeat(95));

        for (int i = 0; i < displayCount; i++) {
            FrequentItemset bf = resultsBestFirst.get(i);
            FrequentItemset dfs = resultsDFS.get(i);
            FrequentItemset bfs = resultsBFS.get(i);

            System.out.printf("  %-4d │ %-25s %6d │ %-25s %6d │ %-25s %6d%n",
                i + 1,
                truncate(bf.toStringWithCodec(), 25), bf.getSupport(),
                truncate(dfs.toStringWithCodec(), 25), dfs.getSupport(),
                truncate(bfs.toStringWithCodec(), 25), bfs.getSupport());
        }
        System.out.println();
    }

    /**
     * Print analysis and recommendations.
     */
    private static void printAnalysis() {
        System.out.println("╔════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                              ANALYSIS                                  ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        System.out.println("  PERFORMANCE RANKING:");
        System.out.println("  " + "─".repeat(70));

        // Sort by time
        long[] times = {timeBestFirst, timeDFS, timeBFS};
        String[] names = {"Best-First", "DFS", "BFS"};

        // Simple bubble sort for 3 elements
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2 - i; j++) {
                if (times[j] > times[j + 1]) {
                    long tmpTime = times[j];
                    times[j] = times[j + 1];
                    times[j + 1] = tmpTime;
                    String tmpName = names[j];
                    names[j] = names[j + 1];
                    names[j + 1] = tmpName;
                }
            }
        }

        System.out.printf("    1st: %-15s (%d ms)%n", names[0], times[0]);
        System.out.printf("    2nd: %-15s (%d ms)%n", names[1], times[1]);
        System.out.printf("    3rd: %-15s (%d ms)%n", names[2], times[2]);
        System.out.println();

        System.out.println("  WHY BEST-FIRST IS TYPICALLY FASTEST FOR TOP-K:");
        System.out.println("  " + "─".repeat(70));
        System.out.println("    1. EARLY TERMINATION:");
        System.out.println("       When the best remaining candidate has support < threshold,");
        System.out.println("       Best-First can stop immediately (all others are worse).");
        System.out.println("       DFS/BFS must continue checking (mixed support values).");
        System.out.println();
        System.out.println("    2. FAST THRESHOLD RISE:");
        System.out.println("       High-support patterns enter Top-K first → threshold rises fast");
        System.out.println("       → more candidates pruned earlier.");
        System.out.println();
        System.out.println("    3. OPTIMAL SELECTION:");
        System.out.println("       Always processes most promising candidate next,");
        System.out.println("       minimizing wasted computation on low-support patterns.");
        System.out.println();

        System.out.println("  WHY DFS MIGHT BE USEFUL:");
        System.out.println("  " + "─".repeat(70));
        System.out.println("    - Lower memory for deep, narrow search spaces");
        System.out.println("    - Simpler implementation (stack vs priority queue)");
        System.out.println("    - Good for finding ANY solution quickly (not Top-K)");
        System.out.println();

        System.out.println("  WHY BFS MIGHT BE USEFUL:");
        System.out.println("  " + "─".repeat(70));
        System.out.println("    - Guarantees finding smallest (shortest) patterns first");
        System.out.println("    - Complete level-by-level exploration");
        System.out.println("    - Predictable memory usage per level");
        System.out.println();

        System.out.println("  RECOMMENDATION:");
        System.out.println("  " + "─".repeat(70));
        System.out.println("    Use BEST-FIRST SEARCH (TUFCI) for production Top-K mining.");
        System.out.println("    DFS and BFS are provided for educational comparison only.");
        System.out.println();
    }

    /**
     * Verify that two result lists contain the same patterns.
     */
    private static boolean verifyResults(List<FrequentItemset> list1, List<FrequentItemset> list2) {
        if (list1.size() != list2.size()) {
            return false;
        }

        for (int i = 0; i < list1.size(); i++) {
            FrequentItemset fi1 = list1.get(i);
            FrequentItemset fi2 = list2.get(i);

            if (!fi1.equals(fi2) || fi1.getSupport() != fi2.getSupport()) {
                return false;
            }
        }

        return true;
    }

    /**
     * Truncate string to specified length.
     */
    private static String truncate(String s, int maxLen) {
        if (s.length() <= maxLen) {
            return s;
        }
        return s.substring(0, maxLen - 3) + "...";
    }

    /**
     * Print usage information.
     */
    private static void printUsage() {
        System.out.println("╔════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║       TUFCI Search Strategy Comparison: Best-First vs DFS vs BFS      ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  java CompareSearchStrategies <database_file> [tau] [k]");
        System.out.println();
        System.out.println("Arguments:");
        System.out.println("  database_file : Path to the uncertain database file (required)");
        System.out.println("  tau          : Probability threshold (optional, default: 0.7)");
        System.out.println("  k            : Number of top patterns (optional, default: 5)");
        System.out.println();
        System.out.println("Example:");
        System.out.println("  java CompareSearchStrategies data/database.txt 0.7 10");
        System.out.println();
        System.out.println("This tool compares three search strategies:");
        System.out.println("  1. Best-First Search (PriorityQueue) - RECOMMENDED");
        System.out.println("  2. Depth-First Search (Stack - LIFO)");
        System.out.println("  3. Breadth-First Search (Queue - FIFO)");
        System.out.println();
    }
}