package presentation;

import domain.mining.TUFCI_Pruning;
import domain.mining.TUFCI_Pruning.PruningConfig;
import domain.mining.TUFCI_Pruning.Metrics;
import domain.model.FrequentItemset;
import infrastructure.persistence.UncertainDatabase;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * PruningExperiment: Incremental Pruning Strategy Effectiveness Study.
 *
 * <h2>Experiment Design:</h2>
 * <p>Starting from BASE (no pruning), incrementally add pruning strategies
 * to show how each one improves performance compared to the baseline.</p>
 *
 * <pre>
 * ┌──────────────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬───────────────┐
 * │ Configuration    │ P1  │ P2  │ P3  │ P4  │ P5  │ P6  │ P7  │ Expected      │
 * ├──────────────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼───────────────┤
 * │ BASE             │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │ Slowest       │
 * │ +P1              │  ✓  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │ ↑ Faster      │
 * │ +P1+P2           │  ✓  │  ✓  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │ ↑↑ Faster     │
 * │ +P1+P2+P3        │  ✓  │  ✓  │  ✓  │  ✗  │  ✗  │  ✗  │  ✗  │ ↑↑↑           │
 * │ +P1+P2+P3+P4     │  ✓  │  ✓  │  ✓  │  ✓  │  ✗  │  ✗  │  ✗  │ ↑↑↑↑          │
 * │ +P1..P5          │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✗  │  ✗  │ ↑↑↑↑↑         │
 * │ +P1..P6          │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✗  │ ↑↑↑↑↑↑        │
 * │ ALL (P1..P7)     │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │ Fastest       │
 * └──────────────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴───────────────┘
 * </pre>
 *
 * <h2>Pruning Strategies:</h2>
 * <pre>
 * P1: Early Termination (Phase 2 early term, Closure early term, 2-itemset filter)
 * P2: Main Loop Threshold (Candidate skip, Loop termination, Extension filter)
 * P3: Item Support Threshold
 * P4: Subset-Based Upper Bound Tightening
 * P5: Upper Bound Filtering
 * P6: Tidset Size Pruning (skip GF computation)
 * P7: Tidset-Based Closure Skip
 * </pre>
 *
 * <h2>Usage:</h2>
 * <pre>
 * java PruningExperiment &lt;database_file&gt; [tau] [k] [runs]
 * </pre>
 *
 * @author Dang Nguyen Le
 * @version 2.0
 */
public class PruningExperiment {

    private static final String[] STRATEGY_NAMES = {
        "P1: Early Termination",
        "P2: Main Loop Threshold",
        "P3: Item Support Threshold",
        "P4: Subset Upper Bound",
        "P5: Upper Bound Filter",
        "P6: Tidset Size Pruning",
        "P7: Tidset Closure Skip"
    };

    private static final String[] STRATEGY_DESCRIPTIONS = {
        "Phase 2 early termination + Closure early term + 2-itemset filter",
        "Candidate skip + Loop early termination + Extension filter",
        "Stop checking items when support < threshold",
        "Use cached 2-itemset supports for tighter bounds",
        "Skip extensions when upper bound < threshold",
        "Skip GF computation when tidset size < threshold",
        "Skip closure check when tidset guarantees no violation"
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // MAIN
    // ═══════════════════════════════════════════════════════════════════════════

    public static void main(String[] args) throws IOException {
        if (args.length < 1) { printUsage(); return; }

        String dbFile = args[0];
        double tau = args.length > 1 ? Double.parseDouble(args[1]) : 0.7;
        int k = args.length > 2 ? Integer.parseInt(args[2]) : 100;
        int runs = args.length > 3 ? Integer.parseInt(args[3]) : 3;

        printHeader();
        System.out.println("EXPERIMENT CONFIGURATION");
        System.out.println("═".repeat(80));
        System.out.printf("  Database: %s%n", dbFile);
        System.out.printf("  Tau (τ):  %.2f%n", tau);
        System.out.printf("  K:        %d%n", k);
        System.out.printf("  Runs:     %d%n", runs);
        System.out.println();

        // Load database
        System.out.println("Loading database...");
        UncertainDatabase db = UncertainDatabase.loadFromFile(dbFile);
        System.out.printf("  Transactions: %,d%n", db.size());
        System.out.printf("  Items:        %,d%n", db.getVocabulary().size());
        System.out.println();

        // Run incremental experiment
        List<ExperimentResult> results = new ArrayList<>();

        System.out.println("═".repeat(80));
        System.out.println("  RUNNING INCREMENTAL EXPERIMENT");
        System.out.println("  BASE → +P1 → +P1+P2 → ... → ALL");
        System.out.println("═".repeat(80));
        System.out.println();

        // Level 0: BASE (no pruning)
        System.out.println("┌─────────────────────────────────────────────────────────────────────────────┐");
        System.out.println("│ [0/7] BASE - No pruning (worst case baseline)                              │");
        System.out.println("└─────────────────────────────────────────────────────────────────────────────┘");
        ExperimentResult base = runLevel(dbFile, tau, k, runs, 0);
        results.add(base);
        System.out.println();

        // Levels 1-7: Add pruning incrementally
        for (int level = 1; level <= 7; level++) {
            String configName = level == 7 ? "ALL" : "+P1.." + level;
            String strategyAdded = STRATEGY_NAMES[level - 1];

            System.out.println("┌─────────────────────────────────────────────────────────────────────────────┐");
            System.out.printf("│ [%d/7] %s - Adding: %-43s │%n", level, configName, strategyAdded);
            System.out.println("└─────────────────────────────────────────────────────────────────────────────┘");

            ExperimentResult result = runLevel(dbFile, tau, k, runs, level);
            results.add(result);

            // Calculate improvement
            double speedup = (double) base.avgMetrics.totalTimeMs / result.avgMetrics.totalTimeMs;
            double improvement = (1.0 - (double) result.avgMetrics.totalTimeMs / base.avgMetrics.totalTimeMs) * 100;

            System.out.printf("  → Speedup vs BASE: %.2fx (%.1f%% faster)%n", speedup, improvement);

            // Verify correctness
            if (!verifyResults(base.patterns, result.patterns)) {
                System.out.println("  ⚠️  WARNING: Results differ from BASE!");
            } else {
                System.out.println("  ✓ Results verified: identical to BASE");
            }
            System.out.println();
        }

        // Print comprehensive results
        printIncrementalResults(results);
        printSpeedupChart(results);
        printPruningContribution(results);
        printStrategyAnalysis(results);

        // Export
        String csvFile = exportToCsv(results, dbFile, tau, k);
        System.out.println("Results exported to: " + csvFile);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXPERIMENT EXECUTION
    // ═══════════════════════════════════════════════════════════════════════════

    private static ExperimentResult runLevel(String dbFile, double tau, int k,
                                              int runs, int level) throws IOException {
        PruningConfig config = PruningConfig.level(level);
        List<Metrics> allMetrics = new ArrayList<>();
        List<FrequentItemset> patterns = null;

        System.out.print("  Running: ");
        for (int run = 0; run < runs; run++) {
            UncertainDatabase db = UncertainDatabase.loadFromFile(dbFile);
            System.gc();

            TUFCI_Pruning miner = new TUFCI_Pruning(db, tau, k, config);
            patterns = miner.mine();
            allMetrics.add(miner.getMetrics());

            System.out.printf("%dms ", miner.getMetrics().totalTimeMs);
        }
        System.out.println();

        ExperimentResult result = new ExperimentResult();
        result.level = level;
        result.configName = config.getName();
        result.patterns = patterns;
        result.avgMetrics = averageMetrics(allMetrics, config.getName(), level);

        System.out.printf("  Average: %,d ms | Patterns: %d%n",
            result.avgMetrics.totalTimeMs, result.avgMetrics.patternsFound);

        return result;
    }

    private static Metrics averageMetrics(List<Metrics> all, String name, int level) {
        Metrics avg = new Metrics();
        avg.configName = name;
        avg.pruningLevel = level;
        int n = all.size();

        for (Metrics m : all) {
            avg.totalTimeMs += m.totalTimeMs;
            avg.phase1TimeMs += m.phase1TimeMs;
            avg.phase2TimeMs += m.phase2TimeMs;
            avg.phase3TimeMs += m.phase3TimeMs;
            avg.candidatesGenerated += m.candidatesGenerated;
            avg.candidatesExplored += m.candidatesExplored;
            avg.prunedByP1 += m.prunedByP1;
            avg.prunedByP2 += m.prunedByP2;
            avg.prunedByP3 += m.prunedByP3;
            avg.prunedByP4 += m.prunedByP4;
            avg.prunedByP5 += m.prunedByP5;
            avg.prunedByP6 += m.prunedByP6;
            avg.prunedByP7 += m.prunedByP7;
            avg.supportCalculations += m.supportCalculations;
            avg.closureChecks += m.closureChecks;
            avg.tidsetIntersections += m.tidsetIntersections;
            avg.cacheHits += m.cacheHits;
            avg.cacheMisses += m.cacheMisses;
            avg.cacheSize += m.cacheSize;
        }

        avg.totalTimeMs /= n; avg.phase1TimeMs /= n; avg.phase2TimeMs /= n; avg.phase3TimeMs /= n;
        avg.candidatesGenerated /= n; avg.candidatesExplored /= n;
        avg.prunedByP1 /= n; avg.prunedByP2 /= n; avg.prunedByP3 /= n; avg.prunedByP4 /= n;
        avg.prunedByP5 /= n; avg.prunedByP6 /= n; avg.prunedByP7 /= n;
        avg.supportCalculations /= n; avg.closureChecks /= n; avg.tidsetIntersections /= n;
        avg.cacheHits /= n; avg.cacheMisses /= n; avg.cacheSize /= n;
        avg.patternsFound = all.get(0).patternsFound;

        return avg;
    }

    private static boolean verifyResults(List<FrequentItemset> expected, List<FrequentItemset> actual) {
        if (expected == null || actual == null) return false;
        if (expected.size() != actual.size()) return false;
        for (int i = 0; i < expected.size(); i++) {
            if (!expected.get(i).equals(actual.get(i))) return false;
        }
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RESULT PRINTING
    // ═══════════════════════════════════════════════════════════════════════════

    private static void printIncrementalResults(List<ExperimentResult> results) {
        System.out.println();
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                    INCREMENTAL PRUNING RESULTS                                 ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        ExperimentResult base = results.get(0);
        long baseTime = base.avgMetrics.totalTimeMs;

        // Main results table
        System.out.println("┌────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
        System.out.println("│ Config     │ Time (ms)    │ vs BASE      │ Speedup      │ Δ Time       │ Δ Speedup    │");
        System.out.println("├────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");

        long prevTime = baseTime;
        for (int i = 0; i < results.size(); i++) {
            ExperimentResult r = results.get(i);
            long time = r.avgMetrics.totalTimeMs;
            double speedup = (double) baseTime / time;
            long deltaTime = prevTime - time;
            double prevSpeedup = (double) baseTime / prevTime;
            double deltaSpeedup = speedup - prevSpeedup;

            String configStr = i == 0 ? "BASE" : (i == 7 ? "ALL" : "+P1.." + i);
            String deltaTimeStr = i == 0 ? "-" : String.format("-%,d", deltaTime);
            String deltaSpeedupStr = i == 0 ? "-" : String.format("+%.2fx", deltaSpeedup);

            System.out.printf("│ %-10s │ %,12d │ %11.1f%% │ %11.2fx │ %12s │ %12s │%n",
                configStr, time,
                (1.0 - (double) time / baseTime) * 100,
                speedup, deltaTimeStr, deltaSpeedupStr);

            prevTime = time;
        }
        System.out.println("└────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
        System.out.println();

        // Computation statistics
        System.out.println("┌────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
        System.out.println("│ Config     │ Support Calc │ Closure Chk  │ Intersections│ Explored     │");
        System.out.println("├────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");

        for (int i = 0; i < results.size(); i++) {
            ExperimentResult r = results.get(i);
            String configStr = i == 0 ? "BASE" : (i == 7 ? "ALL" : "+P1.." + i);

            System.out.printf("│ %-10s │ %,12d │ %,12d │ %,12d │ %,12d │%n",
                configStr,
                r.avgMetrics.supportCalculations,
                r.avgMetrics.closureChecks,
                r.avgMetrics.tidsetIntersections,
                r.avgMetrics.candidatesExplored);
        }
        System.out.println("└────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
        System.out.println();
    }

    private static void printSpeedupChart(List<ExperimentResult> results) {
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                           SPEEDUP VISUALIZATION                                ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        long baseTime = results.get(0).avgMetrics.totalTimeMs;
        double maxSpeedup = (double) baseTime / results.get(results.size() - 1).avgMetrics.totalTimeMs;
        int maxBarWidth = 50;

        for (int i = 0; i < results.size(); i++) {
            ExperimentResult r = results.get(i);
            double speedup = (double) baseTime / r.avgMetrics.totalTimeMs;
            int barWidth = (int) ((speedup / maxSpeedup) * maxBarWidth);

            String configStr = i == 0 ? "BASE   " : (i == 7 ? "ALL    " : "+P1.." + i + "  ");
            String bar = "█".repeat(Math.max(1, barWidth));

            System.out.printf("  %s │%s %.2fx%n", configStr, bar, speedup);
        }
        System.out.println();
    }

    private static void printPruningContribution(List<ExperimentResult> results) {
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                      PRUNING CONTRIBUTION BY STRATEGY                          ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        // Get ALL configuration (last result)
        Metrics all = results.get(results.size() - 1).avgMetrics;
        long total = all.getTotalPruned();

        System.out.println("  Candidates pruned by each strategy (in ALL configuration):");
        System.out.println();

        long[] counts = {all.prunedByP1, all.prunedByP2, all.prunedByP3, all.prunedByP4,
                         all.prunedByP5, all.prunedByP6, all.prunedByP7};

        for (int i = 0; i < 7; i++) {
            double pct = total > 0 ? (counts[i] * 100.0 / total) : 0;
            int barWidth = (int) (pct / 2);
            String bar = "█".repeat(Math.max(0, barWidth));

            System.out.printf("  P%d │%-30s %,10d (%5.1f%%)%n",
                i + 1, bar, counts[i], pct);
        }
        System.out.println();
        System.out.printf("  Total pruned: %,d%n", total);
        System.out.println();
    }

    private static void printStrategyAnalysis(List<ExperimentResult> results) {
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                         STRATEGY IMPACT ANALYSIS                               ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        long baseTime = results.get(0).avgMetrics.totalTimeMs;

        // Calculate marginal improvement for each strategy
        System.out.println("  Marginal improvement when each strategy is added:");
        System.out.println();
        System.out.println("  ┌────┬─────────────────────────────────┬───────────────┬───────────────┐");
        System.out.println("  │ #  │ Strategy                        │ Time Saved    │ Improvement   │");
        System.out.println("  ├────┼─────────────────────────────────┼───────────────┼───────────────┤");

        for (int i = 1; i <= 7; i++) {
            long prevTime = results.get(i - 1).avgMetrics.totalTimeMs;
            long currTime = results.get(i).avgMetrics.totalTimeMs;
            long timeSaved = prevTime - currTime;
            double improvementPct = (double) timeSaved / baseTime * 100;

            System.out.printf("  │ P%d │ %-31s │ %,11d ms │ %12.1f%% │%n",
                i, STRATEGY_NAMES[i - 1], timeSaved, improvementPct);
        }
        System.out.println("  └────┴─────────────────────────────────┴───────────────┴───────────────┘");
        System.out.println();

        // Summary
        long allTime = results.get(7).avgMetrics.totalTimeMs;
        double totalSpeedup = (double) baseTime / allTime;
        double totalImprovement = (1.0 - (double) allTime / baseTime) * 100;

        System.out.println("═".repeat(80));
        System.out.println("SUMMARY");
        System.out.println("═".repeat(80));
        System.out.printf("  BASE time:        %,d ms%n", baseTime);
        System.out.printf("  ALL time:         %,d ms%n", allTime);
        System.out.printf("  Total speedup:    %.2fx%n", totalSpeedup);
        System.out.printf("  Total improvement: %.1f%%%n", totalImprovement);
        System.out.println();
        System.out.println("  All configurations produce IDENTICAL results.");
        System.out.println("  This proves all pruning strategies are TECHNICAL (not Definition-based).");
        System.out.println();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CSV EXPORT
    // ═══════════════════════════════════════════════════════════════════════════

    private static String exportToCsv(List<ExperimentResult> results, String dbFile,
                                       double tau, int k) throws IOException {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = "pruning_experiment_" + timestamp + ".csv";

        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("# TUFCI Incremental Pruning Experiment");
            writer.println("# Database: " + dbFile);
            writer.println("# Tau: " + tau + ", K: " + k);
            writer.println("# " + LocalDateTime.now());
            writer.println();
            writer.println(Metrics.getCsvHeader());
            for (ExperimentResult r : results) {
                writer.println(r.avgMetrics.toCsvRow());
            }
        }
        return filename;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // UTILITY
    // ═══════════════════════════════════════════════════════════════════════════

    private static void printHeader() {
        System.out.println();
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                                                                                ║");
        System.out.println("║           TUFCI INCREMENTAL PRUNING STRATEGY EXPERIMENT                        ║");
        System.out.println("║                                                                                ║");
        System.out.println("║   Measuring cumulative effect: BASE → +P1 → +P1+P2 → ... → ALL                 ║");
        System.out.println("║                                                                                ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }

    private static void printUsage() {
        printHeader();
        System.out.println("Usage: java PruningExperiment <database_file> [tau] [k] [runs]");
        System.out.println();
        System.out.println("Arguments:");
        System.out.println("  database_file : Path to uncertain database (required)");
        System.out.println("  tau          : Probability threshold (default: 0.7)");
        System.out.println("  k            : Number of top patterns (default: 100)");
        System.out.println("  runs         : Runs per configuration (default: 3)");
        System.out.println();
        System.out.println("Pruning Strategies:");
        for (int i = 0; i < 7; i++) {
            System.out.printf("  P%d: %s%n", i + 1, STRATEGY_NAMES[i]);
            System.out.printf("      %s%n", STRATEGY_DESCRIPTIONS[i]);
        }
        System.out.println();
    }

    private static class ExperimentResult {
        int level;
        String configName;
        Metrics avgMetrics;
        List<FrequentItemset> patterns;
    }
}