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
 * TUFCI_Pruning: Configurable TUFCI for Incremental Pruning Strategy Experiment.
 *
 * <h2>Experiment Design: Incremental Addition</h2>
 * <pre>
 * ┌─────────────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
 * │ Configuration   │ P1  │ P2  │ P3  │ P4  │ P5  │ P6  │ P7  │
 * ├─────────────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
 * │ BASE            │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │  ← Slowest
 * │ +P1             │  ✓  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │
 * │ +P1+P2          │  ✓  │  ✓  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │
 * │ +P1+P2+P3       │  ✓  │  ✓  │  ✓  │  ✗  │  ✗  │  ✗  │  ✗  │
 * │ +P1+P2+P3+P4    │  ✓  │  ✓  │  ✓  │  ✓  │  ✗  │  ✗  │  ✗  │
 * │ +P1..P5         │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✗  │  ✗  │
 * │ +P1..P6         │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✗  │
 * │ ALL_PRUNING     │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │  ← Fastest
 * └─────────────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
 * </pre>
 *
 * <h2>Pruning Strategies (7 Categories):</h2>
 * <pre>
 * P1: Early Termination Pruning (Phase 2)
 *     - Phase 2 early termination for 1-itemsets
 *     - Closure check early termination
 *     - 2-itemset filtering before seeding
 *
 * P2: Main Loop Threshold Pruning
 *     - Candidate threshold skip
 *     - Main loop early termination (break)
 *     - Extension filtering before adding
 *
 * P3: Item Support Threshold Pruning
 *
 * P4: Subset-Based Upper Bound Tightening
 *
 * P5: Upper Bound Filtering
 *
 * P6: Tidset Size Pruning (skip GF computation)
 *
 * P7: Tidset-Based Closure Skip
 * </pre>
 *
 * @author Dang Nguyen Le
 * @version 2.0
 */
public class TUFCI_Pruning extends AbstractFrequentItemsetMiner {

    // ═══════════════════════════════════════════════════════════════════════════
    // PRUNING CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════

    public static class PruningConfig {
        public boolean P1 = true;  // Early Termination
        public boolean P2 = true;  // Main Loop Threshold
        public boolean P3 = true;  // Item Support Threshold
        public boolean P4 = true;  // Subset Upper Bound
        public boolean P5 = true;  // Upper Bound Filter
        public boolean P6 = true;  // Tidset Size Pruning
        public boolean P7 = true;  // Tidset Closure Skip

        /** BASE: No pruning (slowest). */
        public static PruningConfig base() {
            PruningConfig c = new PruningConfig();
            c.P1 = c.P2 = c.P3 = c.P4 = c.P5 = c.P6 = c.P7 = false;
            return c;
        }

        /** ALL: All pruning enabled (fastest). */
        public static PruningConfig all() {
            return new PruningConfig();
        }

        /** Incremental configurations */
        public static PruningConfig upToP1() {
            PruningConfig c = base(); c.P1 = true; return c;
        }
        public static PruningConfig upToP2() {
            PruningConfig c = upToP1(); c.P2 = true; return c;
        }
        public static PruningConfig upToP3() {
            PruningConfig c = upToP2(); c.P3 = true; return c;
        }
        public static PruningConfig upToP4() {
            PruningConfig c = upToP3(); c.P4 = true; return c;
        }
        public static PruningConfig upToP5() {
            PruningConfig c = upToP4(); c.P5 = true; return c;
        }
        public static PruningConfig upToP6() {
            PruningConfig c = upToP5(); c.P6 = true; return c;
        }
        public static PruningConfig upToP7() { return all(); }

        /** Get configuration by level (0=BASE, 7=ALL). */
        public static PruningConfig level(int n) {
            return switch (n) {
                case 0 -> base();
                case 1 -> upToP1();
                case 2 -> upToP2();
                case 3 -> upToP3();
                case 4 -> upToP4();
                case 5 -> upToP5();
                case 6 -> upToP6();
                default -> all();
            };
        }

        public String getName() {
            if (!P1 && !P2 && !P3 && !P4 && !P5 && !P6 && !P7) return "BASE";
            if (P1 && P2 && P3 && P4 && P5 && P6 && P7) return "ALL";
            StringBuilder sb = new StringBuilder();
            if (P1) sb.append("+P1");
            if (P2) sb.append("+P2");
            if (P3) sb.append("+P3");
            if (P4) sb.append("+P4");
            if (P5) sb.append("+P5");
            if (P6) sb.append("+P6");
            if (P7) sb.append("+P7");
            return sb.toString();
        }

        public int countEnabled() {
            int c = 0;
            if (P1) c++; if (P2) c++; if (P3) c++; if (P4) c++;
            if (P5) c++; if (P6) c++; if (P7) c++;
            return c;
        }

        @Override public String toString() { return getName(); }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // METRICS
    // ═══════════════════════════════════════════════════════════════════════════

    public static class Metrics {
        public String configName;
        public int pruningLevel;

        public long totalTimeMs, phase1TimeMs, phase2TimeMs, phase3TimeMs;
        public long candidatesGenerated, candidatesExplored;
        public long prunedByP1, prunedByP2, prunedByP3, prunedByP4;
        public long prunedByP5, prunedByP6, prunedByP7;
        public long supportCalculations, closureChecks, tidsetIntersections;
        public long cacheHits, cacheMisses, cacheSize;
        public int patternsFound;

        public long getTotalPruned() {
            return prunedByP1 + prunedByP2 + prunedByP3 + prunedByP4 +
                   prunedByP5 + prunedByP6 + prunedByP7;
        }

        public static String getCsvHeader() {
            return "config,level,totalTimeMs,phase1TimeMs,phase2TimeMs,phase3TimeMs," +
                   "candidatesGenerated,candidatesExplored,totalPruned," +
                   "prunedByP1,prunedByP2,prunedByP3,prunedByP4,prunedByP5,prunedByP6,prunedByP7," +
                   "supportCalculations,closureChecks,tidsetIntersections," +
                   "cacheHits,cacheMisses,cacheSize,patternsFound";
        }

        public String toCsvRow() {
            return String.format("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                configName, pruningLevel, totalTimeMs, phase1TimeMs, phase2TimeMs, phase3TimeMs,
                candidatesGenerated, candidatesExplored, getTotalPruned(),
                prunedByP1, prunedByP2, prunedByP3, prunedByP4, prunedByP5, prunedByP6, prunedByP7,
                supportCalculations, closureChecks, tidsetIntersections,
                cacheHits, cacheMisses, cacheSize, patternsFound);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INSTANCE VARIABLES
    // ═══════════════════════════════════════════════════════════════════════════

    private final Vocabulary vocab;
    private final PruningConfig config;
    private final Metrics metrics;

    private TopKHeap topK;
    private PriorityQueue<FrequentItemset> pq;
    private Map<Itemset, CachedFrequentItemset> cache;
    private SupportCalculator calculator;
    private Itemset[] singletonCache;
    private int frequentItemCount;
    private int[] frequentItems;

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTRUCTORS
    // ═══════════════════════════════════════════════════════════════════════════

    public TUFCI_Pruning(UncertainDatabase database, double tau, int k, PruningConfig config) {
        super(database, tau, k);
        this.vocab = database.getVocabulary();
        this.config = config;
        this.metrics = new Metrics();
        this.metrics.configName = config.getName();
        this.metrics.pruningLevel = config.countEnabled();
        this.calculator = new DirectConvolutionSupportCalculator(tau);
        this.cache = new HashMap<>();
    }

    public Metrics getMetrics() { return metrics; }
    public PruningConfig getConfig() { return config; }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 1
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    protected List<FrequentItemset> computeAllSingletonSupports() {
        long startTime = System.nanoTime();
        int vocabSize = vocab.size();
        FrequentItemset[] resultArray = new FrequentItemset[vocabSize];
        ConcurrentHashMap<Itemset, CachedFrequentItemset> concurrentCache = new ConcurrentHashMap<>(vocabSize);

        this.singletonCache = new Itemset[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            singletonCache[i] = createSingletonItemset(i);
        }

        java.util.stream.IntStream.range(0, vocabSize).parallel().forEach(item -> {
            Itemset singleton = singletonCache[item];
            Tidset tidset = getDatabase().getTidset(singleton);
            if (tidset.isEmpty()) { resultArray[item] = null; return; }

            double[] supportResult = calculator.computeProbabilisticSupportFromTidset(tidset, getDatabase().size());
            int support = (int) supportResult[0];
            double probability = supportResult[1];

            resultArray[item] = new FrequentItemset(singleton, support, probability);
            concurrentCache.put(singleton, new CachedFrequentItemset(singleton, support, probability, tidset));
        });

        List<FrequentItemset> result = Arrays.stream(resultArray)
                .filter(Objects::nonNull)
                .sorted(FrequentItemset::compareBySupport)
                .collect(Collectors.toList());

        this.cache = concurrentCache;
        metrics.phase1TimeMs = (System.nanoTime() - startTime) / 1_000_000;
        metrics.supportCalculations += vocabSize;
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 2 - Contains P1
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    protected void initializeTopKWithClosedSingletons(List<FrequentItemset> frequent1Itemsets) {
        long startTime = System.nanoTime();
        this.topK = new TopKHeap(getK());
        this.pq = new PriorityQueue<>(FrequentItemset::compareBySupport);

        int minsup = 0;
        int processedItemCount = 0;
        int totalItems = frequent1Itemsets.size();

        for (int i = 0; i < totalItems; i++) {
            FrequentItemset fi = frequent1Itemsets.get(i);
            int support = fi.getSupport();

            // P1a: Phase 2 Early Termination
            if (config.P1 && topK.isFull() && support < minsup) {
                metrics.prunedByP1 += (totalItems - i);
                break;
            }

            processedItemCount++;
            boolean isClosed = checkClosure1Itemset(fi, support, frequent1Itemsets, minsup);
            metrics.closureChecks++;

            if (isClosed) {
                boolean inserted = topK.insert(fi);
                if (inserted && topK.isFull()) minsup = topK.getMinSupport();
            }
        }

        // Build frequent items array
        List<Integer> frequentItemIndices = new ArrayList<>();
        for (int i = 0; i < processedItemCount; i++) {
            FrequentItemset fi = frequent1Itemsets.get(i);
            if (fi.getSupport() >= minsup) frequentItemIndices.add(fi.getItems().get(0));
        }
        this.frequentItemCount = frequentItemIndices.size();
        this.frequentItems = new int[frequentItemCount];
        for (int i = 0; i < frequentItemCount; i++) frequentItems[i] = frequentItemIndices.get(i);

        // Seed with 2-itemsets
        for (Map.Entry<Itemset, CachedFrequentItemset> entry : cache.entrySet()) {
            Itemset itemset = entry.getKey();
            if (itemset.size() == 2) {
                CachedFrequentItemset cached = entry.getValue();
                // P1c: 2-Itemset Filtering
                if (config.P1 && cached.getSupport() < minsup) {
                    metrics.prunedByP1++;
                    continue;
                }
                pq.add(cached.toFrequentItemset());
                metrics.candidatesGenerated++;
            }
        }

        metrics.phase2TimeMs = (System.nanoTime() - startTime) / 1_000_000;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 3 - Contains P2
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    protected void executePhase3(List<FrequentItemset> frequent1itemsets) {
        long startTime = System.nanoTime();

        while (!pq.isEmpty()) {
            FrequentItemset candidate = pq.poll();
            metrics.candidatesExplored++;
            int threshold = getThreshold();

            if (candidate.getSupport() < threshold) {
                // P2b: Main Loop Early Termination
                if (config.P2) {
                    metrics.prunedByP2 += pq.size() + 1;
                    break;
                }
                metrics.prunedByP2++;
                continue;
            }

            ClosureCheckResult result = checkClosureAndGenerateExtensions(candidate, threshold);
            if (result.isClosed()) topK.insert(candidate);

            int newThreshold = getThreshold();
            for (FrequentItemset ext : result.getExtensions()) {
                metrics.candidatesGenerated++;
                // P2c: Extension Filtering
                if (config.P2 && ext.getSupport() < newThreshold) {
                    metrics.prunedByP2++;
                    continue;
                }
                pq.add(ext);
            }
        }

        metrics.phase3TimeMs = (System.nanoTime() - startTime) / 1_000_000;
        metrics.cacheSize = cache.size();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CLOSURE AND EXTENSIONS - Contains P1b, P3, P4, P5, P6, P7
    // ═══════════════════════════════════════════════════════════════════════════

    protected boolean checkClosure1Itemset(FrequentItemset oneItemFI, int supOneItem,
                                         List<FrequentItemset> frequent1Itemset, int minsup) {
        int itemA = oneItemFI.getItems().get(0);

        for (FrequentItemset otherFI : frequent1Itemset) {
            int itemB = otherFI.getItems().get(0);
            if (itemA == itemB) continue;

            // P1b: Closure Check Early Termination
            if (config.P1 && otherFI.getSupport() < supOneItem) break;

            Itemset unionItemset = oneItemFI.union(otherFI);
            CachedFrequentItemset cached = cache.get(unionItemset);
            int supAB;
            double probAB;
            Tidset tidsetAB;

            if (cached != null) {
                supAB = cached.getSupport();
                metrics.cacheHits++;
            } else {
                metrics.cacheMisses++;
                tidsetAB = cache.get(oneItemFI).getTidset().intersect(cache.get(otherFI).getTidset());
                metrics.tidsetIntersections++;

                if (!tidsetAB.isEmpty()) {
                    double[] result = calculator.computeProbabilisticSupportFromTidset(tidsetAB, getDatabase().size());
                    supAB = (int) result[0];
                    probAB = result[1];
                    metrics.supportCalculations++;
                } else { supAB = 0; probAB = 0.0; }

                if (otherFI.getSupport() >= minsup)
                    cache.put(unionItemset, new CachedFrequentItemset(unionItemset, supAB, probAB, tidsetAB));
            }

            if (supAB == supOneItem) return false;
        }
        return true;
    }

    protected ClosureCheckResult checkClosureAndGenerateExtensions(FrequentItemset candidate, int threshold) {
        int supX = candidate.getSupport();
        boolean isClosed = true;
        List<FrequentItemset> extensions = new ArrayList<>();
        int maxItemInX = getMaxItemIndex(candidate);
        boolean closureCheckingDone = false;

        for (int idx = 0; idx < frequentItemCount; idx++) {
            int item = frequentItems[idx];
            if (candidate.contains(item)) continue;
            int itemSupport = getItemSupport(item);

            // P3: Item Support Threshold
            if (config.P3 && itemSupport < threshold) {
                metrics.prunedByP3 += (frequentItemCount - idx - 1);
                break;
            }

            if (!closureCheckingDone && itemSupport < supX) closureCheckingDone = true;

            boolean needClosureCheck = !closureCheckingDone && isClosed;
            boolean needExtension = (item > maxItemInX);
            int upperBound = Math.min(supX, itemSupport);

            // P4: Subset Upper Bound Tightening
            if (config.P4 && topK.isFull() && needExtension) {
                for (int existingItem : candidate.getItemsArray()) {
                    Itemset twoItemset = Itemset.of(vocab, Math.min(existingItem, item), Math.max(existingItem, item));
                    CachedFrequentItemset cachedSubset = cache.get(twoItemset);
                    if (cachedSubset != null) {
                        int oldBound = upperBound;
                        upperBound = Math.min(upperBound, cachedSubset.getSupport());
                        if (upperBound < oldBound) metrics.prunedByP4++;
                        if (upperBound < threshold) break;
                    }
                }
            }

            boolean canEnterTopK = (upperBound >= threshold);
            boolean shouldGenerateExtension = needExtension && canEnterTopK;

            // P5: Upper Bound Filtering
            if (config.P5) {
                if (!needClosureCheck && !shouldGenerateExtension) { metrics.prunedByP5++; continue; }
            } else {
                if (!needClosureCheck && !needExtension) continue;
                shouldGenerateExtension = needExtension;
            }

            Itemset itemItemset = singletonCache[item];
            Itemset Xe = candidate.union(itemItemset);
            int supXe; double probXe; Tidset tidsetXe;

            CachedFrequentItemset cached = cache.get(Xe);
            if (cached != null) {
                supXe = cached.getSupport();
                probXe = cached.getProbability();
                tidsetXe = cached.getTidset();
                metrics.cacheHits++;
            } else {
                metrics.cacheMisses++;
                CachedFrequentItemset xInfo = cache.get(candidate);
                CachedFrequentItemset itemInfo = cache.get(itemItemset);

                if (xInfo == null || itemInfo == null) {
                    tidsetXe = getDatabase().getTidset(candidate).intersect(getDatabase().getTidset(itemItemset));
                } else {
                    tidsetXe = xInfo.getTidset().intersect(itemInfo.getTidset());
                }
                metrics.tidsetIntersections++;
                int tidsetSize = tidsetXe.size();

                // P6: Tidset Size Pruning
                if (config.P6 && tidsetSize < threshold && !needClosureCheck) {
                    cache.put(Xe, new CachedFrequentItemset(Xe, 0, 0.0, tidsetXe));
                    metrics.prunedByP6++;
                    continue;
                }

                // P7: Tidset Closure Skip
                if (config.P7 && needClosureCheck && tidsetSize < supX) {
                    if (!shouldGenerateExtension) {
                        cache.put(Xe, new CachedFrequentItemset(Xe, 0, 0.0, tidsetXe));
                        metrics.prunedByP7++;
                        continue;
                    }
                    needClosureCheck = false;
                    metrics.prunedByP7++;
                }

                double[] result = calculator.computeProbabilisticSupportFromTidset(tidsetXe, getDatabase().size());
                supXe = (int) result[0];
                probXe = result[1];
                metrics.supportCalculations++;
                cache.put(Xe, new CachedFrequentItemset(Xe, supXe, probXe, tidsetXe));
            }

            if (needClosureCheck) { metrics.closureChecks++; if (supXe == supX) isClosed = false; }
            if (shouldGenerateExtension) extensions.add(new FrequentItemset(Xe, supXe, probXe));
        }

        return new ClosureCheckResult(isClosed, extensions);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RESULTS
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    protected List<FrequentItemset> getTopKResults() {
        List<FrequentItemset> results = topK.getAll();
        results.sort(FrequentItemset::compareBySupport);
        metrics.patternsFound = results.size();
        metrics.totalTimeMs = metrics.phase1TimeMs + metrics.phase2TimeMs + metrics.phase3TimeMs;
        return results;
    }

    protected Itemset createSingletonItemset(int item) {
        Itemset itemset = new Itemset(vocab);
        itemset.add(item);
        return itemset;
    }
    protected int getThreshold() { return topK.getMinSupport(); }
    protected int getItemSupport(int item) {
        if (item < 0 || item >= singletonCache.length) return 0;
        CachedFrequentItemset cached = cache.get(singletonCache[item]);
        return (cached != null) ? cached.getSupport() : 0;
    }
    protected int getMaxItemIndex(Itemset itemset) {
        int[] items = itemset.getItemsArray();
        return items.length == 0 ? -1 : items[items.length - 1];
    }
}