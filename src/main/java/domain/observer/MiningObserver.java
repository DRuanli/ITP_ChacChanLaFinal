package domain.observer;

import domain.model.Pattern;

public interface MiningObserver {
    void onPhaseStart(int phase, String description);
    void onPhaseComplete(int phase, long durationMs);
}
