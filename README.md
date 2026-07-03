# Spot the Fake Photo — Note

## Approach
Classic computer-vision feature engineering (no deep net) feeding a RandomForest classifier. Features target the physical artifacts a screen recapture leaves behind: multi-patch FFT ring/off-axis energy (moiré from the pixel grid), chroma-channel decorrelation (color fringing), local sharpness/texture, color clipping and blue/green cast, glare/specular-highlight blobs, bezel/border edges (Hough lines), and denoised-residual noise structure. 29 features total, standardized and fed into a 500-tree RandomForest (`class_weight="balanced"`).

## Data
170 self-collected photos on one phone: 88 real, 82 screen/printout recaptures.

## Accuracy — honest number
**83.5%** on 5-fold stratified cross-validation (confusion matrix: 73/88 real correct, 69/82 screen correct). This is **below the 95% bar**, and I don't expect it to clear 95% on a fresh held-out set either: the data comes from a single device with limited screen/lighting variety, and the 28 CV misclassifications mostly land close to the 0.5 threshold (0.37–0.76) rather than being confidently wrong — that pattern points to genuine overlap between the two classes in feature space, not a broken pipeline.

## Latency
**823.8 ms mean / 823.7 ms median per image**, measured end-to-end via `predict.py` on a shared Google Colab CPU. Of that, ~597 ms is feature extraction (the FFT-heavy part); the rest is model load + inference, since `predict.py` loads `model.joblib` fresh on each call to match the `python predict.py image.jpg` CLI contract. In a persistent/warm process (model loaded once), expect closer to 600–650 ms — still feature-extraction bound, and well short of feeling "instant."

## Cost per image
- **On-device:** effectively $0 marginal cost — runs on the user's phone, no server round-trip.
- **Cloud (rough estimate):** assuming a small single-vCPU instance at ~$0.05/hr and ~650 ms warm CPU time per image → ~5,540 images/hour/core → **~$0.01 per 1,000 images (~$9 per million)**, scaling roughly linearly with added cores/machines. Assumptions: single-threaded, no batching, no GPU.

## What I'd improve with more time
1. Collect more diverse data — multiple phones, more screen/printout types, more lighting and angles. This is the most likely fix for the accuracy gap.
2. Try simpler/regularized models (logistic regression, fewer/shallower trees, feature selection). Given the borderline nature of the misclassifications, more model capacity isn't obviously the answer.
3. Speed up feature extraction: cap working resolution lower, use fewer/smaller FFT patches, drop the costlier ops (Hough lines).
4. Keep the model loaded across calls instead of reloading per-invocation, once this moves off the one-shot CLI contract.
5. Tune the decision threshold from a validation ROC/PR curve instead of the default 0.5.

## More experienced notes
- **Keeping it accurate as cheaters adapt:** log low-confidence and false-negative cases in production and periodically retrain on newly collected recaptures — an active-learning loop that tracks new screen tech and evasion tricks (tighter crops to hide bezels, higher-PPI displays with less moiré).
- **Making it phone-sized:** lower the working resolution and prune the most expensive features first; a small quantized on-device model (TFLite/Core ML) is likely an easier path to sub-50ms than continuing to optimize the OpenCV+sklearn pipeline.
- **Choosing the cutoff:** pick it from a validation ROC/PR curve based on the real cost asymmetry — wrongly flagging a genuine user is probably more costly than missing one cheat — and consider a two-threshold band (auto-accept / manual review / auto-flag) instead of a single global cutoff.
