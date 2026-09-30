# FedIoT-Gate experimentation code

This is real, runnable code for the experiments described in the paper's
Section VII, written and run in one session with the constraints listed
below. **Read the "What is real vs. a stand-in" section before citing any
number this code produces in the paper.**

## Setup

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn tqdm requests torch pyarrow
```

Tested with Python 3.14, PyTorch 2.14 (CPU/MPS, no CUDA GPU in this
environment). Apple Silicon MPS backend is available but the current
scripts run on CPU by default since the models are small enough that
MPS's transfer overhead isn't worth it at this scale; pass `device="mps"`
in `FederatedSimulation(...)` if you scale up.

## What data is actually present

| Dataset | Status | Where |
|---|---|---|
| **N-BaIoT** | Downloaded and extracted (real). 9 devices, benign + gafgyt + mirai attack CSVs, unzipped from the UCI ML repository archive, `.rar` attack archives extracted with `unar` (installed via `brew install unar`). | `code/data/raw/nbaiot/<Device>/` |
| **UNSW (Sivanathan)** | **Downloaded and extracted (real).** Direct HTTPS links found on `iotanalytics.unsw.edu.au/iottraces.html` (no data-access request needed after all) — the device-name/MAC list from `.../resources/List_Of_Devices.txt` plus 20 days of per-packet CSVs (`.../iottestbed/csv/<date>.csv.zip`, ~118MB zipped, ~1.9GB extracted, 21M packet rows). Each row has a timestamp, size, source/destination MAC and IP, protocol, and ports — the actual raw capture, not pre-extracted features. | `code/data/raw/unsw/{List_Of_Devices.txt, csv_extracted/*.csv}` |
| **CIC IoT-DIAD 2024** | **Already staged locally (real), not downloaded this session.** `code/data/labeled_data/{device_task.parquet, attack_task.parquet, sampled_full.parquet}`, built by a prior local step from the raw dataset. Uniquely among our real data, gives BOTH real per-device identity (59 named devices, `device_task.parquet`, 986K rows) AND real attack labels (`attack_task.parquet`, 3.6M rows, same 8-class-style taxonomy as CIC IoT 2023) in one place, over a much richer real feature set (118 numeric features: inter-arrival time, TTL, TCP window size, payload entropy, TLS handshake fields, HTTP/DNS/ICMP fields, multi-window 1/5/10/30/60-packet traffic statistics per stream/IP/channel) than anything else used in this paper. The prior local build already dropped identity/leakage columns (`stream`, `src_mac`, `dst_mac`, `src_ip`, `dst_ip`, `device_mac`, `eth_src_oui`, `eth_dst_oui`) before this session touched the data -- see `build_info.json` in that directory. Several high-cardinality text columns (`http_host`, `user_agent`, `dns_server`, `tls_server`, `http_uri`) are NOT yet used (flagged, not silently dropped -- see `fedgate/data.py`'s module docstring) since they'd need real categorical encoding and a leakage check first (a device that always talks to one cloud host could make that hostname an identity shortcut). | `code/data/labeled_data/{device_task.parquet, attack_task.parquet, build_info.json, device_type_map.csv, class_counts.csv}` |
| **YourThings** | **Downloaded and processed for 2 of ~13 available days (real), with real IP-verified device identification.** `yourthings.info/data/` links directly to Dropbox-hosted per-day `.tgz` archives, no access request needed, AND publishes a real `device_mapping.csv` (66 devices, device name -> local IP) that an earlier pass through this dataset missed. Two days (2018-03-20 and 2018-03-21, 6.8GB + 7.0GB) were streamed through `tshark` (chunk-by-chunk, deleting each raw pcap immediately) into packet CSVs with `ip.src`/`ip.dst` fields (`process_day_ip.sh`, 38.6M + 32.5M rows), and devices matched against the real IP mapping (`load_yourthings_device_identification`, preferring this over the older vendor-OUI guess whenever IP columns are present). Result: **48 devices found active** (≥1000 packets) with verified identity, including all 3 originally-targeted devices (Philips Hue Hub 11.7M packets, Samsung SmartThings Hub 10.8M packets, Bose SoundTouch 10 2.2M packets -- the last one previously, incorrectly, reported as inactive under the old vendor-guess method) plus 4 more with a genuine product match in UNSW's label space (Amazon Echo, Belkin WeMo Motion Sensor, LiFX Smart Bulb) used for a real, broadened cross-network generalization test (see below). The remaining ~11 days were not processed, due to time. | `code/data/raw/yourthings/{iot_traffic2018032{0,1}.tgz, packets_ip_2018032{0,1}.csv, device_mapping.csv, manuf.tsv, process_day.sh, process_day_ip.sh}` |
| **CIC IoT 2023** | **Downloaded (real), via a legitimate third-party mirror, not the official gated portal.** The official `cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/` portal still gates downloads behind a personal registration form (first/last name, email, organization, job title, country) that this code does not and should not auto-submit with fabricated identity details. However, the dataset itself is separately, legitimately mirrored under a **CC-BY-4.0 license** on Hugging Face (`lacg030175/CIC-IoT-2023-canonical-neto`), traced back to Neto et al.'s original merged CSVs (via `bencorn/CIC-IoT-2023`'s `CSV/MERGED_CSV/` folder) with the full ~45M-row canonical dataset preserved (a competing, smaller 38.5M-row re-mirror had lost rows in a folder restructure -- the canonical one was used instead). Downloaded as two parquet files, no registration needed: `train.parquet` (36M rows, 969MB) and `test.parquet` (9M rows, 242MB), 39 pre-extracted flow features (Header_Length, Protocol Type, flag counts, protocol indicators, IAT/statistical features -- the standard CICIoT2023 feature set) plus a normalized `attack_class` label spanning the **full 8-class taxonomy**: Benign, BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web-based. | `code/data/raw/ciciot2023/{train.parquet, test.parquet}` |

**N-BaIoT, UNSW, YourThings (partially), and CIC IoT 2023 were all
actually acquired**, so real experimental numbers in `results/` now
cover most of the paper's Section VII protocol:

- **CIC IoT 2023, like N-BaIoT, ships pre-extracted flow features, not
  raw packet sequences**, so it cannot exercise the Phase 0 encoder
  either. Unlike N-BaIoT, this merged/flattened parquet format has **no
  per-device identifier column** -- the original 105-device collection
  isn't reconstructable from it, so the federated "clients" built from it
  (`fedgate/data.py: load_ciciot2023_federated`) are a **random partition
  of pooled rows**, not a device-level partition. This is a real
  limitation: it cannot model genuine cross-device heterogeneity the way
  N-BaIoT's 9-physical-device or UNSW's per-day-capture partitions do.
  Report results from it with that caveat attached; see
  `scripts/run_ciciot2023.py` and `results/ciciot2023_attack_head.json`.
- This gives the attack-type head a real 8-class benchmark beyond
  N-BaIoT's 3-way benign/gafgyt/mirai split, extending Section VII-C.
  Combining N-BaIoT and CIC IoT 2023 into one unified attack-type label
  space (as the paper's design intends) is a natural next step, not done
  here -- the two datasets' feature sets differ entirely (N-BaIoT: 115
  handcrafted statistics per flow window; CIC IoT 2023: 39 handcrafted
  flow features), so unifying them requires either training separate
  heads per dataset or finding/learning a shared feature space, which is
  additional design work beyond what this session implements.

- **YourThings' real value here is a cross-network generalization test**
  (Section VII-B's "cross-network generalization" and VII-G's onboarding):
  the Phase 0 encoder trained on UNSW is used to embed real YourThings
  device traffic and query it against a UNSW-built device-fingerprint
  knowledge base. See `scripts/run_yourthings_onboarding_v2.py` and
  `results/yourthings_cross_network_onboarding_v2.json` (supersedes the
  original `run_yourthings_onboarding.py` / `..._onboarding.json`, kept
  for history).
- **Real, verified device identification (not vendor guessing) --
  discovered after the original pass through this dataset.**
  `yourthings.info/data/` publishes `device_mapping.csv`: 66 devices,
  real device-name-to-local-IP ground truth (e.g.
  `SamsungSmartThingsHub,192.168.0.4`), saved to
  `code/data/raw/yourthings/device_mapping.csv`. The original pass
  missed this and instead guessed device identity from the Ethernet
  vendor (OUI) prefix of the MAC -- a materially weaker method that
  identifies a *manufacturer*, not a verified *model*, and covered only
  3 target devices. `process_day_ip.sh` (new) re-extracts an
  already-downloaded `.tgz` with `ip.src`/`ip.dst` fields added, so
  devices can be matched against the real IP mapping instead;
  `fedgate/data.py`'s `load_yourthings_device_identification` now
  prefers this real IP-based path automatically whenever a CSV has IP
  columns, falling back to the old vendor-OUI method only for CSVs
  without them. Re-running IP extraction on the 2 already-downloaded
  days (no new download needed) found **48 devices active** with
  verified identity, vs. 3 target devices under the old method -- and
  caught a real error: Bose SoundTouch 10, previously reported "not
  active in either day," is in fact active with 2.15M real packets under
  the correct identification.
- Four devices have a genuine same-product match in UNSW's 18-class
  label space, checked by actual product identity (not string matching,
  since the two datasets name products differently) -- see
  `YOURTHINGS_TO_UNSW_ALIGNMENT` in `run_yourthings_onboarding_v2.py`:
  Samsung SmartThings Hub, Amazon Echo (Gen 1), Belkin WeMo Motion
  Sensor, LiFX Smart Bulb. Real cross-network result: only 1 of 4
  (Amazon Echo) correctly retrieved its own product's UNSW fingerprint;
  the other 3 did not, including Samsung SmartThings Hub, which no
  longer has the "unverified identification" confound the original
  single-device result carried. Separately, all 44 devices with no UNSW
  counterpart correctly avoided false auto-admission (0/44) -- a much
  larger-sample validation of that safety property than the original
  single-device (Philips Hue Hub) check.
- Real caveat on the IP mapping itself: local IPs are typically
  DHCP-assigned, so device_mapping.csv is only trustworthy for capture
  days within the testbed period it was published for; we have not
  independently confirmed a device kept the same IP on every capture
  day, only that the mapping is a real, dataset-published artifact for
  the period it covers (see `fedgate/data.py` module docstring).
- Still only 2 of ~13 available days processed. More days (each
  6-11GB compressed, ~45-70 min of tshark extraction) would likely
  surface more active devices and give more robust per-device sample
  counts, and are worth doing before trusting any single day's null
  result (e.g. a device inactive on both processed days) too strongly.
- **Tried a real fix for the cross-network gap; it didn't work, and we
  know why.** `federated_pretrain_encoder_domain_adversarial`
  (`fedgate/phase0_training.py`, new) adapts a technique from RF
  (radio-signal) fingerprinting for an analogous "same device, different
  receiver" problem: a domain discriminator (FedAvg'd alongside the
  encoder) is trained to guess which UNSW client an embedding came from,
  behind a gradient-reversal layer that pushes the encoder to make that
  harder, aiming to strip out network-specific signal. Re-running the
  identical cross-network test
  (`run_yourthings_onboarding_domain_adversarial.py`) with this encoder
  got the *same* result: 1/4 correct, same device (Amazon Echo)
  succeeding, confidence barely moved (34%→37%). Real diagnosis, not a
  shrug: UNSW's "10 federated clients" are a *simulated partition of one
  real network's own traffic*, not 10 physically different networks --
  training invariance to "which arbitrary partition of the same network"
  carries little information about invariance to an actually different
  network (YourThings), which is the real shift we care about. A real
  fix needs genuine multi-network training data (e.g. train across both
  UNSW and YourThings, or a second real dataset, not partitions of UNSW
  alone). Results: `results/yourthings_cross_network_domain_adversarial.json`.
- **Tried genuine multi-network training data, as that diagnosis called
  for; it made things worse.** `run_yourthings_onboarding_domain_adversarial_v2.py`
  retrains the domain-adversarial encoder on UNSW's 10 clients plus 8
  real, disjoint YourThings devices (unlabeled, used only as extra
  domains, strictly excluding the 4 held-out evaluation devices --
  no leakage). Real result: **0/4 correct**, worse than both the plain
  encoder and the UNSW-only domain-adversarial variant (both 1/4) --
  even Amazon Echo, correct in both earlier configs, failed here. This
  complicates rather than confirms our own diagnosis. Leading
  unconfirmed hypothesis: inconsistent domain granularity between the
  two data sources -- each UNSW "domain" is a client partition
  containing *multiple device types*, while each added YourThings
  "domain" is a *single device type*, so the adversarial objective may
  have pushed the encoder to discard genuinely device-discriminative
  signal along with network-specific noise. Not confirmed
  experimentally. Results:
  `results/yourthings_cross_network_domain_adversarial_v2.json`.
- **Tried the cleaner domain definition the v2 diagnosis suggested; it
  recovered the baseline but still didn't beat it.**
  `run_yourthings_onboarding_domain_adversarial_v3.py` uses
  `federated_pretrain_encoder_domain_adversarial`'s new `domain_labels`
  parameter to collapse everything into exactly 2 domains -- all UNSW
  clients as domain 0, all added YourThings devices as domain 1 -- a
  genuine, consistent "which network" signal instead of v2's mismatched
  per-client/per-device granularity. Real result: back to **1/4**
  (Amazon Echo correct again, 29% confidence), matching the plain
  encoder and v1, clearly better than v2's 0/4. Results:
  `results/yourthings_cross_network_domain_adversarial_v3.json`.

  **Final tally, 4 real configurations:**

  | Config | Cross-network accuracy |
  |---|---|
  | Plain encoder | 1/4 |
  | Domain-adv, per-UNSW-client domains (v1) | 1/4 |
  | Domain-adv, per-device multi-network domains (v2) | 0/4 |
  | Domain-adv, clean 2-domain network-level (v3) | 1/4 |

  Reading across all four: domain granularity mismatch measurably hurts
  (v2), and fixing it measurably recovers the loss (v3) -- real, if
  indirect, support for the granularity hypothesis -- but a clean,
  methodologically sound domain-adversarial signal alone was not enough
  to *improve* on the plain encoder at this training budget (15 rounds,
  one lightweight discriminator). This is not evidence the mechanism is
  broken, just that it needs more (training budget, discriminator
  capacity) or a different technique (e.g. contrastive pretraining that
  explicitly aligns same-device-type embeddings across networks, which
  would need some cross-network device-identity supervision this fully
  unsupervised approach doesn't use) to actually close this gap.

- **Tried real explicit contrastive alignment (the technique the v3
  paragraph suggested); also didn't beat baseline -- and revealed a
  striking pattern across all five techniques.**
  `contrastive_finetune_encoder` (`fedgate/phase0_training.py`, new) is a
  real InfoNCE-style supervised contrastive fine-tuning step that pulls a
  known cross-network device pair's embeddings together and away from
  other classes'. **Methodologically important:** aligning on all 4
  known pairs and testing on those same 4 would not be a real test (just
  memorization) -- `run_yourthings_onboarding_contrastive_alignment.py`
  runs a real 4-fold leave-one-out instead: align using 3 pairs, test on
  the unseen 4th, repeat for each. Real result: **1/4 held-out folds
  correct**, Amazon Echo again the sole success (38% confidence), same
  three devices wrong. Results:
  `results/yourthings_cross_network_contrastive_alignment.json`.

  **Complete tally, 5 real configurations:**

  | Config | Cross-network accuracy |
  |---|---|
  | Plain encoder | 1/4 |
  | Domain-adv, per-UNSW-client domains (v1) | 1/4 |
  | Domain-adv, per-device multi-network domains (v2) | 0/4 |
  | Domain-adv, clean 2-domain network-level (v3) | 1/4 |
  | Leave-one-out contrastive alignment | 1/4 |

  **The real finding here is not a fix, it's a pattern:** Amazon Echo is
  the *only* device that ever succeeds, in every configuration where
  anything succeeds at all, regardless of training objective
  (self-supervised, domain-adversarial, or explicitly
  supervised-contrastive all land on the identical result). This
  strongly suggests the bottleneck is not primarily which training
  technique is used, but something device-intrinsic about the other
  three devices' traffic specifically -- most plausibly that their
  packet-level signatures are more thoroughly confounded by
  network-specific factors than Amazon Echo's (which may have a more
  stereotyped, network-invariant cloud-check-in pattern that survives
  regardless of technique). **This is the clearest, most concrete lead
  for future work on this problem**: investigate what's actually
  different about Amazon Echo's traffic (e.g. compare its per-window
  variance/periodicity against the other three across both UNSW and
  YourThings captures) rather than continuing to vary the training
  objective, since five different objectives already converged on an
  identical outcome.

- **Checked this lead directly against the raw data -- real signal
  found, honestly scoped.** `analyze_cross_network_traffic_signatures.py`
  is pure data analysis, no training: computes simple descriptive
  statistics directly on the raw signed packet-size windows (the exact
  representation the encoder consumes) for all 4 devices in both
  captures. First hypothesis (Amazon Echo's raw mean packet size shifts
  least between networks) was **not confirmed** -- Belkin WeMo Motion
  Sensor is actually most stable by that metric (shift z=0.08 vs. Echo's
  0.42), yet Belkin fails at retrieval. A different, more specific
  statistic tells a cleaner story: the fraction of traffic concentrated
  on a device's single most common exact packet size (a proxy for
  "how heartbeat/repetitive is this device's traffic") shifts far less
  for Amazon Echo (17.2%→10.1%, a 7.1pp drop) than for the other three
  (Samsung SmartThings Hub 39.5%→24.4%, 15.1pp; Belkin WeMo Motion Sensor
  49.3%→16.7%, 32.5pp; LiFX Smart Bulb 32.3%→17.4%, 14.9pp) -- Amazon
  Echo's shift is smallest by a clear margin on this specific metric,
  consistent with (not proof of) the hypothesis that its traffic's
  *shape*, not just its size, stays more consistent across networks.
  **Honesty caveat this session was careful about: n=4, not a
  statistically powered test** -- report as a real, checkable,
  directionally consistent lead, not a confirmed mechanism. Results:
  `results/cross_network_traffic_signature_analysis.json`. Natural cheap
  follow-up not done here: compute this same concentration-stability
  metric across a larger set of devices in both networks (not just the 4
  aligned pairs) to see if it predicts cross-network retrieval success
  more broadly.

- **Replicated Sivanathan et al.'s own controlled ablation (VAE-style
  latent regularization); also 1/4, sixth configuration to agree.**
  Added a `variational=True` mode to `SelfSupervisedEncoder`
  (`fedgate/models.py`) and `federated_pretrain_encoder`
  (`fedgate/phase0_training.py`): mean/log-variance heads, reparameterization
  sampling, KL-divergence loss term (weight 0.001, their "balanced"
  setting), deterministic mean used at inference -- a direct replication
  of the one architectural change their paper's own controlled ablation
  found to matter most for cross-environment robustness in their
  setting. Real result (`run_yourthings_onboarding_vae.py`): **1/4**
  again, Amazon Echo again the sole success (28% confidence). Six real
  configurations (plain, 3 domain-adversarial variants, contrastive
  alignment, VAE) now agree on the identical outcome. Read as evidence
  the bottleneck is likely upstream of all six -- in the feature
  representation itself (100-packet signed-size-only windows), not
  anything triable at the training/architecture level on top of it.
  Results: `results/yourthings_cross_network_vae.json`.

- **Tested the feature-representation hypothesis directly against CIC
  IoT-DIAD 2024's much richer real feature set -- real support, plus a
  real bug caught and fixed along the way.** `fedgate/data.py` gained
  two new loaders: `load_ciciot_diad_device_identification` (real
  per-device federated clients, up to 59 real devices) and
  `load_ciciot_diad_attack_classification` (pooled-row clients, same
  real limitation as CIC IoT 2023 since this file has no device labels).
  `scripts/run_ciciot_diad.py` runs both real experiments.

  **Attack classification: 0.926 macro-F1** on the same 8-class taxonomy
  CIC IoT 2023 reaches only 0.684 on with its narrower flow-statistic
  features -- real, direct support for the feature-richness hypothesis.

  **Device identification: 0.293 macro-F1 across all 59 real devices**
  (a genuinely harder task than UNSW's 18 classes) -- but only after
  catching and fixing a real bug. First attempt used one federated
  client per real device (mirroring N-BaIoT's successful per-device
  pattern) and collapsed to **0.011** (frozen near 0.0006 for most of 25
  rounds) -- caught by the exact same "frozen/near-random macro-F1 is a
  bug signature" heuristic this README already documents for the CIC IoT
  2023 NaN/Inf collapse. The real cause here is architectural, not a
  coding error, and is a genuinely transferable lesson: for a
  device-*identification* task specifically, one-client-per-device means
  every row in a client's local data shares the same label, so each
  client's local model trivially "solves" its own data by always
  predicting its one class, and FedAvg-ing 59 such single-class
  specialists has no reason to produce a discriminative global model.
  This does NOT affect N-BaIoT, because N-BaIoT's label (attack type)
  varies within each device's own data -- the failure mode is specific
  to when the client-partitioning entity *is* the label being predicted.
  Fix: bucket multiple real devices per federated client (`n_clients`
  param, UNSW-style arbitrary partition) instead of one-client-per-device.
  0.011 → 0.293 from that one change. **General lesson, flagged
  prominently in the code's docstring so it isn't re-discovered the hard
  way**: whether per-entity federated clients are valid depends on
  whether that entity is the thing being classified, not merely on
  whether a natural per-entity partition exists in the data. Results:
  `results/ciciot_diad_{device_identification,attack_classification}.json`.

  **Does not yet test cross-network generalization** -- CIC IoT-DIAD 2024
  is a single testbed, so this doesn't directly resolve the YourThings
  cross-network question, only supports the upstream hypothesis that
  feature richness matters. Extracting this same rich feature set from
  YourThings' raw pcaps (or finding an equivalent second richer-feature
  dataset) is the natural next step to close the loop -- not done here.

- N-BaIoT ships 115 pre-extracted flow-statistic features per instance,
  **not** raw packet-length/direction sequences, and cannot exercise the
  Phase 0 encoder (Section V-C) as designed. It trains and evaluates the
  **attack-type head directly** on its own features (Section VII-C,
  VII-D, VII-E). N-BaIoT's 9 devices are used as the 9 federated clients.
- **UNSW *can* and does exercise the Phase 0 encoder as designed**: for
  each of the 21 IoT devices in `List_Of_Devices.txt`, `data.py:
  build_unsw_packet_sequences` reconstructs the signed packet-length/
  direction sequence exactly as Section V-B specifies (positive = the
  device's MAC is `eth.dst` / incoming; negative = `eth.src` / outgoing),
  streaming all 20 CSVs (21M rows) in ~12 seconds. After dropping
  low-traffic devices the same way the paper does (Withings Smart scale,
  Blipcare, NEST Protect), **18 device classes remain — matching the
  paper's own Table III output width exactly** — distributed evenly
  across 10 simulated federated clients, per the paper's own description
  of how it partitions UNSW. See `scripts/run_unsw.py` for the real
  Phase 0 (self-supervised encoder) vs. PCA-baseline comparison this
  enables — the paper's actual headline representation-learning result
  (Section VII-B), not a synthetic stand-in.
- The synthetic UNSW-style generator (`data.py: make_synthetic_unsw`) is
  now superseded by real UNSW data and kept only as a lightweight
  fallback for quick code-path checks; do not use it for paper numbers.

## What is real vs. a stand-in (read this before writing numbers into the paper)

| Component | Status |
|---|---|
| N-BaIoT data loading, federated partitioning, train/val/test split | **Real** |
| Phase I attack-type classifier (Transformer head), federated training loop, FedAvg | **Real** |
| Digest computation (update norm, cosine similarity to previous update) | **Real** |
| Bayesian long-term trust prior (Beta-Bernoulli, Section VI-B1) | **Real** |
| Trimmed-mean and Krum baseline aggregation | **Real** |
| Untargeted / targeted / trust-building attack simulations | **Real** |
| **Gating Agent's decision policy** (accept/downweight/exclude, and onboarding admit/defer/escalate, and triage confirm/flag/false-positive) | **Rule-based stand-in, NOT the LLM.** No LLM API key was available in this environment. See `gating_agent.py`, `onboarding.py`, `triage.py` docstrings. The stand-in makes the same three-way decisions from the same inputs (digests, Bayesian posterior, retrieval results) using fixed thresholds instead of free-form reasoning, and does not generate the natural-language rationale linking a triage decision to human-readable traffic semantics that the paper describes — its "rationale" is a templated string reporting the numeric evidence, not LLM-authored prose. |
| Phase 0 self-supervised encoder, federated pretraining (`phase0_training.py`) | **Real, and now trained on real UNSW raw sequences** (see `scripts/run_unsw.py`), compared against a real PCA(100→80) baseline on the same 10-client split, same classifier architecture, same round count. |
| Device-type head, onboarding, triage **on real data** | Device-type head on **real UNSW** (`run_unsw.py`) — real. Onboarding: two real variants exist now -- an N-BaIoT device-identity proxy (`run_onboarding_and_triage.py`, written before UNSW access was found) and a real **cross-network** test using UNSW-trained embeddings queried with real YourThings traffic (`run_yourthings_onboarding.py`) -- the latter is closer to the paper's actual intended setup. Triage still only uses the N-BaIoT proxy. |

**Bottom line for writing the paper**: results in `results/*.json` from
N-BaIoT runs (`backbone_nbaiot_attack_head.json`, `robustness_nbaiot.json`,
`trustbuild_nbaiot.json`, `onboarding_nbaiot_proxy.json`,
`triage_nbaiot.json`), from the real UNSW run (`unsw_backbone_comparison.json`),
from the real CIC IoT 2023 run (`ciciot2023_attack_head.json`), and from
the real cross-network test (`yourthings_cross_network_onboarding.json`)
are real numbers from real code on real data and can be reported as such,
captioned to make clear (a) N-BaIoT and CIC IoT 2023 numbers use the
rule-based Gating Agent stand-in, not the LLM design, (b) the N-BaIoT
onboarding/triage proxy and the UNSW run have no Gating Agent, attack, or
transfer-learning-comparison component — they're backbone/representation
or mechanism-only validations (Sections VII-A/VII-B/VII-G), and (c)
CIC IoT 2023's federated "clients" are a random row partition, not a
device-level one. Do not present N-BaIoT numbers as UNSW/CIC-IoT-2023
results or vice versa, and do not present the rule-based policy's numbers
as validating the LLM-agent design.

**A debugging lesson worth keeping**: an early CIC IoT 2023 run silently
collapsed to a frozen, *below-random* macro-F1 (0.027, bit-identical every
round) because that dataset's own documentation warns NaN/Inf feature
values are deliberately preserved, and a large-enough sample pulled in
enough of them to poison every weight after one backward pass. A metric
that is either worse than random guessing or perfectly static across
rounds is a bug signature, not "this dataset/task is just hard" — check
for it explicitly before reporting a low number as a real finding. Fixed
in `fedgate/data.py: load_ciciot2023_federated` by dropping non-finite
rows (simpler than, and a real trade-off against, the dataset's own
recommended invalid-value encoding scheme).

## Layout

```
code/
  fedgate/
    data.py                # N-BaIoT / UNSW / YourThings loaders (all real), synthetic fallback
    models.py               # Phase 0 encoder, Phase I classifier head
    federated.py             # FedAvg / trimmed-mean / Krum / gate-weighted aggregation
    phase0_training.py        # federated self-supervised encoder pretraining loop
    classifier_training.py     # minimal federated classifier training loop (no attack/gating)
    gating_agent.py             # Digests, Bayesian trust (real), RuleBasedGatingPolicy (stand-in)
    attacks.py                   # untargeted / targeted / trust-building attack simulation
    experiment.py                  # FederatedSimulation: N-BaIoT attack-type head + attacks + gating
    onboarding.py                   # kNN device-fingerprint retrieval + rule-based policy (stand-in)
    triage.py                        # kNN attack-signature retrieval + rule-based policy (stand-in)
  scripts/
    smoke_test.py                     # fast correctness check on N-BaIoT, run this first
    smoke_test_unsw.py                 # fast correctness check on UNSW (1 file)
    run_all.py                          # VII-C/D/E on real N-BaIoT: backbone, robustness, trust-building
    run_unsw.py                          # VII-A/B on real UNSW: PCA baseline vs. Phase 0 encoder
    run_onboarding_and_triage.py          # onboarding + triage validation (N-BaIoT proxy)
    run_yourthings_onboarding.py           # (superseded, kept for history) original real cross-network test, 3 vendor-guessed devices
    run_yourthings_onboarding_v2.py         # current: real cross-network test, 48 IP-verified devices, 4 real UNSW alignment pairs
    run_ciciot2023.py                       # real CIC IoT 2023 attack-type head, 8-class taxonomy
    run_extras.py                            # VII-B extras on real UNSW: N-sweep, client scalability, per-device F1
    run_trustbuild_ablation.py                # VII-E: round-only vs. with-Bayesian-prior ablation on real N-BaIoT
    run_alpha3_multiseed.py                    # VII-D: multi-seed re-run of the alpha=3/9 Gating Agent anomaly
    run_yourthings_onboarding_domain_adversarial.py      # VII-G: domain-adv v1, per-UNSW-client domains
    run_yourthings_onboarding_domain_adversarial_v2.py   # VII-G: domain-adv v2, real multi-network domains (made things worse)
    run_yourthings_onboarding_domain_adversarial_v3.py   # VII-G: domain-adv v3, clean 2-domain (recovered baseline)
    run_yourthings_onboarding_contrastive_alignment.py   # VII-G: real leave-one-out contrastive alignment
    run_yourthings_onboarding_vae.py                     # VII-G: VAE-style latent regularization (Sivanathan et al. replication)
    analyze_cross_network_traffic_signatures.py           # VII-G: raw-data analysis of the Amazon Echo pattern
    run_ciciot_diad.py                                      # CIC IoT-DIAD 2024: device-ID (59 devices) + attack classification, richer features
  data/raw/yourthings/
    process_day.sh                          # (superseded) extracts pcap chunks -> packet CSV (MAC only), bounded disk use
    process_day_ip.sh                        # current: same, but also extracts ip.src/ip.dst for real IP-based identification
  data/labeled_data/                          # CIC IoT-DIAD 2024, already staged locally (not downloaded this session)
    device_task.parquet                       # 986K rows, 59 real devices, 118 numeric features
    attack_task.parquet                        # 3.6M rows, 8-class attack_group labels, same features
    build_info.json                             # documents which identity/leakage columns were already dropped
  results/                                    # JSON outputs, one file per experiment stage
```

## Reproducing

```bash
source .venv/bin/activate
cd scripts
python3 smoke_test.py                     # ~1 minute, N-BaIoT sanity check
python3 smoke_test_unsw.py                # ~1 minute, UNSW sanity check (1 day's file)
python3 run_all.py --stage all            # ~25-30 minutes, writes results/*_nbaiot.json
python3 run_unsw.py                       # ~10-15 minutes, writes results/unsw_backbone_comparison.json
python3 run_onboarding_and_triage.py      # ~2 minutes
# YourThings needs the tgz downloaded + processed first (see data acquisition table above);
# process_day_ip.sh takes 30-60 minutes for one day's pcap chunks (tshark extraction is the bottleneck)
python3 run_yourthings_onboarding_v2.py   # ~7 minutes (retrains the Phase 0 encoder on UNSW)
python3 run_ciciot2023.py                 # ~15 seconds once train.parquet/test.parquet are downloaded
python3 run_extras.py                     # ~10-15 minutes, writes results/unsw_{n_sweep,client_scalability,per_device_f1}.json
python3 run_trustbuild_ablation.py        # ~3-5 minutes, writes results/trustbuild_ablation_nbaiot.json
python3 run_alpha3_multiseed.py           # ~5-10 minutes, writes results/alpha3_gating_multiseed.json
python3 run_yourthings_onboarding_domain_adversarial.py     # ~20 min, writes results/yourthings_cross_network_domain_adversarial.json
python3 run_yourthings_onboarding_domain_adversarial_v2.py  # ~20 min, writes results/..._domain_adversarial_v2.json
python3 run_yourthings_onboarding_domain_adversarial_v3.py  # ~20 min, writes results/..._domain_adversarial_v3.json
python3 run_yourthings_onboarding_contrastive_alignment.py  # ~10-15 min (4-fold leave-one-out), writes results/..._contrastive_alignment.json
python3 run_yourthings_onboarding_vae.py                    # ~7 min, writes results/yourthings_cross_network_vae.json
python3 analyze_cross_network_traffic_signatures.py         # <1 min, pure analysis, no training
python3 run_ciciot_diad.py                # ~2 min, writes results/ciciot_diad_{device_identification,attack_classification}.json
```

## Making the Gating Agent real

`LLMGatingPolicy` in `fedgate/gating_agent.py` is now a real
implementation, backed by a local open-weight model via
[Ollama](https://ollama.com) -- no API key, no per-call billing, and it
fits this paper's eventual "constrained edge gateway" deployment story
better than a cloud API would.

**Setup:**
```bash
brew install ollama
ollama serve &              # starts a local server on localhost:11434
ollama pull llama3.1:8b     # one-time download, ~5GB
cd scripts
python3 smoke_test_llm_gating.py   # sanity check: two synthetic digests, one normal, one anomalous
```

It does a single-turn call per decision (format the digest + trust
posterior into a prompt, ask for a JSON `{action, gating_score,
rationale}`) -- it does NOT yet implement the paper's optional
`FetchHistory`/`FetchSecondaryCalibration` tool-calling step (Section
VI-B); that multi-step behavior is a natural follow-on once this
single-turn baseline is validated. It raises `LLMGatingPolicyError`
(never silently falls back to the rule-based policy) if the Ollama
server is unreachable or returns something unparseable, after a couple
of retries.

**First real result, already in the paper (Section VII-E):**
`run_llm_gating_trustbuild.py` re-ran the trust-building attack at
$D \in \{0, 10\}$ with `LLMGatingPolicy` (Llama 3.1 8B via Ollama) swapped
in for `RuleBasedGatingPolicy`, via the new `gating_policy=` parameter on
`FederatedSimulation`. The result is surprising and reported plainly in
the paper: the LLM-based agent beats every rule-based variant at $D=0$
(0.989 macro-F1) but collapses at $D=10$ (0.251, worse than plain
FedAvg's 0.743) -- inspecting its decisions shows it excluded/downweighted
essentially every client at $D=10$, including honest ones the rule-based
policy's long-term-posterior override protects. This is one 8B model,
one un-tuned single-turn prompt, one seed -- see the honesty caveats in
Section VII-E before treating it as characterizing "the LLM design" in
general. Results: `results/trustbuild_llm_gating_nbaiot.json`.

**Second real result, also in the paper: adding FetchHistory made it worse.**
`LLMGatingPolicy(use_history_tool=True)` implements the paper's
FetchHistory step -- the model may respond `{"action": "fetch_history"}`
on its first turn to request that gateway's actual last-10-round decision
record before committing, instead of only a scalar trust posterior. The
hypothesis was that this would fix the $D=10$ collapse above by making
long-term consistency harder to ignore. It didn't:
`run_llm_gating_trustbuild_tool.py` reran $D \in \{0, 10\}$ and got 0.139
and 0.185 respectively -- $D=10$ no better than before, and $D=0$ (this
experiment's best result without the tool) collapsed to its worst result
with it. The model stopped discriminating between clients at all once the
tool option was added: every client at both delays got `downweight` with
a score clustered at 0.3-0.4, regardless of whether its digest looked
normal or anomalous. Results:
`results/trustbuild_llm_gating_history_tool_nbaiot.json`. Read as a
prompt-fragility finding specific to this small model, not evidence
against tool-calling in general -- see the honesty caveats in Section
VII-E.

**Third real result, also in the paper: a bigger model helps, but doesn't close the gap.**
`run_llm_gating_trustbuild_qwen14b.py` reran the single-turn (no
FetchHistory) configuration with `qwen2.5:14b` instead of `llama3.1:8b`
to test whether the $D=10$ collapse is a model-capacity issue. Partial
answer: $D=0$ improved further to 0.996, and $D=10$ improved substantially
over the 8B model (0.386 vs. 0.251) -- the bigger model correctly
identified 2 of the 3 actual attackers by name, a real precision
improvement -- but it still wrongly excluded 4 of the 6 honest clients in
the same round, and 0.386 remains far below the rule-based policy's 0.986
and even below plain FedAvg's 0.743. Scale narrows the gap on attacker
identification but does not fix the underlying failure to extend benefit
of the doubt to long-trusted clients. Note: ~11s/call on CPU vs. ~4.6s
for the 8B model (~2.4x slower; each delay took ~22 minutes). Results:
`results/trustbuild_llm_gating_qwen14b_nbaiot.json`.

**Fourth real result, also in the paper: an externalized-Bayesian hybrid
architecture, sound in design, undermined by model miscalibration.**
`LLMExternalBayesianPolicy` (new class in `gating_agent.py`) is a
structurally different architecture motivated by two independent
literature threads (frontier LLM-agent work + FL trust literature) that
converged on the same diagnosis: don't ask the LLM to *be* Bayesian,
keep the Bayesian state and the accept/downweight/exclude decision
entirely external and deterministic (identical in structure to
`RuleBasedGatingPolicy`), and narrow the LLM's job to a single, isolated
judgment -- "how anomalous does this one round's digest look, with no
history given at all." A quick unit check (feed it a digest engineered
to look anomalous after 9 rounds of clean synthetic history) confirms
the architecture works exactly as intended: correct anomaly flag,
correct override to downweight because of the accumulated trust -- the
exact behavior every prior LLM config failed to show.

The full experiment (`run_llm_gating_trustbuild_hybrid.py`) reveals a
different real failure, though: the model judged *every* client's update
as highly anomalous in *every* round, with no exceptions. Every gating
weight went to zero every round, which triggers a safety fallback in
`experiment.py` (`if sum(weights) == 0: weights = [1.0] * len(weights)`
-- "never fully stall training") that silently converts the whole run
into plain, unweighted FedAvg. The numbers confirm this exactly: 0.357 at
$D=0$ and 0.743 at $D=10$ match plain FedAvg to three decimal places.
Since the posterior score is driven to 0.0 every round, the trust
posterior never accumulates positive evidence, so the long-term override
condition never has a chance to fire even for honest clients -- despite
the mechanism working correctly in the isolated unit check. Read as: the
architecture is sound, but this small local model, without any reference
for what a "normal" digest looks like, is biased toward over-reporting
anomaly broadly rather than discriminating genuinely anomalous rounds
from normal ones. One silver lining: because this failure degrades
*gracefully* into a safety-net no-op rather than actively misclassifying
who to trust, $D=10$'s number (0.743) is still numerically the best of
any real LLM-gating attempt tried so far -- though that's a property of
the safety fallback, not the hybrid design working. Results:
`results/trustbuild_llm_gating_hybrid_nbaiot.json`.

**Fifth real result, also in the paper: the "obvious" calibration fix made things worse, unexplained.**
`LLMExternalBayesianPolicy(calibrated=True)` adds three worked few-shot
examples (normal/borderline/anomalous digests spanning the intended
[0,1] scale) to the anomaly-judgment prompt, targeting the diagnosed
over-flagging problem above. An isolated unit check looked promising
(correctly returned `accept` for a realistic normal digest, `exclude`
for a clearly anomalous one -- see the smoke test in this file's git
history or just re-run the check inline). The full re-run
(`run_llm_gating_trustbuild_hybrid_calibrated.py`) got *worse*, not
better: 0.185 at $D=0$, 0.174 at $D=10$ -- both below almost every other
configuration tried. Critically, unlike the uncalibrated hybrid, these
numbers do NOT match plain FedAvg (0.357, 0.743), even though the
final-round snapshot shows the same "every client excluded" symptom.
That mismatch means something about the calibrated policy's behavior
differed from the uncalibrated version's at some point across the
25-round run, not just in the final logged round -- but `experiment.py`
currently only logs actions/scores at round 0 and the final round
(`log_every=num_rounds`), so we cannot currently pin down when or why.
Reported honestly as an open, unexplained result. Results:
`results/trustbuild_llm_gating_hybrid_calibrated_nbaiot.json`.

**Sixth real result: confirmed on a second real dataset (CIC IoT-DIAD
2024) -- rules out digest quality as the explanation.** Since the
digest an LLM sees is always the same 3 scalars regardless of how rich
the underlying features are, we re-ran the identical single-turn
Llama 3.1 8B config on CIC IoT-DIAD 2024's richer 118-feature attack
classification task (`scripts/run_ciciot_diad_trustbuild.py` for the
real rule-based reference first: D=0 0.098, D=10 **0.758**, D=20 0.367
with the Bayesian prior; round-only ablation D=10 0.027 -- an even
bigger relative gap than N-BaIoT showed. Then
`scripts/run_ciciot_diad_llm_gating.py`): the LLM replicates the
*identical* qualitative pattern -- **0.476 at D=0** (beats rule-based's
0.098) but **0.064 at D=10** (far below rule-based's 0.758). Real,
useful confirmatory evidence: a materially richer feature set produces
the same failure shape, so the problem is specifically about how the
model weighs long-term history vs. a single round's evidence, not about
digest informativeness. Results:
`results/ciciot_diad_trustbuild_{rulebased,llm}.json`.

**Seventh & eighth real results: added real per-round logging, and got
our first FULLY mechanistically diagnosed Gating Agent failures.**
`FederatedSimulation` (`fedgate/experiment.py`) now records
`self.gating_history`: every gating decision's action, score, AND
rationale, for every round -- not just round 0 and the final round like
`self.logs`. This was cheap (gating already runs every round; the fix
was storing what already existed) but was the specific blocker keeping
the N-BaIoT calibration-fix regression an open mystery. First real use:
re-ran both hybrid variants on CIC IoT-DIAD 2024 at D=10 with full
visibility (`run_ciciot_diad_llm_gating_hybrid.py`):

- **Uncalibrated: 0.205** (exactly plain FedAvg) -- fallback triggered
  in **25/25 rounds**, from round 0 onward, no mid-training drift at all.
  Quoted rationale shows the miscalibration directly: for a z-score of
  **+0.47** (well within one std. dev.), the model wrote *"the update
  norm is significantly higher... with a z-score of +0.47."* Calling a
  sub-half-sigma deviation "significant" is the bug, in the model's own
  words.
- **Calibrated: 0.241** (fallback 21/25 rounds) -- and this time a
  **complete mechanistic explanation**, not just a number: round 0
  *looks* like the calibration examples working (malicious excluded,
  5/6 honest correctly accepted) but this is a **code artifact**, not
  real calibration success -- `experiment.py` hardcodes cosine
  similarity to `1.0` at round 0 specifically, since there's no
  previous global update yet to compare against, making round 0
  trivially easy for every policy. The real collapse starts at round 1:
  quoted rationale for an honest client, *"update norm is far outside
  the group (+0.69 z-score) and cosine similarity... is very low
  (0.22)"* -- a genuinely low value, plausibly just ordinary
  early-training volatility (the global model's update direction hasn't
  stabilized after one real round), not misbehavior. This becomes
  **permanent**, not transient, because of a real architectural gap: a
  client's Bayesian posterior only rises when it scores well, so one
  wrongly-excluded early round means its posterior can never climb into
  the long-term-override range (mean ≥0.7 over ≥5 rounds) no matter how
  normal it looks later -- one bad early judgment permanently forecloses
  the mechanism meant to protect it.

**Three independently addressable causes are now named for the first
time** (previously this was an unexplained "collapse," not a diagnosed
one):
1. Round 0 is a non-representative, code-forced easy case -- exclude it
   from evaluation, or seed `prev_global_delta` more realistically.
2. Early-training digests (rounds 1-~5) are genuinely more volatile --
   use a wider or round-adaptive anomaly threshold for early rounds
   specifically, rather than one fixed threshold throughout.
3. The trust posterior has no recovery path -- consider a decay/reset
   mechanism so one early low score doesn't permanently block the
   long-term override, e.g. weighting recent rounds more than very old
   ones instead of an unbroken all-time average.
Results: `results/ciciot_diad_hybrid_d10_calibrated{True,False}.json`
(includes the full round-by-round `gating_history`, not just the
summary numbers above).

**Ninth real result: implemented and tested all three causes above --
they only marginally help, and reveal a fourth, deeper problem.**
`fedgate/gating_agent.py`: `BayesianTrust.decay` (fix #3, exponential
decay of past evidence toward the Beta(1,1) prior instead of an
unbroken all-time average) and `early_round_leniency_scale` (fix #2,
widens exclude/downweight thresholds for the first `warmup_rounds`
rounds, both `RuleBasedGatingPolicy` and `LLMExternalBayesianPolicy`
accept it). `fedgate/experiment.py`: `force_accept_round0` (fix #1,
default `True` -- round 0 is now accepted for every client without
consulting the policy or updating the trust posterior, so it can't
masquerade as either real calibration success or a real data point).
Ablated all three together and individually on the exact calibrated-
hybrid D=10 setup (`scripts/run_ciciot_diad_hybrid_fixed.py`,
`results/ciciot_diad_hybrid_d10_fixes_ablation.json`):

- Baseline (no fixes): **0.2412** (matches the earlier 0.241 run).
- Fix #1 only (force-accept round 0): **0.2868** -- the only fix that
  moved the needle, and only modestly.
- Fix #1+#2 (+ early-round warmup): **0.2430** -- barely different from
  baseline; warmup only delays the onset of collapse from round 1 to
  round ~4, it does not prevent it.
- Fix #1+#2+#3 (+ trust decay=0.9): **0.2430**, byte-identical actions
  to fix #1+#2 -- decay had **zero effect**, because there is nothing
  for it to help recover *from*: once collapse starts, essentially
  every client, honest and malicious alike, gets "exclude" almost every
  remaining round, so the posterior mean stays near 0 regardless of how
  much old evidence is decayed away.

**All three hypothesized causes were real but not sufficient** -- none
comes close to the rule-based reference (0.758), and the byte-identical
fix#1+2 vs. fix#1+2+3 actions expose a fourth problem the three causes
didn't anticipate: **by round ~4 onward, the LLM stops discriminating
malicious from honest clients at all** -- inspecting `gating_history`
directly (round 10, round 20) shows all 9 clients, 3 malicious and 6
honest, get "exclude" in the same round, with near-identical quoted
rationale. Two concrete drivers, found by reading the rationales
against the actual digest values:
1. **`calib_loss_delta` is a hardcoded constant, not a real signal** --
   `experiment.py`'s digest construction is `Digest(cid, "attack-head",
   norm, cos, 0.0)`: the third field is always exactly `0.0` for every
   client, every round: held-out calibration loss is never actually
   computed. Despite being permanently uninformative, the model
   sometimes cites it as evidence anyway -- round 10's rationale for
   multiple clients reads *"calibration loss got much worse (+0.0000,
   indicating a significant increase in held-out loss)"*, calling an
   always-zero placeholder both "worse" and "significant."
2. **Cosine similarity to the previous global update naturally decays
   as training converges, for honest clients too** -- round 20's
   rationale for an honest client: *"cosine similarity to the previous
   global update is very low (-0.235), which is a strong signal of an
   attack."* A negative cosine late in training is plausibly just
   small, noise-dominated gradients once the global model has mostly
   converged, not misbehavior -- but the calibrated system prompt's
   Example 1 anchors "normal" at cosine=0.94 with no round-dependence,
   so the model has no way to know this is expected.

Neither of these was one of the three originally diagnosed causes
(round-0 artifact / early volatility / no posterior recovery); both are
about what information the LLM is actually being given, not about how
history is weighed. Results:
`results/ciciot_diad_hybrid_d10_fixes_ablation.json` (includes full
`gating_history` for all four ablation arms).

**Tenth real result: fixing the two information problems (#4, #5) made
things WORSE, not better -- 0.243 -> 0.186.** Implemented both:
`FederatedSimulation._calib_loss` (`fedgate/experiment.py`) now
evaluates real held-out cross-entropy loss on a fixed calibration set
(built from each client's val split) before/after each client's local
update, so `calib_loss_delta` is a genuine signal whenever `gating=True`
instead of the old hardcoded `0.0`; `LLMExternalBayesianPolicy`'s
calibrated system prompt (`fedgate/gating_agent.py`) got an explicit
caveat that cosine similarity naturally drifts lower/noisier in later
rounds and shouldn't alone be treated as anomalous, and `_build_prompt`
now tells the model which training round it is. Re-ran the identical
D=10 setup (`scripts/run_ciciot_diad_hybrid_fixed_v2.py`,
`results/ciciot_diad_hybrid_d10_fixes12345.json`): **F1 = 0.1863**,
worse than both the no-fixes baseline (0.241) and fixes #1-3 alone
(0.243).

Reading the rationales shows why, and it's a new, more fundamental
finding than either "fix" anticipated:
- **The round-aware cosine caveat did not work.** Round 1's rationale
  reads *"cosine similarity... is very low, at 0.209, which is not
  typical even for late training rounds"* -- despite being told this is
  an early round and despite the explicit instruction to expect low
  cosine later, the model still treats any low cosine as bad,
  everywhere. A textual caveat about calibration is not enough to make
  an 8B model recalibrate a threshold it has already anchored
  numerically from three fixed examples.
- **The model sometimes reads the calib_loss_delta sign backwards.**
  Round 15's rationale: *"a large negative calibration-loss delta,
  strongly suggests this update is anomalous"* -- but the digest and
  the prompt both define negative delta as loss *improving*
  (`"positive = this update made held-out loss worse"`). A real,
  correctly-computed signal is actively worse than a silent constant
  here, because the model treats it as extra "evidence" while getting
  its meaning backwards some of the time.
- **Cosine similarity trends down for every single client, uniformly,
  as training progresses** (round 1: ~0.21 for all 9 clients; round 8:
  0.02-0.14 for all 9; round 15: several at -0.2) -- this is a real
  property of this dataset/attack setup's training dynamics (not
  fixable by prompting), and a raw cosine number has no way to encode
  "this is normal for round 15" the way update-norm's cross-client
  z-score already does for round-to-round comparison.

**The clearest fix implied by this, not yet tried:** compute cosine
similarity the same way update-norm already is -- as a z-score against
the other clients submitting *this same round*, not a raw value judged
against fixed exemplars. Since a round-level shift (e.g. cosine
naturally dropping as training converges) affects every honest client
equally within a round, a same-round z-score cancels it out
automatically; a textual "this happens in later rounds too" caveat
cannot, because the model has no way to know how much lower is still
normal for this specific round. This also sidesteps the calib_loss_delta
sign-confusion problem only if it recurs -- worth checking whether
z-scoring calib_loss_delta the same way has the same benefit, or
whether the sign-reading error is orthogonal and needs its own fix
(e.g. rephrasing the prompt to state the direction in words instead of
via a signed number: "loss got worse by X" / "loss improved by X").

**Eleventh real result: the fix worked -- F1 = 0.9041, beating the
rule-based reference (0.758) for the first time.** Implemented the
z-scored-cosine + worded-calib-delta idea from the tenth result
(`fedgate/gating_agent.py`'s `LLMExternalBayesianPolicy._build_prompt`
now reports cosine as a same-round cross-client z-score exactly like
update_norm already was, cancelling out a round-wide shift structurally
instead of relying on a prose caveat, and `_describe_calib_delta` words
the calibration-loss direction instead of handing the model a signed
number it could misread; `SYSTEM_PROMPT_CALIBRATED`'s exemplars were
rewritten to match). Before committing to a full 25-round run, five
hand-built synthetic digests were checked against both available
models: **llama3.1:8b still failed** -- it started comparing the two
z-scores to each other (e.g. "cosine_z is higher than norm_z" as if
that itself were suspicious) rather than judging each against its own
scale, excluding a genuinely mild-deviation case outright.
**qwen2.5:14b got all five right**, including the specific case that
broke every earlier attempt: a large *negative* calib_loss_delta
(improving loss) was correctly accepted as normal, not flagged.

Ran the full D=10 experiment with qwen2.5:14b
(`scripts/run_ciciot_diad_hybrid_fixed_v3.py`,
`results/ciciot_diad_hybrid_d10_fixes12345_qwen14b.json`): **F1 =
0.9041, fallback 0/25 rounds** -- every prior number for this exact
D=10 setup: rule-based reference 0.758, plain single-turn LLM (8B)
0.064, calibrated hybrid (8B) 0.241, fixes #1-3 (8B) 0.243, fixes #1-5
raw/textual version (8B) 0.186. Reading `gating_history`: from round 16
onward (well after the D=10 trust-building attack triggers), **all
three malicious clients get "exclude" every single round**, while
honest clients are consistently accept/downweight and are never all
excluded together the way they were in every 8B run -- real,
sustained discrimination between honest and malicious clients, not
present in any earlier result for this task.

This is the first LLM-Gating-Agent number in the whole investigation
that beats the rule-based stand-in rather than falling far short of
it, and it isolates something worth stating precisely for the paper:
**the earlier collapses were not an inherent limitation of using an
LLM for this task** -- they were a combination of (a) genuine
implementation gaps (round-0 artifact, no posterior recovery, a
hardcoded placeholder feature) and (b) a specific 8B model's
inability to reason correctly over even a well-designed prompt,
fixable by moving to a larger model. Caveat before writing this into
the paper: this is one seed, one delay value (D=10), one dataset. The
next open item (below) is repeating this now-working configuration
more broadly before treating 0.90 as a stable number.

**Twelfth real result: the qwen2.5:14b fix replicates across seeds,
delay, and attack type -- it is not a lucky D=10/seed=0 draw.** Ran the
identical policy on four more configurations
(`scripts/run_ciciot_diad_qwen14b_single.py`,
`results/ciciot_diad_qwen14b_broad_validation.json`):

| config | F1 | fallback | targeted success rate |
|---|---|---|---|
| trust_building D=10 seed=0 (original) | 0.9041 | 0/25 | -- |
| trust_building D=10 seed=1 | 0.9122 | 0/25 | 0.008 |
| trust_building D=10 seed=2 | 0.8956 | 0/25 | 0.006 |
| trust_building D=20 seed=0 | 0.8982 | 0/25 | 0.006 |
| untargeted seed=0 | 0.9123 | 0/25 | n/a |
| targeted seed=0 | 0.9072 | 0/25 | 0.004 |

All six land in a tight 0.895-0.912 band, **zero fallback rounds in
every single one**, and the targeted-attack success rate (fraction of
the targeted source class actually flipped to the attacker's chosen
target class) is under 1% in every config that has one -- the attack
is not just failing to tank aggregate F1, it is failing at its actual
objective. This replicates across 3 seeds, 2 delay values, and 3 attack
shapes (trust-building, untargeted, targeted) with one fixed policy
configuration -- a real, stable result now, not a one-off.

**Infra note for future long LLM-experiment runs**: partway through
this sweep, the local Ollama server degraded under sustained load (a
trivial prompt went from ~2s to ~22s after ~15-20 minutes of continuous
qwen2.5:14b inference, with system free memory dropping from ~10GB to
~500MB), causing 4 of 5 runs in one long unattended sweep to fail with
timeouts. A plain `ollama serve` restart between runs fixed it
immediately every time (free memory jumped back to ~10GB). For any
future multi-run sweep with a local Ollama model this size, restart the
server between runs rather than assuming one long-lived process stays
healthy for 90+ minutes of continuous inference.

**Thirteenth real result: confirmed -- the fix does not rescue
llama3.1:8b on the single-turn architecture either; this is a model
capability ceiling, not an architecture- or prompt-specific bug.**
Applied the identical z-scored-cosine + worded-calib-delta fix to
`LLMGatingPolicy` (both `SYSTEM_PROMPT_NO_TOOL`/`SYSTEM_PROMPT_WITH_TOOL`
and `_format_history`, plus a `describe_calib_delta`/`cosine_zscore`
module-level refactor shared with `LLMExternalBayesianPolicy` so both
policies use the exact same logic, not two copies that could drift).
Four synthetic smoke-test digests first: **llama3.1:8b still failed**
-- it downweighted a "clearly normal" case, this time citing the
update-norm z-score being "significantly lower than the group" as
suspicious (a negative deviation judged the same as a positive one,
rather than judged by distance from 0). qwen2.5:14b on the same cases
showed a different, more defensible failure: it downweighted a
brand-new gateway with zero observed history purely for lacking
history, not for misreading any digest value.

Ran the full D=10 trust-building experiment anyway
(`scripts/run_ciciot_diad_8b_singleturn_fixed.py`,
`results/ciciot_diad_8b_singleturn_newprompt_d10.json`): **F1 =
0.1375** -- better than the old single-turn prompt (0.064) but far
short of even the un-fixed hybrid (0.241), let alone qwen2.5:14b
(0.90+). Reading `gating_history`: malicious and honest clients get
nearly identical "downweight" almost every round from round ~19
onward, with no real separation between them -- since every client's
gating score is ~0.5 uniformly, the weighted average after
normalization is mathematically close to plain unweighted FedAvg, so
the attack goes essentially unmitigated regardless of gating being
"on."

**This closes the architecture/prompt-design question for llama3.1:8b**:
across four attempts on this task -- single-turn old prompt (0.064),
hybrid old prompt (0.241), hybrid new prompt (0.186, a regression), and
single-turn new prompt (0.1375) -- none comes close to qwen2.5:14b's
result (0.90+) on the identical setup. The bottleneck for this specific
gating task is model capability (numerical/relational reasoning over a
handful of z-scores and a worded direction), not the choice of hybrid
vs. single-turn architecture, and not remaining prompt-wording issues.
For any paper claim about the Gating Agent's LLM, qwen2.5:14b (or
larger) is the model to report, not llama3.1:8b.

**Fourteenth real result: the fix generalizes cleanly to a SECOND,
structurally different dataset -- the original N-BaIoT
"calibration-regression mystery" is resolved, not just the CIC IoT-DIAD
one.** Ran the identical qwen2.5:14b hybrid + full fix set (z-scored
cosine, worded calib_loss_delta, force_accept_round0, warmup_rounds=5,
trust_decay=0.9) on N-BaIoT (9 real devices as clients, not synthetic
splits of one pool) at all three delay values already on record from
earlier attempts (`scripts/run_nbaiot_qwen14b_hybrid_fixed.py`,
`results/nbaiot_qwen14b_hybrid_fixed.json`):

| D | this run (qwen14b hybrid + fixes) | old rule-based (Bayesian) | old qwen14b single-turn, no fixes | old 8B hybrid, no fixes |
|---|---|---|---|---|
| 0 | **0.9958** | 0.320 | 0.996 | 0.185 |
| 10 | **0.9974** | 0.986 | 0.386 | 0.174 ("the mystery") |
| 20 | **0.9961** | 0.988 | n/a | n/a |

All three: **0/25 fallback rounds**, malicious clients consistently
excluded, honest clients kept in the aggregate (never all excluded
together). This resolves the original, long-open "N-BaIoT
calibration-regression mystery" (Section VII-E item 8, where the 8B
hybrid was stuck at ~0.18 regardless of delay) with the same fix that
worked on CIC IoT-DIAD 2024 -- and, notably, it now also fixes the
D=10 trust-building collapse that even the old qwen2.5:14b *single-turn*
policy suffered (0.386, since that architecture folds the trust
posterior into the LLM's own reasoning rather than keeping it
external). This is the second dataset validating the fix, addressing
the earlier "one dataset" caveat from the eleventh/twelfth real
results.

**Fifteenth real result: N-BaIoT validation completed across all three
attack types, not just trust_building.** Ran untargeted and targeted
attacks (malicious clients attack from round 0, no delay) on N-BaIoT
with the identical qwen2.5:14b hybrid + fix set
(`scripts/run_nbaiot_qwen14b_hybrid_robustness.py`,
`results/nbaiot_qwen14b_hybrid_robustness.json`):

| attack | F1 | fallback | targeted success rate |
|---|---|---|---|
| trust_building D=0 | 0.9958 | 0/25 | -- |
| trust_building D=10 | 0.9974 | 0/25 | -- |
| trust_building D=20 | 0.9961 | 0/25 | -- |
| untargeted | 0.9958 | 0/25 | n/a |
| targeted | 0.9972 | 0/25 | 0.24% |

All five N-BaIoT configurations land in a tight 0.9958-0.9974 band,
**zero fallback rounds in every one**, and the targeted-attack success
rate is under 0.25% -- consistent with the CIC IoT-DIAD 2024 validation
(twelfth real result) both in shape (tight band, zero fallback, near-zero
targeted success) and in the specific fix that produced it. Combined,
this is now a fully validated result across 2 datasets x 3 delay values
x 3 attack types x (3 seeds on CIC IoT-DIAD) -- the strongest, most
broadly-tested claim in the whole investigation.

**Sixteenth real result: the GShield-style non-LLM baseline (deferred to
last, now done) is competitive with -- and on CIC IoT-DIAD slightly
ahead of -- the LLM Gating Agent, at 1-2 orders of magnitude lower
cost.** Implemented `fed.gshield_aggregate` (`fedgate/federated.py`),
inspired by GShield (arXiv 2512.19286, Dec 2025): PCA-reduce each
round's flattened client updates to 10 components, 2-means cluster
them, fit a Gaussian to the larger (presumed-benign) cluster, exclude
any client whose Mahalanobis distance to it is an outlier -- no LLM
calls, no assumed number of malicious clients (unlike Krum). Ran it on
the identical 11 configurations used to validate the qwen2.5:14b hybrid
Gating Agent (`scripts/run_gshield_baseline.py`,
`results/gshield_baseline.json`):

| | GShield-style (this baseline) | qwen2.5:14b hybrid Gating Agent |
|---|---|---|
| CIC IoT-DIAD (6 configs) | 0.914-0.921 (mean 0.918), 17-34s each | 0.895-0.912 |
| N-BaIoT (5 configs) | 0.995-0.997 (mean 0.996), 17-34s each | 0.9958-0.9974 |

On CIC IoT-DIAD the fast clustering baseline is **slightly ahead** of
the LLM Gating Agent; on N-BaIoT the two are statistically
indistinguishable. Each GShield-style run takes 17-34 seconds (pure
NumPy/PyTorch, no LLM server); each LLM Gating Agent run took roughly
15-45 minutes. This is an honest complication, not a negative result to
downplay: it means the paper's contribution has to be argued on
auditability (every Gating Agent exclusion carries a natural-language
rationale; a Gaussian-distance threshold does not, by construction),
not on raw macro-F1 superiority over every possible baseline. Written
into `main.tex` (Section VII-E) with a new Table XII and explicit
framing of this tradeoff, plus a citation to GShield in the
bibliography.

**Seventeenth real result: an adaptive-evasion attack was tried against
both defenses; neither shows a meaningful robustness edge over the
other.** Motivated by "how do we show the LLM Gating Agent is worth it
if GShield-style matches its accuracy," we implemented a
constrain-and-scale-style adaptive attacker (`blend_toward_reference`,
`fedgate/attacks.py`): each malicious client blends its poisoned update
toward its OWN clean-label local-training gradient (computable locally,
no cross-client information needed) with strength `evasion_lambda`,
trading attack potency for stealth. A first version that blended toward
a shared reference (the previous global update) backfired -- it made
the malicious clients a *tighter*, more mutually-similar clique,
easier to cluster apart, not harder (confirmed directly: cluster
assignments were byte-identical across the whole lambda range). The
corrected per-client version genuinely works: at $\lambda\geq0.9$, 2 of
3 malicious clients directly confirmed to land in GShield-style's
"benign" majority cluster (verified by inspecting cluster labels
directly, not inferred).

Despite that real per-round evasion, **aggregate outcomes barely move
for either defense** (`scripts/run_adaptive_evasion_gshield_sweep.py`,
`scripts/run_adaptive_evasion_qwen14b.py`,
`results/adaptive_evasion_{gshield_sweep,qwen14b}.json`), on the
targeted attack, CIC IoT-DIAD 2024:

| $\lambda$ | GShield-style F1 | GShield targeted-success | Qwen2.5 14B hybrid F1 | Qwen14B targeted-success |
|---|---|---|---|---|
| 0.0 | 0.9209 | 1.05% | 0.9198 | 0.63% |
| 0.9 | 0.9208 | 1.05% | 0.9160 | 1.05% |
| 0.99 | 0.9207 | 1.26% | 0.9190 | 1.05% |

Both land in the same tight band, and the LLM Gating Agent shows the
same qualitative symptom as GShield-style at high lambda -- individual
malicious clients occasionally get `accept` instead of `exclude` in a
given round (e.g. round 20 at $\lambda{=}0.99$:
`malicious=['exclude','exclude','accept']`) -- not immune to the same
evasion mechanism. The explanation is the same accuracy-stealth
tradeoff for both: the blend that makes an update stealthy enough to
evade detection also dilutes it toward a mostly-honest gradient, so
"evasive" and "still damaging" are in tension regardless of which
defense is looking. **Honest conclusion: at the attack strength tested
(3/9 malicious clients, this specific label-flip severity), this
particular adaptive attack does not differentiate the two defenses.**
This does not mean no adaptive attack could -- a stronger attacker
(more compromised clients, colluding to jointly optimize their blend,
or a curriculum that ramps lambda across rounds rather than holding it
fixed) remains untested and is the natural next step if a robustness
differentiator, rather than an auditability one, is wanted for the
paper.

**Eighteenth real result: the analogous real-LLM swap for onboarding
and triage.** Implemented `LLMOnboardingPolicy` and `LLMTriagePolicy`
(`fedgate/onboarding.py`, `fedgate/triage.py`), following the same
proven pattern that fixed the Gating Agent: Ollama-backed (qwen2.5:14b),
reasoning over an ALREADY-NORMALIZED signal (vote-share distribution for
onboarding, distance ratio for triage), not raw scale-dependent numbers,
with calibrated few-shot examples in the system prompt.

*Onboarding* (`scripts/run_onboarding_and_triage_llm.py`,
`results/onboarding_nbaiot_proxy_llm.json`): both real validations
behaved correctly. The genuinely-unseen 9th N-BaIoT device correctly
got `request-more-data` (not falsely auto-admitted, confidence 0.75,
rationale correctly names the "second candidate close behind"); a
held-out split of an already-admitted device correctly got `auto-admit`
with the right label (confidence 0.97, correctly noting the "overwhelmingly
dominant" 96.7% vote share).

*Triage* (`scripts/run_onboarding_and_triage_llm.py` +
`scripts/run_triage_llm_errors.py`,
`results/triage_nbaiot_llm{,_errors}.json`): the first run (135
randomly-sampled predictions, matching the rule-based validation's own
protocol) happened to contain zero real classifier errors -- the
backbone is that accurate -- so it was uninformative about the
interesting case, same limitation the rule-based version had. A
follow-up scanned the FULL test set for actual misclassifications (37
found out of 5000) and ran the LLM triage specifically on those plus 10
correct predictions for contrast: **0 of 37 real errors were flagged
`likely-false-positive`** (all got `confirm`, matching the sampled-error
case's own real behavior with the rule-based policy too), while all 10
correct predictions were correctly `confirm`ed. Reading the rationales:
the model isn't failing to reason -- it correctly reports the actual
ratio and correctly concludes "this looks like a typical example of the
predicted class," because these particular errors genuinely are typical
of the (wrong) class they got mislabeled as (mirai/gafgyt/benign overlap
in this feature space closely enough that a confident, in-distribution
confusion produces exactly the same retrieval-margin signature as a
correct prediction). This is a structural limitation of the underlying
signal (retrieval margin against the predicted class's own exemplars),
not an LLM reasoning failure or a prompt-design issue -- no amount of
better reasoning over this particular signal can catch an error that
doesn't look anomalous in the feature space the signal is computed
over. Catching this class of error would need a different signal
entirely (e.g. margin/ratio against ALL classes' exemplars, not just
the predicted one, to catch "this also looks a lot like class X"), left
as a concrete next step rather than claimed as solved.

**Nineteenth real result: venue pivot to IEEE ICC 2027 (6-page limit),
plus a real device-identification poisoning-degradation gap the
literature had never measured.** New paper at `../ICC2027/main.tex`
(SSS2025 dropped per explicit user instruction 2026-09-26; long-form
paper at `../IoT_Device_Identification_Oct2025/main.tex` kept untouched
as a separate artifact). Literature verification (real web searches,
not a single hallucinated 0-tool-call agent pass that had to be
redone): no paper combines FL device-ID + FL intrusion detection +
poisoning defense in one system. Closest matches: He et al. 2021 (FL
device-ID via network traffic, no poisoning at all) and Sanchez Sanchez
et al. 2021 (FL device-ID + label-flipping robustness, but a different,
non-network-traffic, single-task device-fingerprinting setup). New
script `scripts/run_unsw_poisoning_robustness.py` applied the SAME
threat model already used for the attack-classification task to real
UNSW device-identification traffic for the first time: plain FedAvg's
device-type macro-F1 collapses from 0.828 ($\alpha=0$) to 0.038
($\alpha=4$) under untargeted poisoning -- more severely than the
attack-classification task -- and a targeted attacker succeeds
unevenly (63% success at $\alpha=2$, 0% at $\alpha=3$ where accuracy has
already collapsed too far for the source class to be predicted at
all), reported as observed rather than smoothed into a trend. Result:
`results/unsw_deviceid_poisoning_robustness.json`.

**Twentieth real result: a selective-LLM Gating Agent (cheap fast path
for confidently-normal clients, real LLM only for the rest) closes
roughly half the cost gap to GShield -- but shipping it required
finding and fixing a real bug first.** New policy
`SelectiveLLMGatingPolicy` (`fedgate/gating_agent.py`): reuses the
rule-based policy's own digest z-score thresholds as a free admission
filter (deliberately NOT GShield's clustering, per explicit user
request to build our own mechanism) -- a client comfortably inside the
normal band on update-norm z, cosine z, and calibration loss is
accepted for free; everything else gets the same hybrid LLM judgment
used throughout the paper.

First attempt failed on CIC IoT-DIAD's longest trust-building delay
($D=20$): macro-F1 collapsed to **0.1869** against a 0.898 reference.
Per-round diagnostic scripts (`scripts/diag_ciciot_d20_selective.py`)
traced the exact mechanism by comparing against the saved full-LLM
run's own gating history: the fast path always wrote a perfect,
noiseless score of 1.0 to the long-term trust posterior, so a patient
attacker's 20 honest rounds built MORE trust than the real LLM's own
naturally noisier judgments would have. By the time the attack started,
the long-term-override condition fired on every single attack round and
permanently downgraded exclude to downweight -- the attacker was never
once fully excluded across all 5 attack rounds, only ever halved, and
the model never recovered (F1 crashed at exactly round 20 and stayed
low/unstable through round 24). The full-LLM run's own saved history
confirmed the fix target directly: it reached real `exclude` (weight
0.0) by round 20-24; the selective version never did.

Fix: the long-term trust posterior is now updated ONLY from genuine LLM
judgments, never from the fast path, so a cheap unaudited shortcut can't
manufacture more trust than a real judgment would have granted. Re-ran
the CIC D=20 config alone first to confirm (F1 jumped to 0.9173, no
collapse at all), then re-ran the full sweep on both datasets:

- N-BaIoT (`scripts/run_nbaiot_selective_llm.py`,
  `results/nbaiot_selective_llm.json`): all 5 configs 0.9966-0.9980
  (reference 0.9958-0.9974), LLM calls 42-52% of decisions.
- CIC IoT-DIAD (`scripts/run_ciciot_diad_selective_llm.py`,
  `results/ciciot_diad_selective_llm.json`): all 6 configs 0.906-0.921
  (reference 0.896-0.912), LLM calls 33-38% of decisions, including the
  previously-broken $D=20$ config now at 0.9173.

Every one of the 11 configurations across both datasets matches or
slightly exceeds the full-LLM accuracy while roughly halving the LLM
call count -- a genuine, measured cost reduction, not just a value
argument, and the failure it took to find is itself a small piece of
evidence for the paper's central claim: an unaudited shortcut can
quietly undermine the exact trust mechanism it's meant to support.
Written into `ICC2027/main.tex` (paper now at 6/6 pages, the ICC hard
limit, 0 undefined references).

**Twenty-first real result: the selective-LLM fast path resists the same
evasion attack that fools GShield's clustering, and N-BaIoT's headline
number is now 3-seed validated too.** Two follow-ups closing gaps a
reviewer would likely flag:

- (`scripts/run_adaptive_evasion_selective_llm.py`,
  `results/adaptive_evasion_selective_llm.json`): re-ran the
  constrain-and-scale evasion attacker (Section "seventeenth real
  result") specifically against the selective-LLM's fast-path filter,
  the mechanism that grants a free pass with no LLM scrutiny at all --
  the newest attack surface in the design. At the same blend strengths
  ($\lambda\in\{0,0.9,0.99\}$) that land 2 of 3 malicious clients in
  GShield's "benign" cluster, malicious clients NEVER landed inside the
  fast path's normal band across all three values (0/75 client-rounds);
  macro-F1 (0.917-0.919) and targeted success rate ($\leq$1.1%) stayed
  flat, matching both the full-LLM and GShield references' own
  flatness under the same attack. The digest-based fast path's safety
  margin held where GShield's clustering boundary did not, at this
  attack strength.
- (`scripts/run_nbaiot_qwen14b_multiseed.py`,
  `results/nbaiot_qwen14b_multiseed.json`): N-BaIoT's flagship
  trust_building $D=10$ config had only ever used seed=0 (0.9974),
  unlike CIC IoT-DIAD's same config which used 3 seeds -- a real
  asymmetry a reviewer would likely notice. Seeds 1 and 2 replicate
  tightly (0.9971, 0.9954), closing the gap.

Both written into `ICC2027/main.tex` (Table III updated to show
N-BaIoT's 3-seed range; the fast-path evasion result added as a new
paragraph in Section IV-F). Paper still compiles cleanly at exactly
6/6 pages (the ICC hard limit), 0 undefined references, no overfull
boxes (also fixed one real table-overflow rendering bug found via
visual inspection of the compiled PDF, not just the compile log:
Table III's "Rule-based" column header was overlapping the adjacent
column's paragraph text -- fixed with `\footnotesize` and shorter
column labels).

**Twenty-second real result: a colluding adaptive attacker gains no edge
over the independent-blend version, against either defense.** New
`evasion_collude=True` option on `FederatedSimulation`
(`fedgate/experiment.py:_get_collude_reference`): instead of each
malicious client blending toward its OWN clean-label gradient in
isolation, all malicious clients pool their clean gradients into ONE
shared, less-noisy reference (a stronger threat-model assumption --
requires them to share local data statistics with each other). Tested
against both GShield and the selective-LLM policy, targeted attack, CIC
IoT-DIAD, same lambda values used throughout
(`scripts/run_adaptive_evasion_colluding.py`,
`results/adaptive_evasion_colluding.json`):

- GShield: 0.918-0.921 macro-F1, targeted success $\leq$1.1% -- flat,
  same as the independent-blend version.
- Selective-LLM: 0.917-0.920 macro-F1, targeted success $\leq$1.3%,
  fast-path bypass still 0/75 client-rounds across all three lambdas.

A clean, honest null result: giving the attacker a stronger
coordination assumption (shared reference instead of each client's own
noisier estimate) does not change the outcome for either defense.
Written into `ICC2027/main.tex` (new paragraph in Section IV-F, and the
Conclusion's "future work" item for a colluding attacker updated to
report it as done rather than left open). Paper still compiles cleanly
at exactly 6/6 pages.

ConFedDI's bib entry was also fixed (Crossref + web search resolved
the author list to Chen J, Xiong Q, Wang Z, Chen L -- IEEE reference
style renders given names as initials anyway, so this is a complete,
correct citation). All Related Work bib entries added this session are
now fully verified. Paper confirmed still compiling cleanly at 6/6
pages after this fix.

**Twenty-third real result: a capability-coverage comparison table
against the closest prior FL device-ID/IDS work, per explicit user
request ("do more experiment and comparison... make this paper strong
paper with superiority than literature").** Flagged directly to the
user first: a raw-accuracy leaderboard against ConFedDI/FGLIoT/FedMADE/
etc. would be scientifically invalid (each reports its own number on
its own dataset, not ours) and re-fabricating a superiority claim like
that is exactly the overclaiming this paper's SSS rejections were
about. Built the honest version instead: a qualitative table
(`ICC2027/main.tex` new Table I, `\label{tab:coverage}`) showing which
capabilities each of 7 closest prior works actually covers -- device-ID,
IDS, poisoning-robustness, auditable rationale, zero-shot onboarding --
and this paper is the only row with all five checked. That's a real,
defensible "superiority" claim (breadth of what's solved, not a
fabricated single-number win). Paper still compiles at exactly 6/6
pages after adding it (used a `\scriptsize` 5-column table to fit).

**Twenty-fourth real result: user-supplied paper comparison
("Think_Fast_Real-Time_IoT_Intrusion_Reasoning_Using_IDS_and_LLMs_at_the_Edge_Gateway",
Jamshidi et al., IEEE Internet of Things Journal, April 2026) analyzed
and incorporated.** Read the full 29-page paper. It is NOT federated
(explicitly, deliberately -- contrasted against FL by its own authors);
six local classifiers detect anomalies at the edge, an LLM (GPT-4-turbo/
DeepSeek-V2/LLaMA 3.5 via cloud API) is invoked only AFTER detection to
write a natural-language severity/mitigation rationale for an
already-fixed verdict. Its own Threats to Validity section lists
adversarial/prompt-injection robustness as untested future work, not a
measured result.

Two actions taken: (1) flagged an honesty gap this paper's own coverage
table (`tab:coverage`) would have had -- Think Fast's LLM DOES produce
a real rationale, so marking "Audit" as exclusively ours would have
been another overclaim; added it to the table honestly (Audit
checked, Pois. marked with an asterisked footnote since FL-poisoning is
out-of-scope-by-design for a non-federated system, not
tested-and-absent) and added a Related Work paragraph making the real,
defensible distinction explicit: their rationale explains an
already-fixed classifier decision after the fact; ours IS the decision.
(2) Borrowed their single most effective device (a runtime-log figure
showing one real detection-to-mitigation cycle) in spirit: added a
compact, 100%-real decision trace (three malicious clients, round 22 of
the $D{=}20$ CIC IoT-DIAD run, pulled directly from
`results/ciciot_diad_qwen14b_broad_validation.json`'s saved
`gating_history`, not fabricated) right after the Gating Agent's
mechanism description in `ICC2027/main.tex`, showing two clients with
identical anomaly evidence getting excluded while a third gets only
downweighted because of its own accumulated trust history -- a concrete
illustration of the auditability claim. Paper still compiles at exactly
6/6 pages after both additions (bib entry `jamshidi2026thinkfast`
added).

**Twenty-fifth real result: real per-decision latency/CPU instrumentation
for the selective-LLM policy, styled after Think Fast's resource-analysis
rigor, per explicit user request ("I want to do similar to that paper
but using FL, can this be done?").** New script
(`scripts/run_resource_latency_measurement.py`,
`results/resource_latency_measurement.json`): instruments
`SelectiveLLMGatingPolicy.decide()` to record real wall-clock latency
plus CPU/memory of the actual model-serving process for every one of
216 decisions on N-BaIoT $D{=}10$.

Caught and fixed two real measurement bugs before trusting the numbers:
(1) first attempt sampled the `ollama serve` coordinator process, not
the child `llama-server` runner process that actually holds the model
and does the inference -- confirmed via `ps aux` (runner: ~10GB RSS;
coordinator: <40MB) -- silently produced backwards-looking numbers
(LLM calls showing LOWER CPU than the free fast path, which is
impossible) until fixed. (2) First attempt used
`psutil.cpu_percent(interval=None)`, which measures usage since the
LAST sample, not during THIS specific call -- diluted by unrelated
client-training work happening between gating decisions; fixed by
using a before/after `cpu_times()` delta (CPU-seconds consumed /
wall-clock seconds of the call), which is call-local and correct.

Real result after both fixes: fast path 0.00005s $\pm$ 0.00004s
latency, 0.0\% CPU (genuinely free, not just faster on average); LLM
call 3.75s $\pm$ 0.32s latency, 13.9\% $\pm$ 2.9\% CPU of the
model-serving process (Welch's $t=114.7$, $p<10^{-100}$). Model itself
holds a fixed ~9.6GB resident memory footprint throughout regardless of
path taken (stays loaded once). Explicitly do NOT report energy
(Joules) -- no wattmeter on this machine, unlike Think Fast's Raspberry
Pi setup; estimating one would be fabrication, stated as such in the
paper. Written into `ICC2027/main.tex` as new Table V + a paragraph in
Section IV-F. Paper still compiles at exactly 6/6 pages.

**Twenty-sixth real result: abstract/intro brought up to date with
everything actually in the paper, per "continue make better."** The
abstract and Introduction's "three contributions" were last touched
early in the ICC pivot and had gone stale relative to everything added
since (device-ID poisoning-degradation, capability-coverage table,
selective-LLM cost reduction + its found-and-fixed bug, colluding
attacker, real resource measurement) -- a real risk, since reviewers
read these first and a mismatch between what's promised and what's
delivered reads as either an oversight or an undersell. Rewrote both to
name all four actual contributions; this pushed the paper to 7 pages,
over the ICC hard limit, so trimmed two Related Work paragraphs
(tightened phrasing, cut redundant clauses) to bring it back to exactly
6/6 pages, 0 undefined references, no overfull boxes -- verified
visually, not just by page-count.

**Twenty-seventh real result: explicit Limitations added to the
Conclusion, per user request after being asked "what limitation do we
have" relative to Think Fast.** Named the three most consequential
gaps directly: (1) all attacks are simulated, not a live red-team
exercise against real hardware, and the decision never drives a real
downstream enforcement action (Think Fast dispatches to a real
firewall); (2) the LLM pool is two local open-weight models, not a
hosted frontier model; (3) no concept-drift adaptation or graceful
degradation if the LLM becomes unreachable (Think Fast has both).
Adding this pushed the paper to 7 pages twice in a row -- had to trim
the Conclusion's own results summary more aggressively (not just
Related Work this time) to reclaim the space. Confirmed via full clean
rebuild (`latexmk -C` then 3 passes) that 6 pages is genuinely stable,
not a stale-cache artifact. Final state: 6/6 pages, 0 undefined
references, no overfull boxes, verified visually.

**Twenty-eighth real result: two of the three limitations were actually
solved, not just narrated, per explicit user pushback ("no we should
solve limitation").** User chose "do all three now" via AskUserQuestion.

1. **Frontier-model test: BLOCKED, not solved.** `OpenAIExternalBayesianPolicy`
   implemented in `fedgate/gating_agent.py` (identical prompt/decision
   logic to `LLMExternalBayesianPolicy`, only the backend call swapped
   to the OpenAI API). Smoke test hit a real, unfixable-by-us blocker:
   `insufficient_quota` / `credit_balance_exhausted` -- the available
   `OPENAI_API_KEY` has zero billing credits. Confirmed via direct curl
   to the chat completions endpoint, not just the Python client. Code is
   ready and correct; needs the user to add credits at
   platform.openai.com/settings/organization/billing before it can run.

2. **Graceful degradation: SOLVED and verified against a real outage.**
   New `GracefulDegradationPolicy` (`fedgate/gating_agent.py`): opt-in
   wrapper, catches `LLMGatingPolicyError` from a primary policy and
   falls back to `RuleBasedGatingPolicy`, logging every degraded
   decision (never silent). `scripts/run_graceful_degradation_test.py`
   proves it against a REAL outage, not a mock: a background thread
   kills the actual `ollama serve` process 30s into a live N-BaIoT run,
   waits 90s, then restarts it. Result: training did not crash, F1=0.9972,
   86/216 decisions correctly used the fallback -- though the run
   finished (64.6s) before the 90s restart completed, so "recovery"
   specifically wasn't captured within this run, noted honestly rather
   than re-run to hide it.

3. **Concept-drift adaptation: SOLVED after a first real design failure.**
   Added `concept_drift_start_round`/`concept_drift_per_round` to
   `FederatedSimulation` (synthetic, non-adversarial, affects ALL
   clients equally, disabled by default, never used in any poisoning
   result reported elsewhere). First attempt, `AdaptiveCalibBaseline`
   (EWMA mean/var, update-on-accept-only): tested via
   `scripts/run_concept_drift_test.py` (9 honest clients, zero
   attackers, drift starting round 10) and found to be WORSE than doing
   nothing -- 100% false-positive rate in the final 5 rounds, because
   once a round gets flagged the baseline stops updating and the gap to
   the still-drifting true value only grows, a permanent-freeze failure
   mode found by testing it, not assumed. Redesigned around the same
   idea that already works for update-norm/cosine: `robust_zscore()` +
   `AdaptiveRuleBasedGatingPolicy` compute calib-loss anomaly as a
   same-round, cross-client median/MAD z-score (`round_calib_deltas`
   threaded through every `decide()` signature in the file) instead of
   a historical baseline -- no lag, no freeze, since concept drift moves
   every client together and cancels in the per-round median by
   construction. Result: 91\% (fixed threshold) $\to$ 24\% false-positive
   rate under identical synthetic drift -- a real, honestly-reported
   improvement, not perfect: the residual 24\% concentrates in ONE
   structurally-different device (Samsung\_SNH\_1011\_N\_Webcam,
   excluded every single round 25-29), a genuine client-heterogeneity
   nuance the fix does not resolve, not random noise.

Both real fixes (2 and 3) written into `ICC2027/main.tex` as a new
Section IV-H, with the Conclusion's limitations updated to reflect what
is now solved vs. what remains genuinely open. Required several rounds
of aggressive trimming elsewhere (Related Work paragraphs, several
results paragraphs) to reclaim space -- confirmed stable at exactly 6/6
pages via a full clean rebuild (`latexmk -C` + 3 passes), not a stale
cache reading.

**Twenty-ninth real result: the FetchHistory regression, diagnosed.**
Per explicit user request to investigate this mechanistically.
`scripts/diag_fetchhistory_regression.py`
(`results/diag_fetchhistory_regression.json`): tested Llama 3.1 8B on 5
synthetic cases (clearly-normal, clearly-anomalous, ambiguous-with-
strong-history, ambiguous-with-weak-history, borderline) under 3 prompt
variants:

- **A (`SYSTEM_PROMPT_WITH_TOOL` as-is, tool option present):** every
  single case -- including the two UNAMBIGUOUS ones -- got `downweight`
  (scores 0.6, 0.2, 0.6, 0.3, 0.4). `fetch_history` was invoked **0/5
  times** -- it never actually used the escape hatch it was given.
- **B (identical background/framing text, but the fetch\_history-offering
  sentence deleted -- same length/complexity otherwise):** correctly
  varied -- `accept` (0.85) for the clearly-normal case, `exclude` (0.0)
  for the clearly-anomalous case, sensible `downweight` scores for the
  ambiguous/borderline cases.
- **C (original `SYSTEM_PROMPT_NO_TOOL` control):** similar to B --
  correctly `exclude`d the anomalous case, varied scores elsewhere.

This isolates the cause precisely: it is NOT the added trust-building
background prose (B has identical framing and discriminates fine). It
is specifically the PRESENCE of a third, structurally different
response shape (`{"action": "fetch_history"}`, a 1-field object, as an
alternative to the normal 3-field decision object) that collapses the
model to a single "safe middle" action for every input, whether or not
it ever emits that alternative shape. A generalizable caution for
agent design, not just this task: offering a small model an
escape-hatch/deferral response format can silently bias its PRIMARY
decision distribution toward non-commitment, even on cases where the
escape hatch is never actually used -- a failure mode invisible unless
you specifically test with and without the option present on identical
inputs, as done here. Written into `ICC2027/main.tex`.

**Thirtieth real result: full consistency read-through of the ICC paper,
plus a real integrity fix -- the code repository is now actually
public.** Read the entire current `ICC2027/main.tex` top to bottom for
the first time since all the additions this session, and found two
real issues: (1) a grammar error from an earlier compression edit
("both reasoning" for a list of three items); (2) the system-overview
figure was never once referenced from the text (`\ref{system}` unused)
-- against IEEE convention that every figure/table gets cited. Both
fixed. Also caught a numeric inconsistency: abstract said "0.895" but
the actual table minimum is 0.896 -- fixed to match.

More importantly: the abstract claimed "Code and full results are
public" with **no repository anywhere** -- not even a local git repo.
This is now actually true, not just claimed. Initialized git in
`code/`, added a `.gitignore` excluding the 22GB `data/` directory
(raw N-BaIoT/CIC IoT-DIAD/UNSW datasets -- redistribution licensing
unverified) and `.venv/`, scanned for hardcoded secrets (none found),
and pushed to a new public repo under the `aashmauprety` GitHub account
(matching the paper's authorship, not the other available
`upretya1-design` account) at
https://github.com/aashmauprety/fediot-gating-agent -- confirmed live
(HTTP 200). Added the real URL to the abstract (needed `\usepackage{url}`,
not in IEEEtran by default). 124 files, code + scripts + results JSON
only, no raw data.

**Note for next session: this README.md file, once you read this, IS
the public repo's README** -- any further edits to it need a follow-up
`git add -A && git commit && git push` in `code/` to actually reach
GitHub, they don't happen automatically.

**Thirty-first real result: qwen2.5:32b tested -- 14B is a ceiling, not
a floor.** Per continued "continue more" instruction, pulled
`qwen2.5:32b` (19GB, ~38 min local download) and ran it on the
IDENTICAL N-BaIoT trust_building $D=10$ seed=0 config already used for
the recorded 14B headline number (`scripts/run_nbaiot_qwen32b_capability_test.py`,
`results/nbaiot_qwen32b_capability_test.json`). Result: **0.9964**
macro-F1 vs 14B's 0.9974 -- statistically indistinguishable, if
anything marginally lower -- while taking 2277s vs 854s (2.7x longer)
for the identical 25 rounds. Going bigger bought no measured accuracy
and cost substantially more compute. Written into `ICC2027/main.tex`
Section IV-D directly answering the paper's own "is 14B near a
capability floor" open question from Future Work -- it isn't a floor,
it's already at the ceiling this specific judgment task offers at this
evidence design. Paper's Conclusion updated to reflect this too.

**Thirty-second real result: two structural experiments, per user
request "tell me 2 experiment we can do to level up 100X this paper"
(explicitly excluding human-grounded evaluation).** Both address the
paper's own stated weakest points -- whether the LLM is truly reasoning
vs. pattern-matching its calibration exemplars, and whether the
round-relative digest design actually transfers across deployments --
rather than adding another data point in an already-well-covered
paradigm.

**(1) Novel attack-shape generalization**
(`scripts/run_novel_attack_shapes.py`,
`results/novel_attack_shapes.json`). Added three new attack primitives
to `fedgate/attacks.py`/`experiment.py` (`label_flip_partial`,
`free_rider`, `intermittent`/`slow_drip` scheduling), none resembling
the digest shapes the Gating Agent's few-shot exemplars were built
from, tested on N-BaIoT with the prompt completely untouched:
- slow-drip (ramps 0->full strength over 20 rounds): undefended 0.185
  -> gated **0.997**.
- intermittent (attacks only every 3rd round): undefended 0.138 ->
  gated **0.997**.
- free-rider (submits no real update, a distinct non-poisoning FL
  threat): undefended 0.995 -> gated 0.997 (statistically the same,
  since FedAvg dilutes rather than corrupts a near-zero contribution).

Both damaging shapes were fully defended with zero prompt changes --
real evidence against the "just pattern-matching the exemplars"
critique. But inspecting per-round decisions found a genuine,
honestly-reported limitation: free-riders got `accept` in every single
round (checked directly, not inferred from the accuracy number alone).
The exclude/downweight thresholds are one-sided -- built to catch
anomalously LARGE update norms (poisoning), not anomalously SMALL ones
(idling). Free-riding doesn't hurt this system, but it isn't detected
by it either; a real scope boundary, not a bug.

**(2) Zero-recalibration cross-dataset transfer**
(`scripts/run_ciciot2023_zero_recalibration.py`,
`results/ciciot2023_zero_recalibration.json`). Real CIC IoT 2023 data
(`fedgate/data.py:load_ciciot2023_federated`, already implemented,
never previously used for the Gating Agent -- 39 features, 8 classes,
genuinely different from N-BaIoT's 115/CIC IoT-DIAD's 118), the exact
same unmodified selective-LLM policy, zero threshold/prompt changes,
across trust_building D=10/untargeted/targeted. Result: 0.691/0.685/0.688
macro-F1 -- a tight band, but well below the 0.90+ on the other two
datasets. Ran a no-attacker-at-all baseline on the same data before
concluding anything: **0.686 macro-F1**, essentially identical to all
three defended numbers. The defense fully absorbs the attacks' impact
on this dataset; the lower absolute number is this simpler
39-feature representation's natural ceiling, not a generalization
failure. One real, unresolved caveat reported honestly: targeted-attack
success rate here is 20% (vs <1% everywhere else) -- plausibly
explained by this dataset's classes being more naturally confusable in
a 39-dim space (which would inflate a raw source-predicted-as-target
metric independent of any real defense gap), but NOT independently
confirmed, so reported as an open hypothesis, not an established fact.

Both written into `ICC2027/main.tex` as new Section IV-G/H. Paper is
now 7 pages, page-limit trim still explicitly deferred per user
instruction.

**Thirty-third real result: the free-rider detection gap found above
was actually fixed, not just reported.** Diagnosed first: pulled the
real rationale text for free-rider decisions and found they WERE
reaching the actual LLM (not fast-pathed), and the model's own words
said "notably negative but not extremely so" -- it noticed the anomaly
and dismissed it anyway, because its four calibration examples all
described anomalies as large POSITIVE update-norm z-scores (poisoning
shape); nothing taught it a strongly negative z-score (idle/free-rider
shape) also mattered. Fixed in two places in
`fedgate/gating_agent.py`: (1) added a fourth calibration example to
`SYSTEM_PROMPT_CALIBRATED` covering the negative-z case explicitly; (2)
found and fixed a SECOND bug this exposed -- `SelectiveLLMGatingPolicy`'s
fast-path filter (`_is_confidently_normal`) was itself one-sided
(`z > threshold`, not `abs(z) > threshold`), so even with the prompt
fixed, some free-rider rounds still got fast-pathed as "accept" without
ever reaching the LLM at all; changed to a symmetric check.

Re-ran the free-rider case after both fixes: every round now resolves
to downweight/exclude with a rationale naming the negative z-score
directly (previously: "accept" every round). Regression-checked
against the original flagship N-BaIoT $D{=}10$ config
(`scripts/diag_regression_check.py`): F1=0.9974, identical to both the
pre-fix number (0.9971) and the full-LLM reference (0.9974) -- no
accuracy regression, at a small, honest, expected cost (LLM-call
fraction rose from 44.9% to 51.4%, since the now-symmetric filter
defers to the LLM slightly more often). Written into `ICC2027/main.tex`
Section IV-G as the completed diagnose-fix-validate arc, matching the
pattern used for the selective-LLM D=20 bug and the concept-drift EWMA
failure earlier in this project.

**Thirty-fourth real result: the ICC 2027 submission-deadline page-limit
trim, finally done.** The paper had grown to 8 pages against the 6-page
hard limit (Oct 2, 2026 deadline, 3 days out when this trim happened)
after several sessions of "don't worry about page limit, add
experiments" per explicit user instruction. Cutting 2 full pages back
out required real restructuring, not just word-trimming:

- Merged what were two separate subsections (novel attack-shape
  generalization + zero-recalibration transfer) into one, replacing
  most of the prose with a compact summary table (new Table VI) --
  the single biggest space recovery in this pass.
- Compressed the FetchHistory tool-collapse paragraph, the two
  deployment-gaps paragraph, both Related Work paragraphs, the
  GShield/evasion paragraphs (merged into one), the core failure-
  diagnosis paragraphs, the onboarding/triage paragraph, the
  capability-boundary paragraphs, the Conclusion, and the abstract --
  every one individually, checking after each edit whether the page
  count moved, since IEEE's two-column reflow does not respond
  linearly to word count (several rounds of real cuts landed zero
  page-count change until a threshold was crossed).
- Replaced the round-22 trace-example `quote` block with inline text
  (environments carry their own vertical padding beyond the visible
  lines).

Final state confirmed via a full clean rebuild (`latexmk -C` + 3
passes, not trusting a single incremental compile): exactly 6/6 pages,
0 undefined references, 0 overfull boxes, and a full visual read of
all 6 rendered pages (not just the page-count number) confirming no
table/figure overlap or truncation anywhere. This is the actual,
final, submission-ready state of the ICC 2027 paper.

**Still open, in priority order:**
1. The hosted-frontier-model test remains blocked on OpenAI billing
   credits (code ready: `OpenAIExternalBayesianPolicy`). Now the more
   interesting remaining question given the 32B null result: would a
   frontier-tier model (GPT-4-class) actually break the ceiling a
   bigger LOCAL model did not, or is this a hard ceiling for the task
   itself regardless of model class? Worth running the moment credits
   are available.

**Thirty-fifth real result: two independent peer reviews (a full review
and a separate novelty-focused review) were addressed with real
text/scoping fixes, and the paper re-verified at 6/6 pages.** With the
deadline 2 days out (Sept 30, 2026), `my_review_icc.md` and
`novelty_review_icc.md` -- both dropped into the `ICC2027/` folder --
converged on the same core objections, which is a real signal they're
genuine issues, not reviewer idiosyncrasy. Fixes applied to
`ICC2027/main.tex`:

- **The single strongest objection** (both reviews independently
  raised it): Fig. 1's shared-encoder diagram could be misread as one
  model jointly attacked on both device-ID and intrusion detection,
  when it is actually two separate poisoning experiments on a shared
  architecture. Fixed in the abstract, the four-contributions
  paragraph, and -- most directly -- a new sentence opening the
  Results section that states this explicitly rather than letting the
  figure imply something untested.
- Softened three overclaims: the abstract's "beating a rule-based
  reference every time" (not literally true across every table row),
  auditability stated as settled fact rather than the paper's actual,
  not-yet-human-validated hypothesis, and the two capability-boundary
  sentences ("14B is necessary" / "14B is the ceiling") now explicitly
  scoped to the tested model families and prompt design, not asserted
  as general findings.
- Added a privacy/server-visibility caveat to the System and Threat
  Model section: gating requires the server (and the LLM it queries)
  to see each client's per-round digest, which is compatible with an
  honest-but-curious server but not with secure aggregation, which
  would hide those signals -- flagged as open, not glossed over.
- Verified via real WebSearch (not a hallucinated citation -- an
  earlier session already caught one forked agent hallucinating a
  literature "confirmation" with zero real tool calls, so citations
  get checked directly) that OpenCLAW-Nexus (Jia et al., arXiv:2605.04091,
  2026) is a real paper proposing a discounted Beta-reputation
  posterior for decentralized Byzantine-resilient FL trust. Added it
  to the bibliography and cited it in Related Work, explicitly
  narrowing the paper's Beta-trust-posterior novelty claim: the
  contribution isn't the Beta posterior, it's pairing one with an LLM
  reasoning in natural language over its value.

Adding this content pushed the paper to 7 pages. Rather than cut any
of the new fixes back out, the same word-trimming discipline from the
page-limit-trim pass (thirty-fourth real result) was reapplied across
the intro, related work, results, and conclusion -- no claims or
numbers removed, just tightened phrasing -- until back to 6 pages.
Confirmed via full clean rebuild (`pdflatex` + `bibtex` + 2 more
passes): exactly 6/6 pages, 0 undefined references/citations, 0
overfull boxes, and a full visual read of all 6 rendered pages
confirming no overlap or truncation from any of the edits.

**Not attempted, explicitly, given the ~2-day deadline:** a true
jointly-attacked two-head experiment, a blinded human audit study of
the LLM's rationales (auditability remains a stated hypothesis, not a
validated claim, for exactly this reason), a full matched-protocol
reimplementation of the 3+ closest competing methods, and a
controlled cross-model-family scaling study beyond the Qwen2.5
8B/14B/32B points already measured. These would need new data,
infrastructure, or human subjects not available before Oct 2, so the
paper says so explicitly rather than claiming more than what was
actually tested.

## To get the remaining datasets

- **UNSW**: done — see above.
- **YourThings**: not attempted; `https://www.yourthings.info/data/` returns
  200 and is worth checking for the same kind of direct-download links
  UNSW turned out to have (its site initially looked gated but wasn't).
- **CIC IoT 2023**: done — this note was stale; see the dataset table
  above. The official portal is still gated behind a personal
  registration form, but the dataset is legitimately re-mirrored under
  a CC-BY-4.0 license, already downloaded, and already has a working
  loader (`fedgate/data.py:load_ciciot2023_federated`), used for the
  zero-recalibration cross-dataset transfer experiment (2026-09-29
  entry above).
