"""Data loading for FedIoT-Gate experiments.

Two real data sources are supported:

1. N-BaIoT (real, downloaded from UCI ML repository into
   code/data/raw/nbaiot/<Device>/{benign_traffic.csv, gafgyt_attacks/*.csv,
   mirai_attacks/*.csv}). Each of the 9 devices is treated as one federated
   client -- a natural, realistic FL partition since each device really is
   a physically separate gateway/host in the original data collection.
   Features are the 115 pre-extracted statistical flow features N-BaIoT
   ships with; there is no raw packet-length/direction sequence available
   for this dataset, so it feeds the attack-type head directly (bypassing
   the Phase 0 raw-sequence encoder), consistent with the paper's Section
   VII-C protocol.

2. UNSW (real, downloaded directly from iotanalytics.unsw.edu.au into
   code/data/raw/unsw/{List_Of_Devices.txt, csv_extracted/*.csv}). This is
   the actual per-packet capture the paper describes in Section VII
   (Sivanathan et al.): each row is one packet with a timestamp, size,
   source/destination MAC and IP, protocol, and ports. We reconstruct the
   signed packet-length/direction sequence per device exactly as Section
   V-B specifies (positive = incoming, i.e. the device's MAC is eth.dst;
   negative = outgoing, i.e. the device's MAC is eth.src), chunk into
   windows of N packets, and use this as REAL raw-sequence data for both
   Phase 0 (self-supervised encoder) and the Phase I device-type head.
   YourThings and CIC IoT 2023 are still not obtained (CIC IoT 2023 gates
   downloads behind a personal registration form -- name/email/institution
   -- which this code does not and should not submit automatically; see
   code/README.md).

3. A synthetic UNSW-style generator (`make_synthetic_unsw`), kept only as
   a fallback code-validation tool now that real UNSW data is available.
   Do not report numbers from it as paper results.
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
NBAIOT_DIR = os.path.join(RAW_DIR, "nbaiot")
UNSW_DIR = os.path.join(RAW_DIR, "unsw")
UNSW_DEVICE_LIST = os.path.join(UNSW_DIR, "List_Of_Devices.txt")
UNSW_CSV_DIR = os.path.join(UNSW_DIR, "csv_extracted")
YOURTHINGS_DIR = os.path.join(RAW_DIR, "yourthings")
YOURTHINGS_MANUF = os.path.join(YOURTHINGS_DIR, "manuf.tsv")
YOURTHINGS_DEVICE_MAPPING = os.path.join(YOURTHINGS_DIR, "device_mapping.csv")
CICIOT_DIAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "labeled_data")
CICIOT_DIAD_DEVICE_TASK = os.path.join(CICIOT_DIAD_DIR, "device_task.parquet")
CICIOT_DIAD_ATTACK_TASK = os.path.join(CICIOT_DIAD_DIR, "attack_task.parquet")
CICIOT2023_DIR = os.path.join(RAW_DIR, "ciciot2023")
CICIOT2023_TRAIN = os.path.join(CICIOT2023_DIR, "train.parquet")
CICIOT2023_TEST = os.path.join(CICIOT2023_DIR, "test.parquet")

NBAIOT_DEVICES = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]

# Coarse attack-family label used as the attack-type head's target.
# (benign, gafgyt, mirai) -- a 3-way task. A finer-grained variant
# (per attack subtype, e.g. gafgyt.combo vs gafgyt.udp) is left as a
# TODO in the paper (Section VII-C) and can be enabled via
# `fine_grained=True` below.
COARSE_CLASSES = ["benign", "gafgyt", "mirai"]


@dataclass
class ClientData:
    client_id: str
    X: np.ndarray            # (n_samples, n_features) float32
    y_attack: np.ndarray     # (n_samples,) int, index into class list
    y_device: np.ndarray     # (n_samples,) int, constant per client here
    classes: List[str] = field(default_factory=lambda: list(COARSE_CLASSES))


def _read_csv_sample(path: str, n_max: int, seed: int) -> pd.DataFrame:
    """Read a CSV, subsampling to at most n_max rows for tractable training.

    N-BaIoT files range from ~10k to ~1M+ rows; full-scale training on all
    of them is unnecessary for validating the pipeline and would make an
    overnight run infeasible on a laptop. n_max should be raised for a
    final paper run once compute budget allows.
    """
    df = pd.read_csv(path)
    if len(df) > n_max:
        df = df.sample(n=n_max, random_state=seed)
    return df


def load_nbaiot_client(
    device: str,
    n_max_per_class: int = 4000,
    fine_grained: bool = False,
    seed: int = 0,
) -> ClientData:
    """Load one N-BaIoT device as one federated client.

    Returns features (raw, unnormalized -- caller should standardize using
    train-split statistics) and a coarse attack-type label
    {benign, gafgyt, mirai}, or a fine-grained per-subtype label if
    fine_grained=True.
    """
    device_dir = os.path.join(NBAIOT_DIR, device)
    frames = []
    labels = []

    benign_path = os.path.join(device_dir, "benign_traffic.csv")
    if os.path.exists(benign_path):
        df = _read_csv_sample(benign_path, n_max_per_class, seed)
        frames.append(df)
        labels += ["benign"] * len(df)

    for family in ("gafgyt", "mirai"):
        family_dir = os.path.join(device_dir, f"{family}_attacks")
        if not os.path.isdir(family_dir):
            continue
        subfiles = sorted(glob.glob(os.path.join(family_dir, "*.csv")))
        # split the per-class budget evenly across attack subtypes so e.g.
        # gafgyt isn't dominated by whichever subtype file happens largest
        per_subtype = max(1, n_max_per_class // max(1, len(subfiles)))
        for sf in subfiles:
            df = _read_csv_sample(sf, per_subtype, seed)
            frames.append(df)
            if fine_grained:
                subtype = os.path.splitext(os.path.basename(sf))[0]
                labels += [f"{family}.{subtype}"] * len(df)
            else:
                labels += [family] * len(df)

    X = pd.concat(frames, axis=0, ignore_index=True).values.astype(np.float32)
    class_list = sorted(set(labels)) if fine_grained else list(COARSE_CLASSES)
    label_to_idx = {c: i for i, c in enumerate(class_list)}
    y_attack = np.array([label_to_idx[l] for l in labels], dtype=np.int64)
    device_idx = NBAIOT_DEVICES.index(device)
    y_device = np.full(len(y_attack), device_idx, dtype=np.int64)

    # shuffle
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y_attack))
    return ClientData(
        client_id=device,
        X=X[perm],
        y_attack=y_attack[perm],
        y_device=y_device[perm],
        classes=class_list,
    )


def load_nbaiot_federated(
    n_max_per_class: int = 4000,
    fine_grained: bool = False,
    seed: int = 0,
    devices: List[str] | None = None,
) -> Dict[str, ClientData]:
    """Load every available N-BaIoT device as a federated client dict."""
    devices = devices or [
        d for d in NBAIOT_DEVICES if os.path.isdir(os.path.join(NBAIOT_DIR, d))
    ]
    return {d: load_nbaiot_client(d, n_max_per_class, fine_grained, seed) for d in devices}


def train_val_test_split(
    client: ClientData, train_frac=0.8, val_frac=0.1, seed: int = 0
) -> Tuple[ClientData, ClientData, ClientData]:
    n = len(client.y_attack)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    def _sub(idx):
        return ClientData(
            client_id=client.client_id,
            X=client.X[idx],
            y_attack=client.y_attack[idx],
            y_device=client.y_device[idx],
            classes=client.classes,
        )

    return _sub(idx_train), _sub(idx_val), _sub(idx_test)


# ---------------------------------------------------------------------------
# Real UNSW packet-capture loader (Sivanathan et al.)
# ---------------------------------------------------------------------------

# Non-IoT devices excluded from device-type classification, matching the
# paper's Section VII treatment (routers/laptops/phones/tablets are not
# IoT device-type classes). MAC addresses from List_Of_Devices.txt.
UNSW_NON_IOT_NAME_SUBSTRINGS = (
    "laptop", "macbook", "phone", "tablet", "router", "ipad", "printer",
    "galaxy tab", "pixel", "iphone", "samsunggalaxytab",
)


def parse_unsw_device_list(path: str = UNSW_DEVICE_LIST) -> Dict[str, str]:
    """Parse List_Of_Devices.txt into {mac_address_lowercase: device_name}."""
    mac_to_name: Dict[str, str] = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.strip().startswith("List of Devices"):
                continue
            # Format: "<name...>\t<mac>\t<connection type>" with irregular
            # whitespace; a MAC address is the identifying regex.
            import re

            m = re.search(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})", line)
            if not m:
                continue
            mac = m.group(1).lower()
            name = line[: m.start()].strip()
            if not name:
                continue
            mac_to_name[mac] = name
    return mac_to_name


def unsw_iot_devices(path: str = UNSW_DEVICE_LIST) -> Dict[str, str]:
    """Device list filtered to IoT devices only (drops laptops/phones/etc.)."""
    all_devices = parse_unsw_device_list(path)
    out = {}
    for mac, name in all_devices.items():
        lname = name.lower()
        if any(s in lname for s in UNSW_NON_IOT_NAME_SUBSTRINGS):
            continue
        out[mac] = name
    return out


def build_unsw_packet_sequences(
    mac_to_name: Dict[str, str],
    csv_dir: str = UNSW_CSV_DIR,
    seq_len: int = 100,
    max_windows_per_device: int = 2000,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Stream every extracted UNSW day-CSV once, accumulate signed
    packet-length/direction sequences per device MAC, and chunk into
    fixed-length windows (Section V-B). Returns {device_name: (n_windows,
    seq_len) float32 array}.

    Streams file-by-file with only the needed columns to keep memory
    bounded (the full concatenated CSVs are ~21M rows / ~2GB); only rows
    touching a MAC in mac_to_name are kept.
    """
    macs = set(mac_to_name.keys())
    # accumulate (time, signed_size) tuples per device across all files
    per_device_records: Dict[str, List[Tuple[float, float]]] = {m: [] for m in macs}

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    for path in csv_files:
        for chunk in pd.read_csv(
            path,
            usecols=["TIME", "Size", "eth.src", "eth.dst"],
            dtype={"TIME": np.float64, "Size": np.float32, "eth.src": str, "eth.dst": str},
            chunksize=500_000,
        ):
            chunk = chunk.rename(columns={"eth.src": "src", "eth.dst": "dst"})
            chunk["src"] = chunk["src"].str.lower()
            chunk["dst"] = chunk["dst"].str.lower()

            src_mask = chunk["src"].isin(macs)
            if src_mask.any():
                sub = chunk.loc[src_mask, ["TIME", "Size", "src"]]
                for mac, group in sub.groupby("src"):
                    per_device_records[mac].extend(
                        zip(group["TIME"].tolist(), (-group["Size"]).tolist())
                    )

            dst_mask = chunk["dst"].isin(macs) & (chunk["dst"] != chunk["src"])
            if dst_mask.any():
                sub = chunk.loc[dst_mask, ["TIME", "Size", "dst"]]
                for mac, group in sub.groupby("dst"):
                    per_device_records[mac].extend(
                        zip(group["TIME"].tolist(), group["Size"].tolist())
                    )

    rng = np.random.RandomState(seed)
    device_windows: Dict[str, np.ndarray] = {}
    for mac, records in per_device_records.items():
        name = mac_to_name[mac]
        if not records:
            continue
        records.sort(key=lambda r: r[0])
        signed_sizes = np.array([r[1] for r in records], dtype=np.float32)
        n_windows = len(signed_sizes) // seq_len
        if n_windows == 0:
            continue
        windows = signed_sizes[: n_windows * seq_len].reshape(n_windows, seq_len)
        if n_windows > max_windows_per_device:
            idx = rng.choice(n_windows, size=max_windows_per_device, replace=False)
            windows = windows[idx]
        if name in device_windows:
            device_windows[name] = np.concatenate([device_windows[name], windows], axis=0)
        else:
            device_windows[name] = windows
    return device_windows


def load_unsw_device_identification(
    seq_len: int = 100,
    max_windows_per_device: int = 2000,
    min_windows_per_device: int = 20,
    n_clients: int = 10,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Real UNSW device-identification dataset: builds per-device raw
    packet-length/direction windows (see build_unsw_packet_sequences),
    drops devices with too little traffic to be usable (mirrors the
    paper's exclusion of Withings Smart scale / Blipcare / NEST Protect
    for the same reason), then evenly distributes the pooled windows
    across `n_clients` simulated federated clients, exactly as the paper
    describes ("these feature vectors were then distributed to clients to
    simulate a FL environment").

    Returns {client_id: {"X": (n,seq_len) float32, "y_device": (n,) int64}}
    plus the label list is recoverable from the returned "classes" key on
    the first client's dict for convenience.
    """
    mac_to_name = unsw_iot_devices()
    device_windows = build_unsw_packet_sequences(
        mac_to_name, seq_len=seq_len, max_windows_per_device=max_windows_per_device, seed=seed
    )
    device_windows = {
        name: w for name, w in device_windows.items() if len(w) >= min_windows_per_device
    }
    classes = sorted(device_windows.keys())
    label_to_idx = {c: i for i, c in enumerate(classes)}

    all_X, all_y = [], []
    for name, windows in device_windows.items():
        all_X.append(windows)
        all_y.append(np.full(len(windows), label_to_idx[name], dtype=np.int64))
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    clients = {}
    splits_X = np.array_split(X, n_clients)
    splits_y = np.array_split(y, n_clients)
    for i in range(n_clients):
        clients[f"unsw_client_{i}"] = {
            "X": splits_X[i], "y_device": splits_y[i], "classes": classes,
        }
    return clients


# ---------------------------------------------------------------------------
# Real YourThings packet-capture loader
# ---------------------------------------------------------------------------
#
# Unlike UNSW, YourThings ships raw pcap files (5-minute chunks inside a
# per-day .tgz, e.g. code/data/raw/yourthings/iot_traffic<date>.tgz).
# code/data/raw/yourthings/process_day.sh (or process_day_ip.sh, which
# also extracts ip.src/ip.dst) extracts each chunk with `tar -xzO`, runs
# `tshark -T fields` to produce a CSV, and discards the raw pcap
# immediately, since a full day's chunks extracted at once would be
# ~15-20GB.
#
# STATUS UPDATE: yourthings.info/data/ turns out to publish a real,
# verified device-name-to-local-IP mapping (device_mapping.csv, 66
# devices, e.g. "SamsungSmartThingsHub,192.168.0.4") that an earlier pass
# through this codebase missed, and instead inferred device identity from
# the Ethernet OUI (vendor prefix of the MAC) via tshark's own vendor
# database (`tshark -G manuf`, saved to manuf.tsv) matched against a
# hand-picked list of target vendor names -- a weaker method that
# identifies a device's *manufacturer*, not a verified device *model*,
# and covered only 3 target devices. `load_yourthings_device_identification`
# now prefers IP-based identification against the real mapping whenever a
# CSV has ip.src/ip.dst columns (i.e. was produced by process_day_ip.sh),
# covering up to all 66 mapped devices with verified identity rather than
# vendor guesses; it falls back to the old vendor-OUI method only for
# CSVs that lack IP columns (i.e. produced by the original process_day.sh).
# The vendor-OUI path and its caveat below are kept for that fallback case
# and for historical honesty about how the earlier (weaker) result in the
# paper was obtained -- see the paper's cross-network onboarding section
# for both.
#
# One real caveat on the IP mapping itself: local IPs are typically
# DHCP-assigned, so this mapping is only trustworthy for capture days
# within the same testbed period it was published for; we do not have
# independent per-day confirmation that a given device kept the same IP
# on every capture day, only that the mapping is a real, documented
# artifact of the dataset's own release (not vendor guessing) for the
# period it covers.
# so if a household has e.g. multiple Samsung products on the same
# network, they cannot be told apart -- and should be reported as such.

YOURTHINGS_VENDOR_TO_DEVICE = {
    "samsung": "Samsung Smart Things Hub",
    "philips": "Philips Hue Hub",
    "bose": "Bose SoundTouch 10",
}


def load_oui_vendor_map(path: str = YOURTHINGS_MANUF) -> Dict[str, str]:
    """Parse `tshark -G manuf` output into {24-bit OUI prefix (lowercase,
    'xx:xx:xx'): vendor short name}. Longer (28/36-bit) OUI ranges in the
    file are skipped for simplicity -- adequate for identifying the
    handful of vendors we're looking for."""
    oui_to_vendor: Dict[str, str] = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            prefix, vendor = parts[0].strip(), parts[1].strip()
            if "/" in prefix:  # e.g. "00:00:00:00:00:00/28" -- skip non-24-bit ranges
                continue
            if prefix.count(":") != 2:
                continue
            oui_to_vendor[prefix.lower()] = vendor
    return oui_to_vendor


def identify_yourthings_devices(
    mac_series: "pd.Series", oui_map: Dict[str, str]
) -> Dict[str, str]:
    """Map each observed MAC to a device name via vendor-prefix matching
    against YOURTHINGS_VENDOR_TO_DEVICE. Returns {mac: device_name} only
    for MACs whose vendor matched one of the target vendors."""
    result = {}
    for mac in mac_series.unique():
        if not isinstance(mac, str) or mac.count(":") != 5:
            continue
        prefix = mac.lower()[:8]
        vendor = oui_map.get(prefix, "")
        vendor_lower = vendor.lower()
        for key, device_name in YOURTHINGS_VENDOR_TO_DEVICE.items():
            if key in vendor_lower:
                result[mac.lower()] = device_name
                break
    return result


def find_busiest_mac_per_device(
    csv_paths: List[str], oui_map: Dict[str, str], min_packets: int = 1000
) -> Dict[str, str]:
    """First pass: count packet occurrences per vendor-matched MAC across
    all CSVs, and keep only the single busiest MAC per target device name
    (above `min_packets`, to drop noise -- e.g. a phone or TV that happens
    to share a vendor prefix but appears only a handful of times). This is
    what keeps a household with, say, two Samsung products from having
    their traffic silently merged into one "Samsung Smart Things Hub"
    label -- see the module docstring's vendor-vs-model caveat."""
    counts: Dict[Tuple[str, str], int] = {}
    for path in csv_paths:
        for chunk in pd.read_csv(
            path, dtype=str, usecols=["eth.src", "eth.dst"], chunksize=1_000_000
        ):
            chunk = chunk.dropna(subset=["eth.src", "eth.dst"])
            macs = pd.concat([chunk["eth.src"], chunk["eth.dst"]]).str.lower()
            mapping = identify_yourthings_devices(macs, oui_map)
            for mac, device_name in mapping.items():
                key = (mac, device_name)
                counts[key] = counts.get(key, 0) + int((macs == mac).sum())

    best_mac_per_device: Dict[str, Tuple[str, int]] = {}
    for (mac, device_name), cnt in counts.items():
        if cnt < min_packets:
            continue
        if device_name not in best_mac_per_device or cnt > best_mac_per_device[device_name][1]:
            best_mac_per_device[device_name] = (mac, cnt)

    return {device_name: mac for device_name, (mac, cnt) in best_mac_per_device.items()}


def load_yourthings_device_mapping(path: str = YOURTHINGS_DEVICE_MAPPING) -> Dict[str, str]:
    """Parse the real, dataset-published device_mapping.csv into
    {device_name: local_ip}. This is verified ground truth from
    yourthings.info/data/, not a vendor-prefix guess -- see module
    docstring."""
    mapping: Dict[str, str] = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            name, ip = line.rsplit(",", 1)
            mapping[name.strip()] = ip.strip()
    return mapping


def find_active_ip_per_device(
    csv_paths: List[str], device_mapping: Dict[str, str], min_packets: int = 1000
) -> Dict[str, str]:
    """Second pass, IP-based identification: for each device in the real
    device_mapping.csv, count packets to/from its mapped IP across all
    CSVs and keep only devices with at least `min_packets` -- i.e. the
    device was actually present and active in this particular capture,
    not just listed in the mapping. Unlike find_busiest_mac_per_device,
    there is no "busiest MAC" ambiguity to resolve here: the mapping
    already gives one authoritative IP per device name."""
    ip_to_device = {ip: name for name, ip in device_mapping.items()}
    target_ips = set(ip_to_device.keys())
    counts: Dict[str, int] = {ip: 0 for ip in target_ips}

    for path in csv_paths:
        if "ip.src" not in pd.read_csv(path, nrows=0).columns:
            continue
        for chunk in pd.read_csv(
            path, dtype=str, usecols=["ip.src", "ip.dst"], chunksize=1_000_000
        ):
            chunk = chunk.dropna(subset=["ip.src", "ip.dst"], how="all")
            src_counts = chunk["ip.src"].value_counts()
            dst_counts = chunk["ip.dst"].value_counts()
            for ip in target_ips:
                counts[ip] += int(src_counts.get(ip, 0)) + int(dst_counts.get(ip, 0))

    return {
        ip_to_device[ip]: ip for ip, cnt in counts.items() if cnt >= min_packets
    }


def load_yourthings_device_identification(
    csv_paths: List[str] = None,
    seq_len: int = 100,
    max_windows_per_device: int = 2000,
    min_windows_per_device: int = 5,
    n_clients: int = 3,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Real YourThings device-identification dataset, built from
    tshark-extracted packet CSVs (see module docstring). For each CSV,
    prefers real IP-based identification against the dataset's own
    device_mapping.csv (verified ground truth) when the CSV has
    ip.src/ip.dst columns (i.e. was produced by process_day_ip.sh);
    falls back to the older vendor-prefix MAC match (find_busiest_mac_per_device)
    for CSVs without IP columns. Builds signed packet-length/direction
    windows exactly as for UNSW, and evenly distributes them across
    `n_clients` simulated federated clients.
    """
    if csv_paths is None:
        csv_paths = sorted(glob.glob(os.path.join(YOURTHINGS_DIR, "packets_*.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No YourThings packet CSVs found in {YOURTHINGS_DIR}. "
            "Run process_day.sh or process_day_ip.sh on a downloaded .tgz first."
        )

    ip_csv_paths = [p for p in csv_paths if "ip.src" in pd.read_csv(p, nrows=0).columns]
    mac_csv_paths = [p for p in csv_paths if p not in ip_csv_paths]

    per_device_records: Dict[str, List[Tuple[float, float]]] = {}

    # --- IP-based identification (real, verified ground truth) ---
    if ip_csv_paths:
        device_mapping = load_yourthings_device_mapping()
        device_to_ip = find_active_ip_per_device(ip_csv_paths, device_mapping)
        ip_to_device = {ip: name for name, ip in device_to_ip.items()}
        print(f"YourThings (IP-verified): {len(device_to_ip)} devices active: {device_to_ip}")
        ips = set(ip_to_device.keys())
        for name in device_to_ip:
            per_device_records.setdefault(name, [])

        for path in ip_csv_paths:
            for chunk in pd.read_csv(
                path,
                dtype={"TIME": np.float64, "Size": np.float32, "ip.src": str, "ip.dst": str},
                usecols=["TIME", "Size", "ip.src", "ip.dst"],
                chunksize=500_000,
            ):
                chunk = chunk.dropna(subset=["ip.src", "ip.dst"], how="all")
                if not (chunk["ip.src"].isin(ips).any() or chunk["ip.dst"].isin(ips).any()):
                    continue
                src_mask = chunk["ip.src"].isin(ips)
                if src_mask.any():
                    sub = chunk.loc[src_mask, ["TIME", "Size", "ip.src"]]
                    for ip, group in sub.groupby("ip.src"):
                        name = ip_to_device[ip]
                        per_device_records[name].extend(
                            zip(group["TIME"].tolist(), (-group["Size"]).tolist())
                        )
                dst_mask = chunk["ip.dst"].isin(ips) & (chunk["ip.dst"] != chunk["ip.src"])
                if dst_mask.any():
                    sub = chunk.loc[dst_mask, ["TIME", "Size", "ip.dst"]]
                    for ip, group in sub.groupby("ip.dst"):
                        name = ip_to_device[ip]
                        per_device_records[name].extend(
                            zip(group["TIME"].tolist(), group["Size"].tolist())
                        )

    # --- Vendor-OUI MAC fallback (weaker, historical -- only for CSVs
    # that lack IP columns) ---
    if mac_csv_paths:
        oui_map = load_oui_vendor_map()
        device_to_mac = find_busiest_mac_per_device(mac_csv_paths, oui_map)
        mac_to_device = {mac: name for name, mac in device_to_mac.items()}
        print(f"YourThings (vendor-guess fallback, {len(mac_csv_paths)} CSV(s) without IP columns): "
              f"{device_to_mac}")
        for name in device_to_mac:
            per_device_records.setdefault(name, [])
        macs = set(mac_to_device.keys())

        for path in mac_csv_paths:
            for chunk in pd.read_csv(
                path,
                dtype={"TIME": np.float64, "Size": np.float32, "eth.src": str, "eth.dst": str},
                chunksize=500_000,
            ):
                chunk = chunk.dropna(subset=["eth.src", "eth.dst"])
                chunk["eth.src"] = chunk["eth.src"].str.lower()
                chunk["eth.dst"] = chunk["eth.dst"].str.lower()

                if not (chunk["eth.src"].isin(macs).any() or chunk["eth.dst"].isin(macs).any()):
                    continue

                src_mask = chunk["eth.src"].isin(macs)
                if src_mask.any():
                    sub = chunk.loc[src_mask, ["TIME", "Size", "eth.src"]]
                    for mac, group in sub.groupby("eth.src"):
                        name = mac_to_device[mac]
                        per_device_records[name].extend(
                            zip(group["TIME"].tolist(), (-group["Size"]).tolist())
                        )

                dst_mask = chunk["eth.dst"].isin(macs) & (chunk["eth.dst"] != chunk["eth.src"])
                if dst_mask.any():
                    sub = chunk.loc[dst_mask, ["TIME", "Size", "eth.dst"]]
                    for mac, group in sub.groupby("eth.dst"):
                        name = mac_to_device[mac]
                        per_device_records[name].extend(
                            zip(group["TIME"].tolist(), group["Size"].tolist())
                        )

    rng = np.random.RandomState(seed)
    device_windows: Dict[str, np.ndarray] = {}
    for name, records in per_device_records.items():
        if not records:
            continue
        records.sort(key=lambda r: r[0])
        signed_sizes = np.array([r[1] for r in records], dtype=np.float32)
        n_windows = len(signed_sizes) // seq_len
        if n_windows == 0:
            continue
        windows = signed_sizes[: n_windows * seq_len].reshape(n_windows, seq_len)
        if n_windows > max_windows_per_device:
            idx = rng.choice(n_windows, size=max_windows_per_device, replace=False)
            windows = windows[idx]
        device_windows[name] = windows

    device_windows = {
        name: w for name, w in device_windows.items() if len(w) >= min_windows_per_device
    }
    classes = sorted(device_windows.keys())
    label_to_idx = {c: i for i, c in enumerate(classes)}

    all_X, all_y = [], []
    for name, windows in device_windows.items():
        all_X.append(windows)
        all_y.append(np.full(len(windows), label_to_idx[name], dtype=np.int64))
    if not all_X:
        raise ValueError(
            "No target-vendor devices found in the processed YourThings CSVs -- "
            "check manuf.tsv and YOURTHINGS_VENDOR_TO_DEVICE."
        )
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    clients = {}
    splits_X = np.array_split(X, n_clients)
    splits_y = np.array_split(y, n_clients)
    for i in range(n_clients):
        clients[f"yourthings_client_{i}"] = {
            "X": splits_X[i], "y_device": splits_y[i], "classes": classes,
        }
    return clients


# ---------------------------------------------------------------------------
# Real CIC IoT 2023 loader
# ---------------------------------------------------------------------------
#
# The official CIC portal (cicresearch.ca) gates downloads behind a
# personal registration form -- see the code/README.md caveat about not
# auto-submitting that with fabricated identity details. However, the
# *data itself* is legitimately mirrored, CC-BY-4.0 licensed, ungated, on
# Hugging Face: `lacg030175/CIC-IoT-2023-canonical-neto`, traced back to
# Neto et al.'s original merged CSVs (bencorn/CIC-IoT-2023's
# CSV/MERGED_CSV/ folder) with ~45M rows preserved (vs. a smaller,
# folder-restructured 38.5M-row variant that lost rows). Downloaded here
# as two parquet files (train ~36M rows, test ~9M rows, 46 pre-extracted
# flow features + a normalized `attack_class` in
# {Benign, BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web-based}).
#
# Like N-BaIoT, this ships pre-extracted flow features, not raw packet
# sequences -- it cannot exercise the Phase 0 encoder either. Unlike
# N-BaIoT, this merged/flattened format has NO per-device identifier
# column, so there is no natural federated-client partition the way
# N-BaIoT's 9 physical devices or UNSW's per-day captures provide; clients
# here are a random partition of the pooled rows, which is a real
# limitation (it cannot model genuine device-level heterogeneity) that
# should be disclosed alongside any result computed with it.

CICIOT2023_CLASSES = [
    "Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based",
]


def load_ciciot2023_federated(
    n_per_class: int = 4000,
    n_clients: int = 9,
    split: str = "train",
    seed: int = 0,
) -> Dict[str, ClientData]:
    """Load a class-balanced sample of real CIC IoT 2023 flow features and
    partition it randomly across `n_clients` simulated federated clients
    (see module docstring for why this can't be a device-level partition
    for this particular mirror/format). Returns the same ClientData shape
    `load_nbaiot_federated` does, so it's a drop-in alternative/extension
    for the attack-type head with a richer 8-class label space.

    IMPORTANT: this mirror's README explicitly documents that rows with
    NaN or +-Inf feature values are deliberately preserved (not dropped),
    recommending a `ThermometerEncoder(invalid_encoding="single_bit")` to
    turn each invalid value into a learnable per-feature flag rather than
    silently zeroing it. That encoder is not implemented here. Instead,
    this loader drops any row containing a NaN/Inf feature value before
    sampling -- simpler, but it does discard real (if rare) data and,
    unlike the recommended approach, cannot let a model learn from the
    fact that a value was invalid. This matters in practice: an earlier,
    undocumented version of this loader did NOT filter these rows, and a
    federated run at n_per_class=4000 silently collapsed to a frozen,
    near-random macro-F1 from round 0 onward (NaN propagating through
    every weight after the first backward pass) -- caught only by
    noticing the reported F1 was *below* random-guessing baseline and was
    bit-identical every round, both of which should be treated as red
    flags, not just "a hard dataset," when they show up in your own runs.
    """
    import pyarrow.parquet as pq

    path = CICIOT2023_TRAIN if split == "train" else CICIOT2023_TEST
    pf = pq.ParquetFile(path)
    feature_cols = [
        f.name for f in pf.schema_arrow
        if f.name not in ("Label", "Label_orig", "attack_class", "label")
    ]

    rng = np.random.RandomState(seed)
    class_frames = {c: [] for c in CICIOT2023_CLASSES}
    class_counts = {c: 0 for c in CICIOT2023_CLASSES}
    n_dropped_invalid = 0

    for batch in pf.iter_batches(columns=feature_cols + ["attack_class"], batch_size=200_000):
        df = batch.to_pandas()
        finite_mask = np.isfinite(df[feature_cols].values).all(axis=1)
        n_dropped_invalid += int((~finite_mask).sum())
        df = df.loc[finite_mask]
        for cls in CICIOT2023_CLASSES:
            if class_counts[cls] >= n_per_class:
                continue
            sub = df[df["attack_class"] == cls]
            if len(sub) == 0:
                continue
            need = n_per_class - class_counts[cls]
            if len(sub) > need:
                sub = sub.sample(n=need, random_state=seed)
            class_frames[cls].append(sub)
            class_counts[cls] += len(sub)
        if all(v >= n_per_class for v in class_counts.values()):
            break

    print(f"CIC IoT 2023: dropped {n_dropped_invalid} rows with NaN/Inf feature values while scanning")

    label_to_idx = {c: i for i, c in enumerate(CICIOT2023_CLASSES)}
    all_X, all_y = [], []
    for cls, frames in class_frames.items():
        if not frames:
            continue
        df = pd.concat(frames, axis=0, ignore_index=True)
        X = df[feature_cols].values.astype(np.float32)
        y = np.full(len(df), label_to_idx[cls], dtype=np.int64)
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    splits_X = np.array_split(X, n_clients)
    splits_y = np.array_split(y, n_clients)
    clients = {}
    for i in range(n_clients):
        clients[f"ciciot_client_{i}"] = ClientData(
            client_id=f"ciciot_client_{i}",
            X=splits_X[i],
            y_attack=splits_y[i],
            y_device=np.zeros(len(splits_y[i]), dtype=np.int64),  # no device info in this format
            classes=CICIOT2023_CLASSES,
        )
    return clients


# ---------------------------------------------------------------------------
# Real CIC IoT-DIAD 2024 loader
# ---------------------------------------------------------------------------
#
# Already staged locally at code/data/labeled_data/ (device_task.parquet,
# attack_task.parquet, sampled_full.parquet) by a prior local build step,
# not something this session downloaded. Unlike N-BaIoT and CIC IoT 2023
# (pre-extracted flow features, no raw packets) and UNSW/YourThings (raw
# packet-size sequences, no rich per-packet fields), this dataset gives a
# MUCH richer real per-packet/flow feature set: inter-arrival time, TTL,
# TCP window size, payload entropy, TLS handshake fields, HTTP/DNS/ICMP
# fields, and multi-window (1/5/10/30/60-packet) traffic statistics per
# stream/src_ip/src_ip+mac/channel -- exactly the kind of feature richness
# the paper's cross-network generalization investigation (Section VII-G)
# flagged as likely missing from our packet-size-only representation. It
# also, uniquely among our real attack-labeled data, carries REAL per-
# device identity (59 named devices) alongside attack labels in one place
# (device_task.parquet), unlike N-BaIoT/CIC IoT 2023 (attack-only, no
# joint device labels) or UNSW (device-only, no attack labels).
#
# The local build step (code/data/labeled_data/build_info.json) already
# dropped direct-identity/leakage columns before we ever touched this
# data: stream, src_mac, dst_mac, src_ip, dst_ip, device_mac,
# eth_src_oui, eth_dst_oui -- the same class of columns UNSW/YourThings
# loaders never expose to the model either. `source_file`, `row_in_file`,
# `block`, `split` are present but tagged as metadata, not features, in
# that same build_info.json; the loaders below exclude them from X.
#
# This first pass uses only the ~118 numeric ("double"-typed) feature
# columns; several string-valued columns (handshake_version, tls_server,
# http_request_method, http_host, user_agent, dns_server, highest_layer,
# http_uri, http_content_type) are NOT yet used -- some (http_host,
# user_agent, dns_server, tls_server) are high-cardinality free text that
# would need real encoding work (and a leakage check: a device that
# always talks to one cloud endpoint could make its hostname an identity
# shortcut rather than a behavioral signal) before being added; this is
# flagged as a real, documented gap, not silently dropped. Missing values
# in the numeric columns are common and largely MEANINGFUL, not
# missing-at-random (e.g. `icmp_type` is only defined for ICMP packets,
# `*_var` columns are undefined when a window's count is 1) -- filled
# with 0.0 here, a simple choice documented rather than hidden; a
# thermometer/invalid-flag encoding (as CIC IoT 2023's own README
# recommends for its NaNs) is a possible future improvement, not done
# here.

def load_ciciot_diad_numeric_feature_columns(path: str) -> List[str]:
    """The ~118 numeric feature columns in either DIAD parquet file,
    excluding int64 meta columns (row_in_file, block) and all string
    columns (meta + labels + the not-yet-used text/categorical features
    -- see module docstring)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return [f.name for f in pf.schema_arrow if str(f.type) == "double"]


def load_ciciot_diad_device_identification(
    path: str = CICIOT_DIAD_DEVICE_TASK,
    n_max_per_device: int = 3000,
    min_rows_per_device: int = 50,
    n_clients: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Real per-device federated device-identification data. Unlike
    UNSW's arbitrary 10-way partition of one pooled capture, this
    dataset's real per-row device labels let clients be genuine physical
    devices -- one client per real device, the same pattern N-BaIoT uses
    for its 9 devices, just with 59 here instead of 9. Returns
    {device_name: {"X": ..., "y_device": ..., "classes": [...]}}, X being
    the ~118 numeric features (module docstring), robustly scaled the
    same way experiment.py already scales N-BaIoT/CIC IoT 2023 features
    (median/IQR, computed by the caller -- this loader returns raw
    values, unscaled, consistent with the other loaders in this file).

    `n_clients`, if given, additionally groups the (up to 59) real
    per-device clients into `n_clients` buckets by concatenation, for
    experiments that want fewer, larger federated clients rather than
    one per device; default (None) keeps one client per real device.

    REAL BUG CAUGHT BY THIS CODEBASE'S OWN NEAR-RANDOM-F1 HEURISTIC: for
    a DEVICE-IDENTIFICATION task specifically, `n_clients=None` (one
    client per real device) is fatal, not just a valid design choice --
    every row within a given client then shares the SAME device label,
    so each client's local training trivially "solves" its own data by
    always predicting its one class, and FedAvg-ing 59 single-class
    specialists degenerates to a global model with no real
    discriminative signal (observed: macro-F1 frozen near 0.0006 for
    most rounds, final 0.011 -- effectively random for 59 classes,
    caught by the same "frozen/near-random F1 = bug signature" heuristic
    this codebase already uses for the CIC IoT 2023 NaN/Inf bug). This is
    NOT a bug in this loader itself -- it is architecturally different
    from N-BaIoT's per-device clients, which work fine because N-BaIoT's
    *label* (attack type) varies within each device's own data, so local
    training is meaningful there. For device identification, pass
    `n_clients` (e.g. 10) to bucket multiple real devices per client,
    mirroring UNSW's arbitrary-partition approach -- see
    code/scripts/run_ciciot_diad.py for the corrected usage and
    code/README.md for the full real before/after numbers.
    """
    import pyarrow.parquet as pq

    feature_cols = load_ciciot_diad_numeric_feature_columns(path)
    pf = pq.ParquetFile(path)

    rng = np.random.RandomState(seed)
    per_device_frames: Dict[str, List[pd.DataFrame]] = {}
    per_device_counts: Dict[str, int] = {}

    for batch in pf.iter_batches(columns=feature_cols + ["device", "device_type"], batch_size=200_000):
        df = batch.to_pandas()
        for device, sub in df.groupby("device"):
            cnt = per_device_counts.get(device, 0)
            if cnt >= n_max_per_device:
                continue
            need = n_max_per_device - cnt
            if len(sub) > need:
                sub = sub.sample(n=need, random_state=seed)
            per_device_frames.setdefault(device, []).append(sub)
            per_device_counts[device] = cnt + len(sub)

    devices = sorted(per_device_frames.keys())
    classes = devices
    label_to_idx = {d: i for i, d in enumerate(devices)}

    per_device_X: Dict[str, np.ndarray] = {}
    per_device_y: Dict[str, np.ndarray] = {}
    for device, frames in per_device_frames.items():
        df = pd.concat(frames, axis=0, ignore_index=True)
        if len(df) < min_rows_per_device:
            continue
        X = df[feature_cols].fillna(0.0).values.astype(np.float32)
        y = np.full(len(df), label_to_idx[device], dtype=np.int64)
        per_device_X[device] = X
        per_device_y[device] = y

    print(f"CIC IoT-DIAD 2024 device-ID: {len(per_device_X)} real devices with "
          f">= {min_rows_per_device} rows (of {len(devices)} total in this sample)")

    if n_clients is None:
        return {
            device: {"X": per_device_X[device], "y_device": per_device_y[device], "classes": classes}
            for device in per_device_X
        }

    # Bucket the real per-device clients into n_clients groups.
    device_names = list(per_device_X.keys())
    rng.shuffle(device_names)
    buckets = np.array_split(device_names, n_clients)
    clients = {}
    for i, bucket in enumerate(buckets):
        if len(bucket) == 0:
            continue
        clients[f"ciciot_diad_client_{i}"] = {
            "X": np.concatenate([per_device_X[d] for d in bucket], axis=0),
            "y_device": np.concatenate([per_device_y[d] for d in bucket], axis=0),
            "classes": classes,
        }
    return clients


def load_ciciot_diad_attack_classification(
    path: str = CICIOT_DIAD_ATTACK_TASK,
    n_max_per_class: int = 5000,
    n_clients: int = 9,
    label_col: str = "attack_group",
    seed: int = 0,
) -> Dict[str, ClientData]:
    """Real attack-type classification data from the same richer feature
    set as the device-ID loader above. attack_task.parquet has NO device
    labels (confirmed against its schema), so -- same real limitation as
    load_ciciot2023_federated -- clients here are a random partition of
    pooled rows, not genuine per-device federated clients. `label_col`
    defaults to the coarser `attack_group` (Benign/DDoS/DoS/Mirai/Recon/
    BruteForce/...); pass `label_col="attack"` for the finer-grained
    per-technique label instead."""
    import pyarrow.parquet as pq

    feature_cols = load_ciciot_diad_numeric_feature_columns(path)
    pf = pq.ParquetFile(path)

    rng = np.random.RandomState(seed)
    class_frames: Dict[str, List[pd.DataFrame]] = {}
    class_counts: Dict[str, int] = {}

    for batch in pf.iter_batches(columns=feature_cols + [label_col], batch_size=200_000):
        df = batch.to_pandas()
        for cls, sub in df.groupby(label_col):
            cnt = class_counts.get(cls, 0)
            if cnt >= n_max_per_class:
                continue
            need = n_max_per_class - cnt
            if len(sub) > need:
                sub = sub.sample(n=need, random_state=seed)
            class_frames.setdefault(cls, []).append(sub)
            class_counts[cls] = cnt + len(sub)

    classes = sorted(class_frames.keys())
    label_to_idx = {c: i for i, c in enumerate(classes)}

    all_X, all_y = [], []
    for cls, frames in class_frames.items():
        df = pd.concat(frames, axis=0, ignore_index=True)
        X = df[feature_cols].fillna(0.0).values.astype(np.float32)
        y = np.full(len(df), label_to_idx[cls], dtype=np.int64)
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    print(f"CIC IoT-DIAD 2024 attack ({label_col}): {len(classes)} classes: {classes}")

    splits_X = np.array_split(X, n_clients)
    splits_y = np.array_split(y, n_clients)
    clients = {}
    for i in range(n_clients):
        clients[f"ciciot_diad_attack_client_{i}"] = ClientData(
            client_id=f"ciciot_diad_attack_client_{i}",
            X=splits_X[i],
            y_attack=splits_y[i],
            y_device=np.zeros(len(splits_y[i]), dtype=np.int64),
            classes=classes,
        )
    return clients


# ---------------------------------------------------------------------------
# Synthetic UNSW-style raw sequence generator (CODE-VALIDATION ONLY)
# ---------------------------------------------------------------------------

def make_synthetic_unsw(
    n_clients: int = 5,
    n_devices: int = 8,
    seq_len: int = 100,
    samples_per_device_per_client: int = 60,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Synthesize packet-length/direction windows resembling the UNSW
    representation described in Section V-B of the paper: signed values in
    roughly [-1500, 1500], with a distinct mean/variance/burstiness
    "fingerprint" per synthetic device type so a model can actually learn
    to separate them (otherwise this would only test that code runs, not
    that it can learn anything at all).

    THIS IS SYNTHETIC DATA, NOT UNSW. It exists only to validate the Phase
    0 / Phase I / Gating Agent pipeline end-to-end. Do not report numbers
    from it as paper results.
    """
    rng = np.random.RandomState(seed)
    device_profiles = []
    for d in range(n_devices):
        mean_len = rng.uniform(80, 1200)
        std_len = rng.uniform(20, 300)
        direction_bias = rng.uniform(0.3, 0.7)  # fraction incoming
        device_profiles.append((mean_len, std_len, direction_bias))

    clients = {}
    for c in range(n_clients):
        Xs, ys = [], []
        for d, (mean_len, std_len, dir_bias) in enumerate(device_profiles):
            for _ in range(samples_per_device_per_client):
                lengths = np.clip(
                    rng.normal(mean_len, std_len, size=seq_len), 20, 1500
                )
                signs = np.where(rng.random(seq_len) < dir_bias, 1.0, -1.0)
                seq = (lengths * signs).astype(np.float32)
                Xs.append(seq)
                ys.append(d)
        X = np.stack(Xs).astype(np.float32)
        y = np.array(ys, dtype=np.int64)
        perm = rng.permutation(len(y))
        clients[f"synthetic_client_{c}"] = {"X": X[perm], "y_device": y[perm]}
    return clients
