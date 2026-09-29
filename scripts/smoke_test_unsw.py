"""Validate the real UNSW loader on a small subset (1 day's CSV) before
running on the full 20-day, 21M-row dataset."""
import sys
import time
sys.path.insert(0, "..")

from fedgate.data import unsw_iot_devices, build_unsw_packet_sequences, UNSW_CSV_DIR
import glob, os

print("Parsing device list...")
devices = unsw_iot_devices()
print(f"{len(devices)} IoT devices found:")
for mac, name in list(devices.items())[:30]:
    print(f"  {mac} -> {name}")

# test on just the smallest CSV first for speed
files = sorted(glob.glob(os.path.join(UNSW_CSV_DIR, "*.csv")), key=os.path.getsize)
smallest = files[0]
print(f"\nTesting on smallest file: {smallest} ({os.path.getsize(smallest)/1e6:.1f} MB)")

tmp_dir = "/tmp/unsw_smoke"
os.makedirs(tmp_dir, exist_ok=True)
os.symlink(smallest, os.path.join(tmp_dir, os.path.basename(smallest))) if not os.path.exists(os.path.join(tmp_dir, os.path.basename(smallest))) else None

t0 = time.time()
windows = build_unsw_packet_sequences(devices, csv_dir=tmp_dir, seq_len=100, max_windows_per_device=500)
print(f"Elapsed: {time.time()-t0:.1f}s")
print(f"\nDevices with windows: {len(windows)}")
for name, w in sorted(windows.items(), key=lambda kv: -len(kv[1])):
    print(f"  {name}: {len(w)} windows, shape {w.shape}, sample stats: mean={w.mean():.1f} std={w.std():.1f}")
