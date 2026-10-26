# Long batch run — freeze risk, monitoring, hardware/setup speedups

## Can we guarantee the workstation will not freeze?

**No.** A desktop 3080 Ti shared with GNOME / remote-desktop can still freeze under load. What we *can* guarantee with good setup:

1. A freeze of the **screen** does **not** have to kill the batch (tmux / nohup + don’t reboot).
2. You will **notice** a dead or stalled job within ~10–45 minutes (cron watchdog), even if you are away.
3. Risk is **much lower** if the machine is used as a **headless compute box** during the run (SSH only, minimal GUI).

Rebooting because the GUI froze is what wasted progress last time — not the freeze itself.

---

## What to do for surety (operational)

### A. Keep the job detachable
- Prefer **tmux** (`scripts/run_batch_dense_parts_hand.sh` after pull).
- Current `nohup` job is already detachable — **do not kill/restart it** just to switch to tmux.

### B. Install the watchdog (recommended)
```bash
cd ~/KHETRE/DINO_KHETRE/DINO
git pull origin cursor/batch-dense-parts-hand-e54c
chmod +x scripts/watchdog_batch_dense_parts_hand.sh

crontab -e
# add:
*/10 * * * * /home/ifab-agiprobotw-0001/KHETRE/DINO_KHETRE/DINO/scripts/watchdog_batch_dense_parts_hand.sh
```

From phone / another PC anytime:
```bash
ssh ifab-agiprobotw-0001@ifab-ws22-09 \
  'cat ~/KHETRE/DINO_KHETRE/DINO/outputs/batch_sam3_dense_parts_hand/watchdog_status.txt'
```

- `state=OK` → process alive and log recently updated  
- `state=STALL_LOG_SILENT` → process exists but log quiet too long (investigate)  
- `state=DEAD_OR_FINISHED` → process gone (finished or crashed — check QC / log)

### C. If the screen freezes
1. **Do not reboot.**
2. SSH in → `bash scripts/check_batch_dense_parts_hand.sh`
3. If pid alive → leave it; optionally restart `gnome-shell` / re-login later.
4. Only reboot if SSH is dead **and** the machine is truly hung (last resort).

### D. Lower freeze probability
- During the month: **don’t use the workstation interactively** (no Firefox, no video, no remote desktop session if avoidable). Prefer SSH.
- Free disk (aim **>100G** free; 40G at 96% is risky).
- Keep one CUDA job only.
- Ensure good cooling / dust; check `nvidia-smi` temp occasionally (`watchdog` prints it).
- Optional: after current job ends, next run from a text console / multi-user target so GNOME isn’t compositing on the same GPU.

---

## Hardware / setup ways to fasten (honest)

| Lever | Speeds up? | Notes |
|-------|------------|--------|
| **Second GPU / second machine** | Yes (near linear if you split the 370 IDs) | Best hardware speedup; pipeline is single-GPU sequential today |
| **Cloud / lab GPU node (no display)** | Yes + fewer freezes | Ideal for month-long jobs |
| **More VRAM** | Small | Helps avoid OOM, not the concept loop |
| **Faster CPU / RAM** | Small | Detect is GPU-bound |
| **NVMe already** | — | Already fine |
| **Stop sharing GPU with desktop** | Stability + a bit of speed | Biggest *setup* win on this box |
| **Software: stride 4 + fewer prompts** | **Large (4–6×)** | See fast config option — no new hardware needed |

On **this** 3080 Ti alone, hardware cannot turn a ~month job into 2 days without cutting work (stride/prompts) or adding another GPU/machine.

### Multi-machine idea (if another PC exists)
Split `select` / recording lists (e.g. A–M vs N–Z), run two batches with different `output_root`, then merge QC CSVs. Needs a second CUDA box and a short merge step.

---

## Recommendation for your situation

You said you **can wait a month** but need **surety**:

1. **Leave the current run running** (don’t restart for tmux).
2. **Install the cron watchdog today** and check `watchdog_status.txt` daily from SSH/phone.
3. **Don’t reboot on GUI freeze**; SSH first.
4. Free disk + avoid using the desktop during the run.
5. If a month is only acceptable with lower freeze risk: after this run (or if it dies), consider moving the next full corpus to a **headless GPU machine**, or use a **fast config** (stride 4 + lean prompts) so the window of risk is days, not weeks.

*Surety = monitoring + don’t reboot + detachable process. Speed without cutting labels = more GPUs/machines. Speed on this box = fewer frames/concepts.*
