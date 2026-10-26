# Cure workstation freezes during batch (no IT / no SSH)

You want the **machine itself** to stop freezing — not a remote-access workaround.

## Root cause

The **RTX 3080 Ti** is doing two jobs at once:

1. Long SAM3 CUDA compute  
2. Desktop (GNOME + often **gnome-remote-desktop** + Firefox)

Under sustained load the compositor locks up. That is a **local GPU sharing** problem.

---

## Permanent cure (pick the strongest you can)

### 1. Best — display on integrated GPU (no IT)

If the PC has Intel/AMD graphics:

1. Shut down.  
2. Plug monitor into **motherboard** HDMI/DP (not the GeForce card).  
3. BIOS: enable **iGPU / internal graphics**.  
4. Boot; confirm desktop still works.  
5. CUDA/SAM keep using NVIDIA; desktop uses iGPU → freezes become rare.

Check:
```bash
lspci | grep -iE 'VGA|3D|Display'
bash scripts/prepare_workstation_for_batch.sh
```

### 2. Always before a long batch (local script)

```bash
cd ~/KHETRE/DINO_KHETRE/DINO
bash scripts/prepare_workstation_for_batch.sh
# close Firefox when prompted
```

This stops **gnome-remote-desktop**, disables animations, and reminds you to free disk / close browsers.

### 3. Don’t use the GUI while batching

During the month-long run: leave the login screen or an empty desktop. No remote desktop into this box, no browser, no video. Keyboard/`Ctrl+Alt+F3` only if needed.

### 4. If it freezes anyway — fix GUI **without** reboot

```text
Ctrl+Alt+F3
```
Login, then either:

```bash
# restart desktop only (batch usually keeps running)
killall -SIGQUIT gnome-shell
```

or just confirm batch:

```bash
bash ~/KHETRE/DINO_KHETRE/DINO/scripts/check_batch_dense_parts_hand.sh
```

Then `Ctrl+Alt+F2` (or F1) back to GUI.  
**Reboot only if F3 itself is dead** — reboot loses only the current video mid-detect.

### 5. Shorten the risk window (optional software)

Fewer days of load ⇒ fewer freeze chances: stride 4 + fewer prompts (fast config). Same machine, no IT.

### 6. Survive reboot if you must power-cycle

`skip_if_done` + optional systemd service (`scripts/install_batch_systemd_user.sh`) so a forced reboot doesn’t mean starting the corpus from zero. Still no IT.

---

## What will *not* permanently cure freezes

- SSH / VPN / asking IT (access only — you asked to avoid this)  
- Watchdog alone (detects problems; doesn’t stop GNOME locking)  
- “Hope” on the same GPU for desktop + month-long SAM  

---

## Practical recipe for you

1. **Today (when you have keyboard):** `Ctrl+Alt+F3` → check batch → `killall -SIGQUIT gnome-shell` if you need the mouse back — **don’t reboot**.  
2. **Before continuing long runs:** `bash scripts/prepare_workstation_for_batch.sh` + close browsers + stop using remote desktop **to** this PC.  
3. **Permanent hardware:** move monitor cable to **motherboard iGPU** if available.  
4. Keep **watchdog cron** so you notice stalls on the local machine via `cat …/watchdog_status.txt` when the GUI works.

That is the no-IT path: **unload the desktop from the NVIDIA GPU**, and **restart GNOME from F3 instead of rebooting**.
