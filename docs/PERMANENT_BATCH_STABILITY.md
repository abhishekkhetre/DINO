# Permanent solution: freezes + month-long SAM batch

## What “permanent” means here

| Goal | Solution |
|------|----------|
| GUI freeze must **not** kill the job | Detach process (**systemd user service** or tmux) + **never reboot** only because the screen is stuck |
| Job must **survive reboot / crash** | `skip_if_done: true` + **systemd Restart=on-failure** + **linger** so service comes back |
| You can check from **home** | Lab **VPN + SSH (IP/FQDN)** + watchdog status file |
| Freeze must become **rare** | Stop sharing the 3080 Ti with a busy desktop (headless / iGPU for display / other machine) |

There is no magic flag that makes GNOME+CUDA on one 12GB GPU never freeze. Permanent = **architecture + auto-resume + remote access**.

---

## Tier 1 — Do this on the lab workstation (required)

### A. Auto-resume service (survives crash/reboot)

**If a nohup batch is already running, finish or stop it cleanly before enabling the service** (one GPU job only).

```bash
cd ~/KHETRE/DINO_KHETRE/DINO
git pull origin cursor/batch-dense-parts-hand-e54c
bash scripts/install_batch_systemd_user.sh

# only when GPU has no other python batch:
systemctl --user start dense-parts-batch.service
loginctl enable-linger "$USER"
systemctl --user enable dense-parts-batch.service

systemctl --user status dense-parts-batch.service
```

After a reboot the service can start again and **skip finished recordings**.

### B. Watchdog cron (already discussed)

```bash
*/10 * * * * /home/ifab-agiprobotw-0001/KHETRE/DINO_KHETRE/DINO/scripts/watchdog_batch_dense_parts_hand.sh
```

### C. Home access (ask IT once)

- VPN into KIT/lab  
- SSH host: IP or `hostname -f`  
- Test from Windows at home **before** the next long run  

Without this, a freeze while you are away means you depend on a colleague with SSH from another lab PC (`Ctrl+Alt+F3` or SSH — not the frozen mouse).

---

## Tier 2 — Make freezes rare (setup / hardware)

Pick **one** permanent display strategy:

### Option 1 — Best: another GPU machine / headless server
Run the batch only there. Desktop laptop for you. No shared compositor.

### Option 2 — Same box, display on **integrated GPU**, compute on 3080 Ti
If the PC has Intel/AMD iGPU:

1. Plug monitor into **motherboard** HDMI/DP (not the NVIDIA card).  
2. In BIOS, enable iGPU.  
3. NVIDIA becomes compute-only → far fewer GUI freezes under CUDA load.

### Option 3 — Same box, “batch mode” (no GNOME during the month)
When starting a long run (requires sudo; coordinate with IT):

```bash
# WARNING: kills graphical session — use only when you work via SSH
sudo systemctl isolate multi-user.target   # text/SSH only
# after batch:
sudo systemctl isolate graphical.target
```

Also disable remote desktop while batching if it uses the NVIDIA GPU:
```bash
systemctl --user stop gnome-remote-desktop.service 2>/dev/null || true
```

### Option 4 — Software: shorter risk window
`stride 4` + fewer prompts → days instead of a month → less time exposed to freezes (trade some density).

---

## What to do when the screen freezes (permanent habit)

1. **Do not reboot.**  
2. From home: VPN + SSH → `cat …/watchdog_status.txt`.  
3. On site: another PC SSH, or `Ctrl+Alt+F3`.  
4. `state=OK` → walk away.  
5. Reboot only if SSH and TTY are both dead; systemd + `skip_if_done` will continue afterward.

---

## Recommended end state for your internship box

```text
[You at home] --VPN/SSH--> [workstation]
                              ├─ systemd: dense-parts-batch (CUDA)
                              ├─ cron: watchdog every 10 min
                              └─ display on iGPU OR multi-user/SSH-only during run
```

That is the permanent solution: **job is a service**, **progress is resumable**, **you can always check remotely**, **GPU is not fighting the desktop**.
