# Dong Feng — GPU Server Deployment

Ansible playbooks to deploy training + web UI on a rented GPU server
(RTX 5090 / Blackwell), with a **secure named Cloudflare Tunnel** for public
access, protected by **Cloudflare Access** (Zero Trust SSO).

## Prerequisites

```bash
# On your Mac
brew install ansible cloudflared
```

You also need a Cloudflare account with a zone for `chuantran.site`.

### 1. One-time SSH key setup (no passwords in files)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/dongfeng_5090 -C dongfeng-5090
ssh-copy-id -i ~/.ssh/dongfeng_5090.pub -p 15681 ezycloudx-admin@net9.thuepcpro.vn
```

You type the server password **once** during `ssh-copy-id`; afterwards Ansible
uses the key. **Then change the old password on the server** (`passwd`) — it was
shared over chat and should be rotated.

### 2. One-time Cloudflare named-tunnel setup

```bash
cloudflared tunnel login                       # pick the chuantran.site zone
cloudflared tunnel create dongfeng             # prints <TUNNEL_ID>, writes ~/.cloudflared/<TUNNEL_ID>.json
cloudflared tunnel route dns dongfeng chess.chuantran.site
export TUNNEL_ID=<TUNNEL_ID>                    # the playbook reads this
```

### 3. Lock the app behind Cloudflare Access (the real auth layer)

In the **Zero Trust dashboard → Access → Applications → Add self-hosted**:

- Application domain: `chess.chuantran.site`
- Policy: **Allow** only your email (or Google/GitHub SSO).

This forces every visitor to authenticate with Cloudflare before any request
reaches the server. The web UI itself binds to `127.0.0.1` only — it is never
exposed on a public port; `cloudflared` makes an **outbound** connection, so no
inbound firewall ports are opened.

## Quick start

```bash
cd deploy

# 1. Full setup (deps, repo, PyTorch cu128, data, secure tunnel, services)
ansible-playbook setup.yml --ask-become-pass -e tunnel_id=$TUNNEL_ID

# 2. Start/restart training with different params
ansible-playbook start-train.yml --ask-become-pass
ansible-playbook start-train.yml --ask-become-pass -e train_preset=1b -e train_steps=50000

# 3. Check status of everything
ansible-playbook monitor.yml --ask-become-pass
```

> `--ask-become-pass` prompts for the sudo password interactively so it is never
> written to disk.

## Services

| Service              | What it does                          | Port |
|----------------------|---------------------------------------|------|
| `dongfeng-web`       | Web UI (`dfc web --engine board`)     | 8000 |
| `dongfeng-train`     | Training run (one-shot, no restart)   | —    |
| `dongfeng-tunnel`    | Cloudflare Tunnel → web UI            | —    |
| `dongfeng-ckpt-sync` | Timer: back up checkpoints/runs → R2  | —    |

## Ephemeral machines — backup & resume

The GPU box is disposable: when it shuts down, its disk is wiped. The tunnel and
domain survive (they live on Cloudflare), but **training progress does not**
unless it is backed up off-box. This deploy handles that via **Cloudflare R2**
(S3-compatible object storage).

### One-time R2 setup

1. In the Cloudflare dashboard: **R2 → Create bucket** (e.g. `dongfeng`).
2. **R2 → Manage API Tokens → Create** an S3 token with read/write on that bucket.
3. Export the credentials in the shell you run Ansible from (never commit them):

```bash
export R2_BUCKET=dongfeng
export R2_ACCOUNT_ID=<cloudflare-account-id>
export R2_ACCESS_KEY_ID=<token-access-key>
export R2_SECRET_ACCESS_KEY=<token-secret>
```

With these set, `setup.yml` installs rclone, writes a locked-down `rclone.conf`,
and starts a **timer that syncs `checkpoints/` + `runs/` to R2 every 10 min**.
If `R2_BUCKET` is empty, all backup tasks are skipped (no-op).

### When the machine dies → move to a new one

```bash
# 1. Point the inventory at the new host (edit ansible_host / ansible_port)
#    and copy your SSH key over:
ssh-copy-id -i ~/.ssh/dongfeng_5090.pub -p <port> ezycloudx-admin@<new-host>

# 2. Re-run setup with the SAME tunnel id and R2 creds exported.
#    It restores checkpoints/runs from R2 before starting.
cd deploy
ansible-playbook setup.yml --ask-become-pass -e tunnel_id=$TUNNEL_ID

# 3. Resume training (also pulls latest from R2 first).
ansible-playbook start-train.yml --ask-become-pass
```

Training **auto-resumes**: `dfc train-board` detects `runs/<id>/ckpt.pt` and
continues from its saved step (LR schedule recomputed; optimizer momentum is not
restored — a negligible warm-restart transient). To resume from an explicit file:
`dfc train-board ... --resume /path/to/ckpt.pt`.

> The same domain `chess.chuantran.site` works on the new machine with no DNS
> change, because the named tunnel is a Cloudflare resource, not tied to the box.

## Monitoring from your machine

### Quick checks via SSH

```bash
# GPU utilization (live)
ssh -p 15681 ezycloudx-admin@net9.thuepcpro.vn "watch -n1 nvidia-smi"

# Training log (live tail)
ssh -p 15681 ezycloudx-admin@net9.thuepcpro.vn "tail -f ~/dong-feng/runs/train.log"

# All services status
ssh -p 15681 ezycloudx-admin@net9.thuepcpro.vn "systemctl status dongfeng-{web,train,tunnel}"

# Interactive GPU monitor
ssh -p 15681 ezycloudx-admin@net9.thuepcpro.vn "nvtop"
```

### Via the Web UI

The web UI has a **Training** tab that shows live metrics (loss, accuracy)
from all training runs. Access it at the fixed URL — after signing in through
Cloudflare Access:

```
https://chess.chuantran.site
```

### Systemd commands (on the server)

```bash
# Stop/start training
sudo systemctl stop dongfeng-train
sudo systemctl start dongfeng-train

# Restart web UI
sudo systemctl restart dongfeng-web

# View logs
journalctl -u dongfeng-train -f
journalctl -u dongfeng-web -f
journalctl -u dongfeng-tunnel -f

# Backup timer status + last sync
systemctl status dongfeng-ckpt-sync.timer
journalctl -u dongfeng-ckpt-sync -n 20
```

## Files

- `inventory.ini` — Server connection (SSH key, **git-ignored**, never commit)
- `ansible.cfg` — Ansible settings
- `setup.yml` — Full server bootstrap (deps, PyTorch cu128, secure tunnel)
- `start-train.yml` — Start/restart training
- `monitor.yml` — Health check all services

Secrets that must never be committed (already git-ignored): `inventory.ini`,
`~/.cloudflared/*.json` credentials, `cert.pem`, `rclone.conf` (R2 keys).

## Architecture

```
┌─────────────────────────────────────────────────┐
│  GPU Server (RTX 5090, Ubuntu 24.04)             │
│                                                   │
│  dongfeng-train ──→ GPU (CUDA 12.8 training)        │
│                                                   │
│  dongfeng-web ──→ 127.0.0.1:8000 (loopback only)   │
│       ↑                                            │
│  dongfeng-tunnel ──(outbound)──→ Cloudflare edge     │
│       ↑   named tunnel, credentials 0600           │
└───────┼───────────────────────────────────┘
        │
   Cloudflare Access (SSO)  ← authenticates every visitor
        │
   https://chess.chuantran.site  ← fixed public URL
```
