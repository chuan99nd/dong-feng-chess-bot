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

| Service                | What it does                              | Port |
|------------------------|-------------------------------------------|------|
| `dongfeng-web`         | Web UI (`dfc web --engine board`)         | 8000 |
| `dongfeng-train`       | Training run (one-shot, no restart)       | —    |
| `dongfeng-tunnel`      | Cloudflare Tunnel → web UI + Grafana      | —    |
| `dongfeng-metrics`     | Training-metrics Prometheus exporter      | 9105 |
| `node_exporter`        | Host CPU / RAM / disk / net metrics       | 9100 |
| `nvidia_gpu_exporter`  | GPU util / mem / temp / power             | 9835 |
| `prometheus`           | Scrapes the exporters (TSDB under app)    | 9090 |
| `grafana-server`       | Dashboards, served under `/grafana`       | 3000 |

All monitoring ports bind to `127.0.0.1` only. Grafana is reached publicly at
`https://chess.chuantran.site/grafana` through the same Cloudflare Tunnel
(behind Cloudflare Access), on a path rule that routes `/grafana*` to port 3000
while everything else goes to the web UI on 8000.

## Monitoring stack (Prometheus + Grafana)

`setup.yml` imports `observability.yml`, so a single run brings up the full
stack. Run it standalone to (re)deploy just monitoring:

```bash
cd deploy
ansible-playbook observability.yml --ask-become-pass
# optional: set the Grafana admin password (anonymous Viewer is always on)
ansible-playbook observability.yml --ask-become-pass -e grafana_admin_password=…
```

- **Dashboard**: `deploy/monitoring/grafana/dashboards/dongfeng.json` is
  auto-provisioned (folder "Dong Feng") with training loss/accuracy/LR/throughput,
  GPU util/mem/temp/power, and host CPU/RAM/disk panels.
- **Datasource**: Prometheus is auto-provisioned (uid `dongfeng-prom`).
- **Model exporter**: `dfc metrics-export` reads `runs/*/run.json` +
  `metrics.jsonl` and exposes `dongfeng_train_*` / `dongfeng_run_*` series.
- **Anonymous access**: enabled as Viewer, so the dashboard opens without a login
  (the domain is already gated by Cloudflare Access). Log in as `admin` to edit.
- **State**: Prometheus TSDB (`monitoring/prometheus-data`) and the Grafana DB
  (`monitoring/grafana-data`) live under the project dir and are synced to the
  Mac by `backup.yml` alongside `checkpoints/` and `runs/`.

## Ephemeral machines — backup & resume

The GPU box is disposable: when it shuts down, its disk is wiped. The tunnel and
domain survive (they live on Cloudflare), but **training progress does not**
unless it is backed up off-box. Backups are kept **on this Mac** via rsync-over-SSH
(reuses the same SSH key as the playbooks — no extra credentials).

The checkpoint mirror lives in the git-ignored repo-root `checkpoints/` and
`runs/` folders.

### Back up (pull from server → Mac)

```bash
cd deploy
ansible-playbook backup.yml
```

Schedule it every 10 min so a sudden shutdown loses at most ~10 min of progress:

```bash
crontab -e
# */10 * * * * cd /Users/chuan/ghq/github.com/chuan99nd/dong-feng-chess-bot/deploy && \
#   /opt/homebrew/bin/ansible-playbook backup.yml >> /tmp/dongfeng-backup.log 2>&1
```

### When the machine dies → move to a new one

`provision.yml` does the whole rebuild in **one command** — it orchestrates
`setup.yml` (which imports `observability.yml`) → `nopasswd-sudo.yml` →
`pull-model.yml` (only if `-e hf_model=…`) → `start-train.yml`:

```bash
# 1. Point the inventory at the new host (edit ansible_host / ansible_port)
#    and copy your SSH key over:
ssh-copy-id -i ~/.ssh/dongfeng_5090.pub -p <port> ezycloudx-admin@<new-host>

# 2. One-shot full bring-up with the SAME tunnel id (the tunnel is a Cloudflare
#    resource, reused across boxes). Add -e hf_model=<name> to seed weights from
#    the HF Hub for a warm resume; omit it to train from scratch.
cd deploy
ansible-playbook provision.yml --ask-become-pass \
  -e tunnel_id=$TUNNEL_ID -e hf_model=board-5090-mid
```

This leaves the box fully online: deps + PyTorch cu128, data shards, the secure
Cloudflare tunnel, all systemd services, the Prometheus/Grafana monitoring stack,
passwordless `dongfeng-*` restarts (so later `update.yml` needs no sudo prompt),
serving weights pulled from HF, and training started/resumed.

You can still run each step à la carte (they're independent playbooks) — e.g.
just `setup.yml` then `start-train.yml` — if you don't want the full chain:

```bash
ansible-playbook setup.yml --ask-become-pass -e tunnel_id=$TUNNEL_ID
ansible-playbook nopasswd-sudo.yml --ask-become-pass   # passwordless restarts
ansible-playbook start-train.yml --ask-become-pass
```

`dfc train-board` **auto-resumes**: it detects `runs/<id>/ckpt.pt` and continues
from its saved step (LR schedule recomputed; optimizer momentum not restored — a
negligible warm-restart transient). To resume from an explicit file:
`dfc train-board ... --resume /path/to/ckpt.pt`.

> The same domain `chess.chuantran.site` works on the new machine with no DNS
> change, because the named tunnel is a Cloudflare resource, not tied to the box.

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
```

## Files

- `inventory.ini` — Server connection (SSH key, **git-ignored**, never commit)
- `ansible.cfg` — Ansible settings
- `provision.yml` — **One-shot full rebuild** of a wiped box: setup + nopasswd-sudo + (optional) HF pull + start-train
- `setup.yml` — Full server bootstrap (deps, PyTorch cu128, secure tunnel); imports `observability.yml`
- `start-train.yml` — Start/restart training
- `backup.yml` — Pull checkpoints/runs from server down to this Mac
- `monitor.yml` — Health check all services

Secrets that must never be committed (already git-ignored): `inventory.ini`,
`~/.cloudflared/*.json` credentials, `cert.pem`.

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
