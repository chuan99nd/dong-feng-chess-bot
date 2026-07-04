---
name: deploy
description: Deploy or manage the GPU server (Ansible playbooks under deploy/) — full bootstrap, code-only updates, starting/restarting training, monitoring stack, health checks, and checkpoint backup. Use when the user wants to deploy the app, deploy monitoring, start/restart training on the server, check server health, or back up/restore checkpoints.
---

# deploy

All server operations for Dong Feng go through the Ansible playbooks in
`deploy/`. Never SSH in and hand-edit things — every change should be
expressed as a playbook run so it's reproducible after the (ephemeral) GPU
box is thrown away. Full narrative doc: [deploy/README.md](../../../deploy/README.md).

Run every command from the `deploy/` directory. `--ask-become-pass` prompts
for the remote sudo password interactively (never written to disk); only the
playbooks that touch systemd/apt need it.

## Playbooks (what each one does, when to use it)

| Playbook | Use when… | Needs `--ask-become-pass`? |
| --- | --- | --- |
| `setup.yml` | First-time bootstrap on a **new/replacement** GPU box: installs deps, uv, PyTorch cu128, clones the repo, pushes local `checkpoints/`+`runs/`+`monitoring/` mirrors up, installs the secure Cloudflare tunnel, creates all systemd services, then imports `observability.yml`. | Yes |
| `update.yml` | You changed Python/web code only (no new deps) and want it live fast — **`git push` first**, then the server pulls `main`, `uv sync`, restarts `dongfeng-web` + `dongfeng-train`. Code goes via GitHub (no rsync — avoids server-tree conflicts). Training auto-resumes separately. | No, once `nopasswd-sudo.yml` has been run (else Yes) |
| `nopasswd-sudo.yml` | One-time: grant `ezycloudx-admin` restricted NOPASSWD sudo for **just** the `systemctl` commands the deploy needs (restart/start/stop dongfeng-* + daemon-reload), validated with visudo. After this, `update.yml` runs with no password prompt. | Yes (writes /etc/sudoers.d — the one time you still need it) |
| `push-model.yml` | Upload a trained checkpoint to the **Hugging Face Hub** (models live on HF, not git): pulls `runs/<run_id>/ckpt.pt` from the server, then `hf upload`s it to `<model_name>/` in the HF repo. Run with `-e run_id=<run>`. Auth = `HF_TOKEN` in `.env`. Fail-loud. | No |
| `pull-model.yml` | Download a model from HF onto the GPU server (`hf download` into `models/<model_name>/`) so the web/train services can load it. Run with `-e model_name=<name>` (run `update.yml` first so `hf` is installed). | No |
| `start-train.yml` | Start or restart training with specific params (`train_preset`, `train_steps`, `train_batch`). Pushes local `runs/` mirror up first so it can resume from `ckpt.pt`. | Yes |
| `observability.yml` | (Deploy) or redeploy just the monitoring stack: Prometheus, Grafana, node/GPU/model exporters. Already imported by `setup.yml`; run standalone to update dashboards/exporter config without touching training/web. | Yes |
| `monitor.yml` | Health check — GPU status + `systemctl is-active` for every service (web, train, tunnel, metrics, exporters, prometheus, grafana). Read-only, safe to run anytime. | Yes (reads systemd/nvidia-smi as root) |
| `ablation-bias.yml` | Run the 2D-bias-head ablation (WP-BIAS): fixed preset/size, varies only `--n-bias-head` (default `[0,2,4]`), then `dfc eval head-diversity` per arm. Use to measure whether an architecture addition helps at constant model size. | Yes |
| `backup.yml` | Pull `checkpoints/`, `runs/`, `monitoring/` state **down** from the server to this Mac. Run before destroying a box, or on a cron every ~10 min. | No |

## Common tasks

```bash
cd deploy

# Full deploy on a fresh GPU box (one-time Cloudflare setup first — see README §1-3)
ansible-playbook setup.yml --ask-become-pass -e tunnel_id=$TUNNEL_ID

# One-time: restricted NOPASSWD sudo so update.yml needs no password afterwards
ansible-playbook nopasswd-sudo.yml --ask-become-pass

# Ship a code change (no new deps) — fast path (PUSH FIRST: the server pulls main)
git push origin main
ansible-playbook update.yml          # no --ask-become-pass once nopasswd-sudo.yml has run

# Upload a trained checkpoint to the Hugging Face Hub (models live on HF, not git)
ansible-playbook push-model.yml -e run_id=board-5090-mid
# Download a model from HF onto the GPU server (after update.yml has installed hf)
ansible-playbook pull-model.yml -e model_name=board-5090-mid

# Deploy/update just the monitoring stack (Prometheus + Grafana + exporters)
ansible-playbook observability.yml --ask-become-pass
ansible-playbook observability.yml --ask-become-pass -e grafana_admin_password=…

# Start/restart training with different params
ansible-playbook start-train.yml --ask-become-pass
ansible-playbook start-train.yml --ask-become-pass -e train_preset=1b -e train_steps=50000 -e train_batch=128

# Run the fixed-size 2D-bias-head ablation (varies only --n-bias-head)
ansible-playbook ablation-bias.yml --ask-become-pass
ansible-playbook ablation-bias.yml --ask-become-pass -e ablation_preset=mid -e 'bias_heads=[0,4]'

# Check everything is up (GPU + all systemd services)
ansible-playbook monitor.yml --ask-become-pass

# Pull checkpoints/runs/monitoring state back to the Mac (do this before killing the box)
ansible-playbook backup.yml
```

## Moving to a new/replacement machine

1. Edit `deploy/inventory.ini` (`ansible_host` / `ansible_port`) — **git-ignored, never commit**.
2. `ssh-copy-id -i ~/.ssh/dongfeng_5090.pub -p <port> ezycloudx-admin@<new-host>`
3. `ansible-playbook setup.yml --ask-become-pass -e tunnel_id=$TUNNEL_ID` (same tunnel id — the domain doesn't change, it's a Cloudflare resource).
4. `ansible-playbook start-train.yml --ask-become-pass` to resume training (`dfc train-board` auto-resumes from `runs/<id>/ckpt.pt`).

## Services this manages (all loopback-bound; public access only via the named Cloudflare tunnel + Access SSO)

`dongfeng-web` (8000), `dongfeng-train` (one-shot), `dongfeng-tunnel`,
`dongfeng-metrics` (9105), `node_exporter` (9100), `nvidia_gpu_exporter` (9835),
`prometheus` (9090), `grafana-server` (3000, under `/grafana`).

## Secrets — never commit

`deploy/inventory.ini`, `deploy/*.vault`, `deploy/secrets/`, `~/.cloudflared/*.json`
credentials, `*.pem`. All already covered by `.gitignore`.

## When to use a different skill instead

- Training hyperparameters / recipes (BC, distill, DPO) → **train-run** skill;
  this skill only covers *how to launch it on the server*.
- Checking training metrics/results → don't SSH in for this; use
  `dfc eval last` / `dfc ckpt info` (see **train-run**, **checkpoint-inspect**)
  or the web UI's Training tab, reached through `monitor.yml`'s reported URL.
