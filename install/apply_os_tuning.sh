#!/usr/bin/env bash
set -euo pipefail

GOVERNOR="${GOVERNOR:-performance}"
SWAPPINESS="${SWAPPINESS:-10}"

if [[ $EUID -ne 0 ]]; then
  exec sudo GOVERNOR="$GOVERNOR" SWAPPINESS="$SWAPPINESS" "$0"
fi

changed_governors=0
for governor_path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [[ -e "$governor_path" ]] || continue
  if grep -qw "$GOVERNOR" "$(dirname "$governor_path")/scaling_available_governors"; then
    echo "$GOVERNOR" > "$governor_path"
    changed_governors=$((changed_governors + 1))
  else
    echo "governor $GOVERNOR not available for $governor_path" >&2
  fi
done

sysctl -w "vm.swappiness=$SWAPPINESS"

cat >/etc/sysctl.d/99-vip9000-rag.conf <<EOF
vm.swappiness=$SWAPPINESS
EOF

if command -v systemctl >/dev/null 2>&1; then
  cat >/etc/systemd/system/vip9000-cpufreq.service <<EOF
[Unit]
Description=Apply CPU governor for VIP9000 RAG stack
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for p in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo $GOVERNOR > "$p" 2>/dev/null || true; done'

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable vip9000-cpufreq.service
fi

echo "set $changed_governors CPU governors to $GOVERNOR"
echo "set vm.swappiness=$SWAPPINESS"
