# OS Tuning

These are optional board-level tweaks for a dedicated experiment machine.

The development Cubie A7S was observed with:

```text
CPU governor: ondemand on all cores
idle frequency: 416 MHz
vm.swappiness: 100
RAM: about 3.1 GiB available while the stack is running
```

For a dedicated RAG appliance, two conservative changes can help avoid latency spikes:

1. set CPU frequency governor to `performance`
2. lower swappiness to `10`

Apply:

```bash
bash install/apply_os_tuning.sh
```

Override defaults:

```bash
GOVERNOR=schedutil SWAPPINESS=20 bash install/apply_os_tuning.sh
```

The script writes:

```text
/etc/sysctl.d/99-vip9000-rag.conf
/etc/systemd/system/vip9000-cpufreq.service
```

Check current values:

```bash
for p in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "$p=$(cat "$p")"; done
cat /proc/sys/vm/swappiness
```

Do not use `performance` if the board is thermally constrained or battery powered.
