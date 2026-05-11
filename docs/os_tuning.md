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

## A76 Pinning Trial

The faster cores on the tested board are CPUs `6,7`. A user-space trial pinned `llama-server` to those cores and limited llama.cpp to two threads:

```bash
taskset -c 6,7 llama-server ... --threads 2 --threads-batch 2
```

Measured through the WebUI:

```text
run=1 wall=57.31s embedding=19.0720s llm=38.22s total=57.29s embedding_cache_hit=False
run=2 wall=8.19s  embedding=0.0007s  llm=8.18s  total=8.19s  embedding_cache_hit=True
```

This improves repeated cached queries slightly, but makes first/cold questions much slower than the default service setup. It is not the default.

Use A76 pinning only if your workload is mostly repeated cached queries and you want to reserve the A55 cores for other tasks.
