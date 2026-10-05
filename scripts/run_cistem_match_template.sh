#!/bin/bash
# Run cisTEM match_template (2.0.0-alpha prompt order) and time it.
# Usage: run_cistem_match_template.sh <match_template binary> <mic> <template> <out_prefix> \
#        <pixel_size> <df1> <df2> <df_angle> <high_res> <oop_step> <ip_step> <df_range> <df_step>
set -euo pipefail
BIN=$1 MIC=$2 VOL=$3 OUT=$4 PX=$5 DF1=$6 DF2=$7 DFANG=$8 HIRES=$9
OOP=${10} IP=${11} DFRANGE=${12} DFSTEP=${13}
mkdir -p "$(dirname "$OUT")"
/usr/bin/time -f "%e" -o "${OUT}_time.txt" "$BIN" <<EOT > "${OUT}_match.log" 2>&1
$MIC
$VOL
${OUT}_mip.mrc
${OUT}_scaled_mip.mrc
${OUT}_psi.mrc
${OUT}_theta.mrc
${OUT}_phi.mrc
${OUT}_defocus.mrc
${OUT}_pixel_size.mrc
${OUT}_avg.mrc
${OUT}_std.mrc
${OUT}_histogram.txt
$PX
300.0
2.7
0.07
$DF1
$DF2
$DFANG
0.0
$HIRES
$OOP
$IP
$DFRANGE
$DFSTEP
0.0
0.0
1.0
0.0
C1
yes
4
EOT
echo "wall time: $(cat "${OUT}_time.txt") s"
