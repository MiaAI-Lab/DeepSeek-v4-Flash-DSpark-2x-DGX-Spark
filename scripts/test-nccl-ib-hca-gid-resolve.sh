#!/usr/bin/env bash
# CPU-only behavioral gates for the NCCL_IB_HCA -> RoCEv2 GID validation used
# by start-deepseek-v4-flash-dspark.sh's NCCL_IB_GID_AUTO=1 path.
#
# The resolver mirrors NCCL's selector semantics (parseStringList in
# src/misc/utils.cc) on the node that owns the sysfs tree: optional leading "^"
# (exclude), then optional "=" (exact match instead of prefix match), then
# comma-separated name[:port[:rail[:plane]]] tokens. Only the first 32
# non-empty entries are stored, and stored names use netIf::prefix's 63-byte
# payload. An absent *or empty* port field means any port; a non-empty one is
# atoi() (base 10, one conversion over the whole field), and a port outside
# the resolver's conservative nine-digit bound is clamped rather than evaluated, because $(( )) wraps
# modulo 2^64 and one such value wraps to the -1 wildcard. The selector is
# applied to the candidate universe ncclIbInit builds — ACTIVE ports with an
# Ethernet/InfiniBand link layer, separately capped at MAX_IB_DEVS=32 — so a
# DOWN sibling port neither fails the resolve nor
# constrains the index. Every selected member must validate against the match IP
# or an IPv4 on its own netdev; a member with no usable RoCEv2 GID fails closed
# (exit 1). Usable index sets are audited per member on stderr and never
# reconciled: AUTO=1 pins nothing — NCCL picks the RoCEv2/IPv4 GID per HCA
# when NCCL_IB_GID_INDEX is absent — so disjoint per-member usable sets are
# fine, the resolver writes nothing to stdout, and the tail of this suite also
# checks the orchestration contract (AUTO=1 clears both rank variables and the
# worker env always carries the key; AUTO=0 pins verbatim).
#
# The suite extracts the launcher's own functions and runs the launcher's own
# generated lookup script through its unchanged local/SSH branches. Strict
# transport and `ip -4 -o addr show dev` stubs plus redirected fake sysfs trees
# keep the checks CPU-only while exercising the shipped boundaries. Running it
# against a pre-fix launcher fails behaviorally (wrong result / wrong exit
# code), not merely because a helper cannot be extracted.
set -euo pipefail
unset BASH_ENV SSH_STDOUT_NOISE

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
START="$ROOT/start-deepseek-v4-flash-dspark.sh"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

# --- extract the launcher's own code (works on pre-fix launchers too, so the
# regression shows up as behavioral FAILs below) ---
eval "$(awk '/^ipv4_to_gid_suffix\(\) \{$/,/^\}$/' "$START")"
resolver_body_src="$(awk '/^NCCL_HCA_RESOLVER_BODY=/,/^\)"$/' "$START")"
if [ -n "$resolver_body_src" ]; then
  eval "$resolver_body_src"
else
  NCCL_HCA_RESOLVER_BODY=""
fi
eval "$(awk '/^resolve_rocev2_gid_index\(\) \{$/,/^\}$/' "$START")"
eval "$(awk '/^resolve_nccl_gid_indexes\(\) \{$/,/^\}$/' "$START")"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# `ip` stub: accepts only `ip -4 -o addr show dev NAME`, then answers from
# $IP_FIXTURE ("<netdev> <ipv4>" lines) in the real one-line output shape.
stub="$tmp/stub-bin"
mkdir -p "$stub"
cat >"$stub/ip" <<'STUB'
#!/usr/bin/env bash
if [ "$#" -ne 6 ] || [ "$1" != "-4" ] || [ "$2" != "-o" ] \
  || [ "$3" != "addr" ] || [ "$4" != "show" ] || [ "$5" != "dev" ]; then
  printf 'ip stub: unexpected argv\n' >&2
  exit 64
fi
dev=$6
[ -f "${IP_FIXTURE:-}" ] || exit 0
while read -r d a; do
  [ "$d" = "$dev" ] && printf '2: %s    inet %s/24 brd 0.0.0.0 scope global %s\n' "$d" "$a" "$d"
done <"$IP_FIXTURE"
exit 0
STUB

# `ssh` stub preserves stdin/stdout/stderr and exit status while requiring the
# launcher's exact `ssh TARGET "bash -s"` argv. Optional stdout noise models a
# remote login banner without changing the resolver's successful exit status.
cat >"$stub/ssh" <<'STUB'
#!/usr/bin/env bash
if [ "$#" -ne 2 ] || [ "$1" != "${SSH_EXPECT_TARGET:-}" ] || [ "$2" != "bash -s" ]; then
  printf 'ssh stub: unexpected argv\n' >&2
  exit 64
fi
[ -z "${SSH_STDOUT_NOISE:-}" ] || printf '%s\n' "$SSH_STDOUT_NOISE"
exec /bin/bash -s
STUB
chmod +x "$stub/ip" "$stub/ssh"

mk_gid() { # $1=root $2=dev $3=port $4=index $5=gid $6=type [$7=ndev]
  local d="$1/sys/class/infiniband/$2/ports/$3"
  mkdir -p "$d/gids" "$d/gid_attrs/types" "$d/gid_attrs/ndevs"
  printf '%s\n' "$5" >"$d/gids/$4"
  printf '%s\n' "$6" >"$d/gid_attrs/types/$4"
  if [ -n "${7:-}" ]; then printf '%s\n' "$7" >"$d/gid_attrs/ndevs/$4"; fi
}

# Port attributes as the kernel exposes them ("4: ACTIVE" / "Ethernet"). Most
# fixtures leave them absent on purpose: an unreadable attribute must not be
# read as "inactive", so those ports stay candidates.
mk_port_attr() { # $1=root $2=dev $3=port $4=state $5=link_layer
  local d="$1/sys/class/infiniband/$2/ports/$3"
  mkdir -p "$d"
  printf '%s\n' "$4" >"$d/state"
  printf '%s\n' "$5" >"$d/link_layer"
}

resolve() { # $1=root $2=fixture $3=spec $4=match-ip [$5=ssh-target]
  local NCCL_GID_RESOLVE_SYSROOT="$1/sys/class/infiniband"
  local IP_FIXTURE="$2" SSH_EXPECT_TARGET="${5:-}"
  local PATH="$stub:$PATH"
  export NCCL_GID_RESOLVE_SYSROOT IP_FIXTURE SSH_EXPECT_TARGET PATH
  resolve_rocev2_gid_index "${5:-}" "$3" "$4"
}

expect_rc() { # $1=label $2=want-rc $3=root $4=fixture $5=spec $6=ip [$7=ssh]
  local rc=0
  resolve "$3" "$4" "$5" "$6" "${7:-}" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq "$2" ]; then
    ok "$1"
  else
    bad "$1: rc=$rc want=$2"
  fi
}

# --- head layout: three single-port HCAs on one shared link address, plus a
# device on a different address with no ndev metadata ---
# 10.0.22.1 -> ...ffff:0a00:1601 ; 10.0.22.2 -> ...ffff:0a00:1602
h="$tmp/head"
hfx="$tmp/head.fixture"
: >"$hfx"
mk_gid "$h" devA 1 0 '::ffff:0a00:1601' 'RoCE v1'
mk_gid "$h" devA 1 3 '::ffff:0a00:1601' 'RoCE v2'
mk_gid "$h" devB 1 3 '::ffff:0a00:1601' 'RoCE v2'
mk_gid "$h" devC 1 5 '::ffff:0a00:1602' 'RoCE v2'
mk_gid "$h" devE 1 6 '::ffff:0a00:1601' 'RoCE v2'

expect_rc "bare device validates (prefix match)" 0 "$h" "$hfx" 'devA' '10.0.22.1'
expect_rc "exact-match single device" 0 "$h" "$hfx" '=devA' '10.0.22.1'
expect_rc "exact multi-HCA list (original regression)" 0 "$h" "$hfx" '=devA,devB' '10.0.22.1'
expect_rc "comma list without =" 0 "$h" "$hfx" 'devA,devB' '10.0.22.1'
expect_rc "exact token skips missing device, keeps real one" 0 "$h" "$hfx" '=devA,devGone' '10.0.22.1'
expect_rc "exclusion selects the rest" 0 "$h" "$hfx" '^devC,devE' '10.0.22.1'
expect_rc "exact exclusion selects the rest" 0 "$h" "$hfx" '^=devC,devE' '10.0.22.1'
expect_rc "single member on its own address" 0 "$h" "$hfx" 'devC' '10.0.22.2'
expect_rc "member with no usable GID fails closed" 1 "$h" "$hfx" 'devC' '10.0.22.1'
expect_rc "selector matching nothing fails closed" 1 "$h" "$hfx" '=devGone' '10.0.22.1'
expect_rc "shared-IP selection with an unvalidatable member fails closed (empty selector = all ports)" 1 "$h" "$hfx" '' '10.0.22.1'
expect_rc "intra-node disjoint usable sets validate (no index is pinned)" 0 "$h" "$hfx" '=devA,devE' '10.0.22.1'

# the success audit lists every member's own usable index, disjoint or not
rc=0
err="$(resolve "$h" "$hfx" '=devA,devE' '10.0.22.1' 2>&1 >/dev/null)" || rc=$?
if [ "$rc" -eq 0 ] && printf '%s\n' "$err" | grep -Fqx '  member devA:1 -> RoCEv2 gid index 3 (via match-ip 10.0.22.1)' \
  && printf '%s\n' "$err" | grep -Fqx '  member devE:1 -> RoCEv2 gid index 6 (via match-ip 10.0.22.1)'; then
  ok "disjoint members each audit their own usable index"
else
  bad "disjoint audit lines wrong (rc=$rc): $err"
fi

# --- worker layout: dual-port HCAs on distinct per-port link addresses,
# resolved via each member's own netdev (no shared match IP involved) ---
# 10.0.25.x -> ...ffff:0a00:19xx ; 10.0.26.x -> ...ffff:0a00:1axx
w="$tmp/worker"
wfx="$tmp/worker.fixture"
mk_gid "$w" devM 1 2 '::ffff:0a00:1901' 'RoCE v2' enm1
mk_gid "$w" devM 2 2 '::ffff:0a00:1902' 'RoCE v2' enm2
mk_gid "$w" devN 1 2 '::ffff:0a00:1a01' 'RoCE v2' enn1
mk_gid "$w" devN 2 9 '::ffff:0a00:1a02' 'RoCE v2' enn2
printf '%s\n' 'enm1 10.0.25.1' 'enm2 10.0.25.2' 'enn1 10.0.26.1' 'enn2 10.0.26.2' >"$wfx"

expect_rc "omitted port = both ports, validated per member (own-addr)" 0 "$w" "$wfx" 'devM' '10.0.99.99'
expect_rc "explicit port 1" 0 "$w" "$wfx" 'devN:1' '10.0.99.99'
expect_rc "explicit port 2" 0 "$w" "$wfx" 'devN:2' '10.0.99.99'
expect_rc "multiport disjoint usable indexes validate per member" 0 "$w" "$wfx" 'devN' '10.0.99.99'
expect_rc "distinct per-HCA addresses both validate via own netdev" 0 "$w" "$wfx" '=devM:1,devN:1' '10.0.99.99'
expect_rc "match-ip member + own-addr member both validate" 0 "$w" "$wfx" '=devM:1,devN:1' '10.0.25.1'
rc=0
audit_err="$tmp/member-audit.err"
audit_out="$tmp/member-audit.out"
resolve "$w" "$wfx" '=devM:1,devN:1' '10.0.25.1' >"$audit_out" 2>"$audit_err" || rc=$?
if [ "$rc" -eq 0 ] && [ ! -s "$audit_out" ] \
  && grep -Fqx '  member devM:1 -> RoCEv2 gid index 2 (via match-ip 10.0.25.1)' "$audit_err" \
  && grep -Fqx '  member devN:1 -> RoCEv2 gid index 2 (via own-addr 10.0.26.1 on enn1)' "$audit_err"; then
  ok "successful multi-HCA validation audits every member, writes nothing to stdout"
else
  bad "success audit lines wrong (rc=$rc): $(cat "$audit_err")"
fi
expect_rc "port exclusion (^dev:port honors the port)" 0 "$w" "$wfx" '^devN:2' '10.0.99.99'
expect_rc "non-numeric port token matches no port" 1 "$w" "$wfx" 'devM:abc' '10.0.99.99'

# --- NCCL port-field grammar ---
# parseStringList splits name[:port[:rail[:plane]]]; an absent or empty port
# field is -1 (any port), a non-empty one is atoi() (base 10, stops at the first
# non-digit). Ports 1/8/10 exist here so leading-zero forms are distinguishable:
# "08" must be 8 (not a bad octal literal) and "010" must be 10 (not octal 8).
# 10.0.32.1 -> ...0a00:2001 ; 10.0.32.8 -> ...0a00:2008 ; 10.0.32.16 -> ...0a00:2010
pg="$tmp/portgrammar"
pgfx="$tmp/pg.fixture"
mk_gid "$pg" devP 1 4 '::ffff:0a00:2001' 'RoCE v2' enp1
mk_gid "$pg" devP 8 7 '::ffff:0a00:2008' 'RoCE v2' enp8
mk_gid "$pg" devP 10 8 '::ffff:0a00:2010' 'RoCE v2' enp10
printf '%s\n' 'enp1 10.0.32.1' 'enp8 10.0.32.8' 'enp10 10.0.32.16' >"$pgfx"

expect_rc "explicit port 1 (baseline for the grammar cases)" 0 "$pg" "$pgfx" 'devP:1' '10.0.99.99'
expect_rc "leading-zero port :08 is decimal 8, not a bad octal literal" 0 "$pg" "$pgfx" 'devP:08' '10.0.99.99'
expect_rc "leading-zero port :010 is decimal 10, not octal 8" 0 "$pg" "$pgfx" 'devP:010' '10.0.99.99'
expect_rc "signed port :+8 parses as 8" 0 "$pg" "$pgfx" 'devP:+8' '10.0.99.99'
expect_rc "atoi stops at the first non-digit (:8abc -> 8)" 0 "$pg" "$pgfx" 'devP:8abc' '10.0.99.99'
expect_rc "atoi does not restart after an embedded LF" 1 "$pg" "$pgfx" $'devP:abc\n8' '10.0.99.99'
expect_rc "atoi stops at the first embedded LF instead of combining lines" 0 "$w" "$wfx" $'devN:1\n+1' '10.0.99.99'
expect_rc "atoi skips leading blanks (: 8 -> 8)" 0 "$pg" "$pgfx" 'devP: 8' '10.0.99.99'
expect_rc "rail/plane fields are parsed off, port still wins (:8:2:0)" 0 "$pg" "$pgfx" 'devP:8:2:0' '10.0.99.99'
expect_rc "non-numeric port is atoi 0 and matches no real port" 1 "$pg" "$pgfx" 'devP:abc' '10.0.99.99'
# Empty port field == absent == any port. All three ports validate here (each
# via its own netdev), so the wildcard reading is observable as success while a
# port-0 reading would instead match nothing and exit 1.
expect_rc "explicit empty port field is a wildcard, not port 0" 0 "$pg" "$pgfx" 'devP:' '10.0.99.99'
expect_rc "empty port field before a rail field is also a wildcard" 0 "$pg" "$pgfx" 'devP::2' '10.0.99.99'
expect_rc "omitted port on a single-port match still validates" 0 "$pg" "$pgfx" 'devP:1:0' '10.0.99.99'

# A port field outside the conservative nine-digit bound must not widen the
# selection. Evaluating arbitrary-width text with $(( )) can wrap modulo 2^64, and 18446744073709551615 wraps to exactly
# -1 — the "any port" wildcard. All ports validate via their own netdevs here,
# so the wildcard reading is observable as success while "matches nothing"
# must exit 1.
expect_rc "port that wraps to the -1 wildcard is clamped, not evaluated" 1 "$pg" "$pgfx" 'devP:18446744073709551615' '10.0.99.99'
expect_rc "absurdly wide port matches no real port" 1 "$pg" "$pgfx" 'devP:99999999999999999999999' '10.0.99.99'
expect_rc "wide negative port matches no real port" 1 "$pg" "$pgfx" 'devP:-99999999999999999999999' '10.0.99.99'
expect_rc "zero padding is not width (:0000000008 is still port 8)" 0 "$pg" "$pgfx" 'devP:0000000008' '10.0.99.99'

# --- per-member index sets validate independently; nothing is reconciled ---
# Each member can have several usable RoCEv2 indexes (devX {1,5}, devY {3,5},
# devZ {2}). Under the unset contract no index is pinned, so overlapping and
# disjoint selections are equally fine — every member just needs at least one
# usable index.
# 10.0.40.1 -> ...0a00:2801 ; 10.0.40.9 -> ...0a00:2809
# 10.0.41.1 -> ...0a00:2901 ; 10.0.41.9 -> ...0a00:2909 ; 10.0.42.1 -> ...0a00:2a01
x="$tmp/multi"
xfx="$tmp/multi.fixture"
mk_gid "$x" devX 1 1 '::ffff:0a00:2801' 'RoCE v2' enx1
mk_gid "$x" devX 1 5 '::ffff:0a00:2809' 'RoCE v2' enx1
mk_gid "$x" devY 1 3 '::ffff:0a00:2901' 'RoCE v2' eny1
mk_gid "$x" devY 1 5 '::ffff:0a00:2909' 'RoCE v2' eny1
mk_gid "$x" devZ 1 2 '::ffff:0a00:2a01' 'RoCE v2' enz1
printf '%s\n' 'enx1 10.0.40.1' 'enx1 10.0.40.9' 'eny1 10.0.41.1' 'eny1 10.0.41.9' 'enz1 10.0.42.1' >"$xfx"

expect_rc "members with overlapping index sets both validate" 0 "$x" "$xfx" '=devX:1,devY:1' '10.0.99.99'
expect_rc "single member with two usable indexes validates" 0 "$x" "$xfx" '=devX:1' '10.0.99.99'
expect_rc "genuinely disjoint index sets still validate (nothing is pinned)" 0 "$x" "$xfx" '=devX:1,devZ:1' '10.0.99.99'

# The audit must show each member's whole usable set, not one arbitrary pick.
rc=0
err="$(resolve "$x" "$xfx" '=devX:1,devZ:1' '10.0.99.99' 2>&1 >/dev/null)" || rc=$?
if [ "$rc" -eq 0 ] \
  && printf '%s\n' "$err" | grep -Fqx '  member devX:1 -> RoCEv2 gid index 1 (via own-addr 10.0.40.1 on enx1)' \
  && printf '%s\n' "$err" | grep -Fqx '  member devX:1 -> RoCEv2 gid index 5 (via own-addr 10.0.40.9 on enx1)' \
  && printf '%s\n' "$err" | grep -Fqx '  member devZ:1 -> RoCEv2 gid index 2 (via own-addr 10.0.42.1 on enz1)'; then
  ok "audit lists each member's full usable set"
else
  bad "usable-set audit wrong (rc=$rc): $err"
fi

# The per-index address-source attribution must be observable: own-address
# validation alone supports 1 and 5 for devQ, and with a match IP of 10.0.48.9
# index 5 is reached "via match-ip" while 1 stays "via own-addr". Both audit
# lines must appear for the same member — no single winner is picked anymore.
# 10.0.48.1 -> ...0a00:3001 ; 10.0.48.9 -> ...0a00:3009
# 10.0.49.1 -> ...0a00:3101 ; 10.0.49.9 -> ...0a00:3109
pref="$tmp/pref"
preffx="$tmp/pref.fixture"
mk_gid "$pref" devQ 1 1 '::ffff:0a00:3001' 'RoCE v2' enq1
mk_gid "$pref" devQ 1 5 '::ffff:0a00:3009' 'RoCE v2' enq1
mk_gid "$pref" devR 1 1 '::ffff:0a00:3101' 'RoCE v2' enr1
mk_gid "$pref" devR 1 5 '::ffff:0a00:3109' 'RoCE v2' enr1
printf '%s\n' 'enq1 10.0.48.1' 'enq1 10.0.48.9' 'enr1 10.0.49.1' 'enr1 10.0.49.9' >"$preffx"

expect_rc "no match-ip hit: own-addr indexes still validate" 0 "$pref" "$preffx" '=devQ:1,devR:1' '10.0.99.99'
expect_rc "match-ip hit validates alongside own-addr hits" 0 "$pref" "$preffx" '=devQ:1,devR:1' '10.0.48.9'
rc=0
err="$(resolve "$pref" "$preffx" '=devQ:1' '10.0.48.9' 2>&1 >/dev/null)" || rc=$?
if [ "$rc" -eq 0 ] \
  && printf '%s\n' "$err" | grep -Fqx '  member devQ:1 -> RoCEv2 gid index 1 (via own-addr 10.0.48.1 on enq1)' \
  && printf '%s\n' "$err" | grep -Fqx '  member devQ:1 -> RoCEv2 gid index 5 (via match-ip 10.0.48.9)'; then
  ok "audit attributes each usable index to its own address source"
else
  bad "per-index source audit wrong (rc=$rc): $err"
fi

# A match-IP member validates next to own-addr members without any pick.
expect_rc "match-ip member validates with own-addr members" 0 "$x" "$xfx" '=devX:1,devY:1' '10.0.40.9'

# --- candidate universe: ncclIbInit skips ports that are not ACTIVE or whose
# link layer is neither Ethernet nor InfiniBand, and it does so *before*
# applying NCCL_IB_HCA. These dual-port cards ship a DOWN sibling port, so
# including it would fail a resolve NCCL itself completes.
# 10.0.80.x -> ...0a00:50xx ; 10.0.81.1 -> ...0a00:5101 ; 10.0.82.1 -> ...0a00:5201
cu="$tmp/candidates"
cufx="$tmp/candidates.fixture"
mk_gid "$cu" devS 1 3 '::ffff:0a00:5001' 'RoCE v2' ens1
mk_port_attr "$cu" devS 1 '4: ACTIVE' 'Ethernet'
mk_gid "$cu" devS 2 9 '::ffff:0a00:5002' 'RoCE v2' ens2
mk_port_attr "$cu" devS 2 '1: DOWN' 'Ethernet'
mk_gid "$cu" devT 1 7 '::ffff:0a00:5101' 'RoCE v2' ent1
mk_port_attr "$cu" devT 1 '5: ACTIVE_DEFER' 'Ethernet'
mk_gid "$cu" devU 1 6 '::ffff:0a00:5201' 'RoCE v2' enu1
mk_port_attr "$cu" devU 1 '4: ACTIVE' 'Unknown'
mk_gid "$cu" devV 1 2 '::ffff:0a00:5301' 'RoCE v2' env1
printf '%s\n' 'ens1 10.0.80.1' 'ens2 10.0.80.2' 'ent1 10.0.81.1' 'enu1 10.0.82.1' 'env1 10.0.83.1' >"$cufx"

expect_rc "DOWN sibling port is not a candidate (omitted port still validates)" 0 "$cu" "$cufx" 'devS' '10.0.99.99'
expect_rc "explicitly selecting a DOWN port matches nothing and fails closed" 1 "$cu" "$cufx" 'devS:2' '10.0.99.99'
expect_rc "ACTIVE_DEFER is not IBV_PORT_ACTIVE" 1 "$cu" "$cufx" '=devT' '10.0.99.99'
expect_rc "unsupported link layer is not a candidate" 1 "$cu" "$cufx" '=devU' '10.0.99.99'
expect_rc "port without readable state attributes stays a candidate" 0 "$cu" "$cufx" '=devV' '10.0.99.99'

# The "matched nothing" FATAL must say why, or a DOWN port looks like a typo.
rc=0
err="$(resolve "$cu" "$cufx" 'devS:2' '10.0.99.99' 2>&1 >/dev/null)" || rc=$?
if [ "$rc" -eq 1 ] && printf '%s' "$err" | grep -q 'not ACTIVE' && printf '%s' "$err" | grep -q 'devS:2'; then
  ok "skipped-candidate diagnostic names the non-ACTIVE port"
else
  bad "candidate diagnostic wrong (rc=$rc): $err"
fi

# NCCL keeps at most MAX_IB_DEVS=32 entries and ignores the rest, so the
# validation must describe those 32 and not a 33rd device NCCL never opens.
# 10.0.90.1 -> ...0a00:5a01
cap="$tmp/maxdevs"
capfx="$tmp/maxdevs.fixture"
: >"$capfx"
for i in $(seq -w 1 32); do
  mk_gid "$cap" "dev$i" 1 4 '::ffff:0a00:5a01' 'RoCE v2'
done
mk_gid "$cap" dev33 1 9 '::ffff:0a00:5a01' 'RoCE v2'

expect_rc "selection is capped at MAX_IB_DEVS=32, mirroring NCCL" 0 "$cap" "$capfx" 'dev' '10.0.90.1'
rc=0
err="$(resolve "$cap" "$capfx" 'dev' '10.0.90.1' 2>&1 >/dev/null)" || rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$err" | grep -q 'truncated at MAX_IB_DEVS=32' \
  && printf '%s' "$err" | grep -q 'dev33:1'; then
  ok "MAX_IB_DEVS truncation is reported with the ignored member"
else
  bad "cap diagnostic wrong (rc=$rc): $err"
fi

# NCCL separately stores only the first 32 non-empty selector entries. Empty
# comma entries do not consume a slot, and entry 33 cannot include or exclude a
# member. This fixture has one member so each boundary outcome is unambiguous.
tokcap="$tmp/selector-cap"
tokcapfx="$tmp/selector-cap.fixture"
: >"$tokcapfx"
mk_gid "$tokcap" devReal 1 9 '::ffff:0a00:5a01' 'RoCE v2'
missing31=""
for i in $(seq -w 1 31); do
  missing31="${missing31}${missing31:+,}miss$i"
done
missing32="$missing31,miss32"
expect_rc "32nd non-empty include entry is retained" 0 "$tokcap" "$tokcapfx" "=$missing31,,devReal," '10.0.90.1'
expect_rc "33rd non-empty include entry is ignored" 1 "$tokcap" "$tokcapfx" "=$missing32,,devReal" '10.0.90.1'
expect_rc "32nd non-empty exclusion entry is retained" 1 "$tokcap" "$tokcapfx" "^=$missing31,,devReal," '10.0.90.1'
expect_rc "33rd non-empty exclusion entry is ignored" 0 "$tokcap" "$tokcapfx" "^=$missing32,,devReal" '10.0.90.1'
rc=0
err="$(resolve "$tokcap" "$tokcapfx" "=$missing32,,devReal" '10.0.90.1' 2>&1 >/dev/null)" || rc=$?
if [ "$rc" -eq 1 ] && printf '%s\n' "$err" \
  | grep -Fqx '  note: selector list truncated to first 32 non-empty entries; NCCL ignores later entries'; then
  ok "selector-list truncation emits the fixed count-only note"
else
  bad "selector-list cap diagnostic wrong (rc=$rc): $err"
fi

# --- independent head/worker layouts in one run (real local/ssh branches) ---
expect_rc "head layout validates independently" 0 "$h" "$hfx" '=devA,devB' '10.0.22.1'
expect_rc "worker layout validates independently over ssh path" 0 "$w" "$wfx" 'devN:2' '10.0.99.99' 'user@worker'

# --- prefix collisions and literal token transport ---
r3="$tmp/roce"
rfx="$tmp/roce.fixture"
: >"$rfx"
mk_gid "$r3" rocep1s0f1 1 3 '::ffff:0a00:1601' 'RoCE v2'
mk_gid "$r3" roceP2p1s0f1 1 3 '::ffff:0a00:1601' 'RoCE v2'

expect_rc "prefix collision: one token matches both HCAs" 0 "$r3" "$rfx" 'roce' '10.0.22.1'
expect_rc "empty selector matches all (agreeing) ports" 0 "$r3" "$rfx" '' '10.0.22.1'
expect_rc "exact form does not prefix-match" 1 "$r3" "$rfx" '=roce' '10.0.22.1'
expect_rc "prefix token shorter than device name" 0 "$r3" "$rfx" 'rocep1' '10.0.22.1'

# netIf::prefix[64] stores at most 63 name bytes before prefix/exact matching.
longroot="$tmp/long-prefix"
longfx="$tmp/long-prefix.fixture"
: >"$longfx"
name63=""
for _ in $(seq 1 63); do name63="${name63}a"; done
mk_gid "$longroot" "$name63" 1 11 '::ffff:0a00:1601' 'RoCE v2'
expect_rc "overlong prefix token is stored as its first 63 bytes" 0 "$longroot" "$longfx" "${name63}x" '10.0.22.1'
expect_rc "overlong exact token is stored as its first 63 bytes" 0 "$longroot" "$longfx" "=${name63}x" '10.0.22.1'

trapdir="$tmp/glob-trap"
mkdir -p "$trapdir"
touch "$trapdir/devA-trap" "$trapdir/rocep1s0f1"
rc=0
( cd "$trapdir" && resolve "$r3" "$rfx" '=dev*' '10.0.22.1' >/dev/null 2>&1 ) || rc=$?
if [ "$rc" -eq 1 ]; then
  ok "glob metacharacters stay literal (no pathname expansion, fail closed)"
else
  bad "glob token misbehaved: rc=$rc (want 1)"
fi
expect_rc "whitespace inside a token transports literally, fails closed" 1 "$r3" "$rfx" '=de vA' '10.0.22.1'
expect_rc "missing sysfs tree fails closed" 1 "$tmp/nonexistent" "$rfx" 'devA' '10.0.22.1'

# --- orchestration contract: AUTO=1 validates then clears both index
# variables (true absence via the entrypoint normalization), and the worker
# env always carries NCCL_IB_GID_INDEX so a stale worker .env.dspark pin is
# masked at compose interpolation; AUTO=0 preserves explicit pins verbatim
# without touching sysfs ---
eval "$(awk '/^remote_nccl_env\(\) \{$/,/^\}$/' "$START")"
pick_gid_match_ip() {
  printf '10.0.22.1'
}
set_orchestration_fixture() {
  NCCL_IB_GID_AUTO=1
  NCCL_IB_HCA='=devA'
  WORKER_NCCL_IB_HCA='=devB'
  NCCL_SOCKET_IFNAME=head0
  WORKER_NCCL_SOCKET_IFNAME=worker0
  WORKER_TP_SOCKET_IFNAME=worker0
  WORKER_GLOO_SOCKET_IFNAME=worker0
  NCCL_IB_GID_MATCH_IP=''
  WORKER_NCCL_IB_GID_MATCH_IP=''
  VLLM_HOST_IP='10.0.22.1'
  WORKER_VLLM_HOST_IP='10.0.22.1'
  MASTER_ADDR='10.0.22.1'
  WORKER_HOST='user@worker'
  VLLM_HOST='127.0.0.1'
  VLLM_PORT='8888'
  ENV_NCCL_IB_GID_INDEX=''
  ENV_WORKER_NCCL_IB_GID_INDEX=''
  NCCL_IB_GID_INDEX=''
  WORKER_NCCL_IB_GID_INDEX=''
  ENV_FILE="$tmp/test.env"
  NCCL_GID_RESOLVE_SYSROOT="$h/sys/class/infiniband"
  IP_FIXTURE="$hfx"
  SSH_EXPECT_TARGET="$WORKER_HOST"
  PATH="$stub:$PATH"
  export NCCL_GID_RESOLVE_SYSROOT IP_FIXTURE SSH_EXPECT_TARGET PATH
}

# AUTO=1: stale pins in the env file are reported and cleared on both ranks;
# remote_nccl_env still emits the key (empty) so a stale worker .env.dspark
# value cannot resurface at compose interpolation.
rc=0
out="$(
  (
    set_orchestration_fixture
    ENV_NCCL_IB_GID_INDEX=7
    ENV_WORKER_NCCL_IB_GID_INDEX=9
    NCCL_IB_GID_INDEX=7
    WORKER_NCCL_IB_GID_INDEX=9
    resolve_nccl_gid_indexes
    printf 'HEAD=<%s> WORKER=<%s>\n' "${NCCL_IB_GID_INDEX:-}" "${WORKER_NCCL_IB_GID_INDEX:-}"
    remote_nccl_env
  ) 2>"$tmp/auto-stale.err"
)" || rc=$?
if [ "$rc" -eq 0 ] \
  && printf '%s\n' "$out" | grep -Fqx 'HEAD=<> WORKER=<>' \
  && printf '%s\n' "$out" | grep -Fq "NCCL_IB_GID_INDEX=''" \
  && printf '%s\n' "$out" | grep -Fq 'ignoring NCCL_IB_GID_INDEX=7' \
  && printf '%s\n' "$out" | grep -Fq 'ignoring WORKER_NCCL_IB_GID_INDEX=9'; then
  ok "AUTO=1 clears stale pins on both ranks and reports them"
else
  bad "AUTO=1 stale-pin clearing wrong (rc=$rc): $out / $(cat "$tmp/auto-stale.err")"
fi

# AUTO=1 with no pins: nothing to report, both ranks still end empty, the
# worker env still carries the (empty) key, and the per-member audits ran.
rc=0
out="$(
  (
    set_orchestration_fixture
    resolve_nccl_gid_indexes
    printf 'HEAD=<%s> WORKER=<%s>\n' "${NCCL_IB_GID_INDEX:-}" "${WORKER_NCCL_IB_GID_INDEX:-}"
    remote_nccl_env
  ) 2>"$tmp/auto-clean.err"
)" || rc=$?
if [ "$rc" -eq 0 ] \
  && printf '%s\n' "$out" | grep -Fqx 'HEAD=<> WORKER=<>' \
  && printf '%s\n' "$out" | grep -Fq "NCCL_IB_GID_INDEX=''" \
  && ! printf '%s\n' "$out" | grep -Fq 'ignoring' \
  && printf '%s\n' "$out" | grep -Fq 'left unset' \
  && grep -Fq 'member devA:1 -> RoCEv2 gid index 3' "$tmp/auto-clean.err" \
  && grep -Fq 'member devB:1 -> RoCEv2 gid index 3' "$tmp/auto-clean.err"; then
  ok "AUTO=1 leaves both ranks unset; worker env key present-but-empty"
else
  bad "AUTO=1 clean-path wrong (rc=$rc): $out / $(cat "$tmp/auto-clean.err")"
fi

# AUTO=0: explicit pins survive verbatim on both ranks and no sysfs tree is
# consulted (the fixture's sysroot points at a nonexistent directory).
rc=0
out="$(
  (
    set_orchestration_fixture
    NCCL_IB_GID_AUTO=0
    ENV_NCCL_IB_GID_INDEX=3
    ENV_WORKER_NCCL_IB_GID_INDEX=6
    NCCL_GID_RESOLVE_SYSROOT="$tmp/nonexistent/sys/class/infiniband"
    export NCCL_GID_RESOLVE_SYSROOT
    resolve_nccl_gid_indexes
    printf 'HEAD=<%s> WORKER=<%s>\n' "${NCCL_IB_GID_INDEX:-}" "${WORKER_NCCL_IB_GID_INDEX:-}"
    remote_nccl_env
  ) 2>"$tmp/pinned.err"
)" || rc=$?
if [ "$rc" -eq 0 ] \
  && printf '%s\n' "$out" | grep -Fqx 'HEAD=<3> WORKER=<6>' \
  && printf '%s\n' "$out" | grep -Fq "NCCL_IB_GID_INDEX='6'" \
  && printf '%s\n' "$out" | grep -Fq 'Using pinned NCCL GID indexes (auto off): head=3 worker=6'; then
  ok "AUTO=0 pins verbatim, no sysfs resolve, worker env carries the pin"
else
  bad "AUTO=0 pin path wrong (rc=$rc): $out / $(cat "$tmp/pinned.err")"
fi

# AUTO=0 without pins still fails closed.
rc=0
err="$(
  (
    set_orchestration_fixture
    NCCL_IB_GID_AUTO=0
    resolve_nccl_gid_indexes
  ) 2>&1 >/dev/null
)" || rc=$?
if [ "$rc" -eq 1 ] && printf '%s\n' "$err" | grep -Fq 'NCCL_IB_GID_AUTO=0 requires NCCL_IB_GID_INDEX'; then
  ok "AUTO=0 without pins fails closed"
else
  bad "AUTO=0 unpinned should fail closed (rc=$rc): $err"
fi

printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
