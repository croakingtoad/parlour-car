#!/usr/bin/env bash
# Parlour production database backup script
# Usage: ./backup.sh [label]

set -Eeuo pipefail

PATH="${PARLOUR_BACKUP_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export PATH

BACKUP_DIR="${PARLOUR_BACKUP_DIR:-/home/marty/parlour-backups}"

# Credentials live in an untracked env file, sourced by absolute path so the
# script behaves identically under cron's minimal environment.
BACKUP_ENV_FILE="${PARLOUR_BACKUP_ENV_FILE:-/home/marty/parlour-backups/backup.env}"
if [[ ! -f "${BACKUP_ENV_FILE}" ]]; then
    echo "[backup] ERROR: backup env file not found at ${BACKUP_ENV_FILE} (see ops/backup.env.example)" >&2
    exit 1
fi
. "${BACKUP_ENV_FILE}"
: "${NEO4J_PASSWORD:?NEO4J_PASSWORD must be set in ${BACKUP_ENV_FILE}}"
LABEL="${1:-manual}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
PREFIX="${TIMESTAMP}_${LABEL}"
LOG_FILE="${BACKUP_DIR}/backup.log"
FAILURE_SENTINEL="${BACKUP_DIR}/BACKUP_FAILED"

PG_CONTAINER="parlour-pg"
NEO4J_CONTAINER="parlour-neo4j"
PG_DB="author_library"
PG_USER="author_library"

PG_MIN_BYTES=1024
NEO4J_MIN_BYTES=20
# Calibration floors: 50% of the newest known-good daily (20260904-030001_daily):
# 15 CREATE TABLE statements, 181944 COPY data rows; 287854 node and 1005258 rels non-empty data lines.
PG_MIN_TABLES=7
PG_MIN_DATA_ROWS=90972
NEO4J_MIN_NODE_ROWS=143927
NEO4J_MIN_REL_ROWS=502629
NEO4J_NODES_HEADER='label, props'
NEO4J_RELS_HEADER='src_label, src_props, rel_type, rel_props, tgt_label, tgt_props'
MAX_TOTAL=50
FAILURE_REPORTED=0
DOCKER_BIN=""
VERIFY_REASON=""
VALIDATED_SIZE=""
VERIFIED_SIZE=""
PG_DUMP_STATS=""
PG_VERIFIED_BYTES=""
NEO4J_NODES_VERIFIED_BYTES=""
NEO4J_RELS_VERIFIED_BYTES=""
declare -A RETENTION_CACHE=()
MULTICA_BIN="${PARLOUR_BACKUP_MULTICA_BIN:-/usr/local/bin/multica}"
MULTICA_HOME="${PARLOUR_BACKUP_MULTICA_HOME:-/home/marty}"
MULTICA_PROJECT_ID="${PARLOUR_BACKUP_MULTICA_PROJECT:-b6d43532-e8d5-4138-98ae-6e1027c48980}"

log_msg() {
    local msg="[backup] $(date '+%Y-%m-%d %H:%M:%S') $*"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

notify_failure() {
    local summary="$1"
    local -a command

    file_multica_failure_issue "$summary"
    [[ -n "${PARLOUR_BACKUP_NOTIFY_CMD:-}" ]] || return 0
    read -r -a command <<< "${PARLOUR_BACKUP_NOTIFY_CMD}"
    if [[ "${#command[@]}" -eq 0 ]] || ! "${command[@]}" "$summary"; then
        log_msg "WARNING: failure notifier failed"
    fi
}

file_multica_failure_issue() {
    local summary="$1"
    local desc_file
    local multica_out
    local issue_id=""

    [[ -x "${MULTICA_BIN}" ]] || { log_msg "WARNING: multica binary not found at ${MULTICA_BIN}; skipping backup failure issue"; return 0; }

    desc_file="${BACKUP_DIR}/.backup-failure-${TIMESTAMP}.md"
    {
        echo "Automated backup failure report from /home/marty/parlour-backups/backup.sh."
        echo
        echo "- Run: ${TIMESTAMP} (${LABEL})"
        echo "- Failure: ${summary}"
        echo
        echo "Recent log (last 50 lines):"
        echo
        tail -n 50 -- "${LOG_FILE}" 2>/dev/null || true
    } > "$desc_file"

    if multica_out=$(HOME="${MULTICA_HOME}" "${MULTICA_BIN}" issue create \
        --title "Backup failure ${TIMESTAMP} (${LABEL})" \
        --description-file "$desc_file" \
        --allow-external-file \
        --project "${MULTICA_PROJECT_ID}" \
        --output json 2>>"${LOG_FILE}"); then
        issue_id=$(printf '%s\n' "${multica_out}" | grep -oE 'LOCO-[0-9]+' | head -n 1 || true)
        if [[ -z "${issue_id}" ]]; then
            issue_id=$(printf '%s\n' "${multica_out}" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -n 1 || true)
        fi
        if [[ -n "${issue_id}" ]]; then
            log_msg "Multica backup failure issue filed: ${issue_id}"
        else
            log_msg "WARNING: multica issue created but the issue id could not be parsed from the output"
        fi
    else
        log_msg "WARNING: multica issue create failed; no backup failure issue was filed"
    fi
    rm -f -- "$desc_file"
    return 0
}

report_failure() {
    local summary="$1"
    local occurred_at

    FAILURE_REPORTED=1
    occurred_at=$(date '+%Y-%m-%d %H:%M:%S %z')
    mkdir -p "${BACKUP_DIR}"
    printf 'timestamp: %s\nfailure: %s\n' "$occurred_at" "$summary" > "${FAILURE_SENTINEL}"
    log_msg "ERROR: ${summary}"
    log_msg "Retention skipped: backup set was not fully verified"
    notify_failure "$summary"
}

handle_unexpected_error() {
    local status="$1"
    local line="$2"

    trap - ERR
    if [[ "${FAILURE_REPORTED}" -eq 0 ]]; then
        report_failure "unexpected backup error at line ${line} (exit ${status})"
    fi
    exit "$status"
}

trap 'handle_unexpected_error "$?" "$LINENO"' ERR

preflight() {
    local container
    local running

    DOCKER_BIN=$(command -v docker || true)
    if [[ -z "${DOCKER_BIN}" ]]; then
        report_failure "preflight: docker binary is not resolvable in PATH=${PATH}"
        return 1
    fi

    for container in "${PG_CONTAINER}" "${NEO4J_CONTAINER}"; do
        running=$("${DOCKER_BIN}" inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)
        if [[ "$running" != "true" ]]; then
            report_failure "preflight: container ${container} is not running"
            return 1
        fi
    done
}

is_pg_structurally_good() {
    local file="$1"

    [[ -f "$file" ]] || return 1
    gzip -t -- "$file" 2>/dev/null || return 1
    gzip -cd -- "$file" 2>/dev/null |
        tail -n 20 |
        grep -Fqx -- '-- PostgreSQL database dump complete'
}

verify_pg_dump_structure() {
    local file="$1"
    local report

    if ! report=$(gzip -cd -- "$file" 2>/dev/null | awk '
        incopy {
            if ($0 == "\\.") { incopy = 0; blocks++ }
            else copy_rows++
            next
        }
        NR == 1 { if ($0 != "--") { print "first line is not the pg_dump header comment"; bad = 1; exit 1 } }
        $0 ~ /^-- Dumped from database version [0-9]/ { if (from == 0) from = NR }
        $0 ~ /^-- Dumped by pg_dump version [0-9]/ { if (by == 0) by = NR }
        $0 ~ /^\\restrict [A-Za-z0-9]+$/ { if (restrict_tok == "") { restrict_tok = substr($0, 11); restrict_ln = NR } }
        $0 ~ /^\\unrestrict [A-Za-z0-9]+$/ { unrestrict_tok = substr($0, 13); unrestrict_ln = NR }
        $0 ~ /^SET / { if (setline == 0) setline = NR }
        $0 ~ /^CREATE TABLE / { creates++; if (first_create == 0) first_create = NR }
        $0 ~ /^COPY .* FROM stdin;$/ { if (first_copy == 0) first_copy = NR; incopy = 1; next }
        $0 ~ /^INSERT INTO / { inserts++ }
        $0 == "-- PostgreSQL database dump complete" { marker = NR; next }
        $0 != "" { last_nonempty = $0 }
        marker {
            if ($0 == "" || $0 == "--" || $0 ~ /^\\unrestrict [A-Za-z0-9]+$/) next
            if (trail_junk == 0) { print "unexpected content after completion marker"; trail_junk = 1 }
        }
        END {
            if (bad) exit 1
            if (from == 0 || by == 0) { print "missing pg_dump version header lines"; exit 1 }
            if (from > by) { print "pg_dump version header lines out of order"; exit 1 }
            if (setline == 0) { print "missing session SET preamble"; exit 1 }
            if (creates == 0) { print "no CREATE TABLE statements"; exit 1 }
            if (blocks == 0 && inserts == 0) { print "no COPY data blocks and no INSERT statements"; exit 1 }
            if (incopy) { print "unterminated COPY data block"; exit 1 }
            if (setline >= first_create) { print "SET preamble not before first CREATE TABLE"; exit 1 }
            if (first_copy != 0 && first_create >= first_copy) { print "first CREATE TABLE not before first COPY block"; exit 1 }
            if (marker == 0) { print "completion marker missing"; exit 1 }
            if (from >= marker || setline >= marker || first_create >= marker) { print "header or DDL after completion marker"; exit 1 }
            if (trail_junk != 0) { print "unexpected content after completion marker"; exit 1 }
            if (restrict_tok != "" && restrict_ln >= from) { print "restrict header not before version lines"; exit 1 }
            if (restrict_tok != "" && unrestrict_tok == "") { print "missing unrestrict trailer"; exit 1 }
            if (unrestrict_tok != "" && (unrestrict_tok != restrict_tok || unrestrict_ln <= marker)) { print "unrestrict trailer does not match restrict header"; exit 1 }
            if (last_nonempty != "-- PostgreSQL database dump complete" && last_nonempty != "\\unrestrict " unrestrict_tok) { print "completion marker is not the last non-empty line"; exit 1 }
            print "ok", creates, (copy_rows + inserts), blocks
            exit 0
        }'
    ); then
        VERIFY_REASON="PostgreSQL dump structure check failed: ${report:-decompression failed}: ${file}"
        return 1
    fi
    PG_DUMP_STATS="$report"
}

verify_neo4j_export_shape() {
    local file="$1"
    local artifact="$2"
    local header="$3"
    local floor="$4"
    local report

    if ! report=$(gzip -cd -- "$file" 2>/dev/null | awk -v header="$header" -v floor="$floor" '
        NR == 1 { if ($0 != header) { print "first line is not the expected column header"; bad = 1; exit 1 } next }
        NF { rows++ }
        END {
            if (bad) exit 1
            if (NR == 0) { print "decompressed content is empty"; exit 1 }
            if (rows + 0 < floor + 0) { print "only " (rows + 0) " non-empty data lines, floor is " floor; exit 1 }
            print "ok " rows
        }'); then
        VERIFY_REASON="${artifact} export does not match the expected cypher-shell plain shape: ${report:-decompression failed}: ${file}"
        return 1
    fi
}

neo4j_pair_same_backup() {
    local nodes_file="$1"
    local rels_file="${nodes_file%.nodes.gz}.rels.gz"
    local cached="${RETENTION_CACHE[pair:${nodes_file}]:-}"
    local report=""

    if [[ -n "$cached" ]]; then
        if [[ "$cached" == 1 ]]; then
            return 0
        fi
        VERIFY_REASON="Neo4j nodes/rels pair rejected by the importer replay contract (schema v1), cached rejection from an earlier check this run: ${nodes_file} / ${rels_file}"
        return 1
    fi
    if [[ ! -f "$nodes_file" || ! -f "$rels_file" ]]; then
        VERIFY_REASON="Neo4j nodes/rels pair rejected by the importer replay contract (schema v1), missing artifact: ${nodes_file} / ${rels_file}"
        return 1
    fi
    if ! report=$(LC_ALL=C awk '
# paircheck.awk — Neo4j backup pair check, schema contract v1
# (ops/neo4j_restore.py @ a35e76e): rejects what the importer cannot replay.
#
# Input: two streams. Stream 1 = decompressed nodes file,
# stream 2 = decompressed rels file.
# Exit 0 = pair consistent; exit 1 = reason on stdout.
# Run under LC_ALL=C (byte semantics). Requires gawk (RT for the
# trailing-newline check; degrades to skipping that check elsewhere).

function fail(msg) {
    if (FAILED) return
    FAILED = 1
    printf "neo4j pair check: %s\n", msg
    exit 1
}

# Record end at position i: the importer requires an LF at this position.
# In our line-joined buffer that is either an embedded \n (internal line
# boundary) or the end of the buffer (final line; its LF is checked via RT).
function rec_end(i) {
    return (i > RL) || (substr(R, i, 1) == "\n")
}

# ---------------------------------------------------------------------------
# Trial parsers (mirror of the importer trial-parse disambiguation).
# They share R/RL and never modify P, FRAW or the entry arrays.
# ---------------------------------------------------------------------------

function tscan_string(start, f, ctx,    rest, pos, sp, nxt, off, rstart) {
    rest = substr(R, start + 1)
    off = start + 1
    while (match(rest, /[\\"]/)) {
        rstart = RSTART
        pos = off + rstart - 1
        sp = substr(rest, rstart, 1)
        if (sp == "\"") { TP = pos + 1; return 1 }
        nxt = substr(rest, rstart + 1, 1)
        if (nxt == "\"") {
            if (trial_ok(pos + 2, f, ctx)) { TP = pos + 2; return 1 }
            rest = substr(rest, rstart + 2)
            off = pos + 2
            continue
        }
        rest = substr(rest, rstart + 1)
        off = pos + 1
    }
    if (length(rest) == 0) return -1
    return 0
}

function tscan_value(start, f, ctx,    c, r, tok) {
    if (start > RL) return 0
    c = substr(R, start, 1)
    if (c == "\"") return tscan_string(start, f, ctx)
    if (c == "[") {
        TP = start + 1
        if (TP > RL) return 0
        if (substr(R, TP, 1) == "]") { TP++; return 1 }
        while (1) {
            r = tscan_value(TP, f, ctx)
            if (r != 1) return r
            if (TP > RL) return 0
            if (substr(R, TP, 1) == "]") { TP++; return 1 }
            if (substr(R, TP, 2) != ", ") return 0
            TP += 2
        }
    }
    if (c == "{") return 0
    if (substr(R, start, 4) == "TRUE") { TP = start + 4; return 1 }
    if (substr(R, start, 5) == "FALSE") { TP = start + 5; return 1 }
    if (match(substr(R, start), /^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/)) {
        tok = substr(R, start, RLENGTH)
        if (tok ~ /\./ || tok ~ /[eE]/) { TP = start + RLENGTH; return 1 }
        if (tok ~ /^-?(0|[1-9][0-9]*)$/) { TP = start + RLENGTH; return 1 }
    }
    return 0
}

function tscan_map(start, f, ctx,    r) {
    TP = start + 1
    if (TP > RL) return 0
    if (substr(R, TP, 1) == "}") { TP++; return 1 }
    while (1) {
        if (!match(substr(R, TP), /^[A-Za-z_][A-Za-z0-9_]*/)) return 0
        TP += RLENGTH
        if (TP > RL || substr(R, TP, 1) != ":") return 0
        TP++
        if (TP > RL || substr(R, TP, 1) != " ") return 0
        TP++
        r = tscan_value(TP, f, ctx)
        if (r != 1) return r
        if (TP > RL) return 0
        if (substr(R, TP, 1) == "}") { TP++; return 1 }
        if (substr(R, TP, 2) != ", ") return 0
        TP += 2
    }
}

# i = position just after the candidate closing quote of a map-value string
# (field f).
function t_map_rest(i, f,    j, r) {
    if (i > RL) return 0
    if (substr(R, i, 1) == "}") return t_field_rest(i + 1, f)
    if (substr(R, i, 2) == ", ") {
        j = i + 2
        while (1) {
            if (!match(substr(R, j), /^[A-Za-z_][A-Za-z0-9_]*/)) return 0
            j += RLENGTH
            if (j > RL || substr(R, j, 1) != ":") return 0
            j++
            if (j > RL || substr(R, j, 1) != " ") return 0
            j++
            TP = j
            r = tscan_value(TP, f, "map")
            if (r != 1) return 0
            j = TP
            if (j > RL) return 0
            if (substr(R, j, 1) == "}") return t_field_rest(j + 1, f)
            if (substr(R, j, 2) != ", ") return 0
            j += 2
        }
    }
    return 0
}

# i = position just after the candidate closing quote of string field f.
function t_field_rest(i, f,    k, j, r) {
    if (f >= NFIELDS) return rec_end(i)
    if (substr(R, i, 2) != ", ") return 0
    j = i + 2
    for (k = f + 1; k <= NFIELDS; k++) {
        if (K[k] == "S") {
            if (j > RL || substr(R, j, 1) != "\"") return 0
            TP = j
            r = tscan_string(TP, k, "field")
            if (r != 1) return 0
            j = TP
        } else {
            if (j > RL || substr(R, j, 1) != "{") return 0
            TP = j
            r = tscan_map(TP, k, "field")
            if (r != 1) return 0
            j = TP
        }
        if (k == NFIELDS) return rec_end(j)
        if (substr(R, j, 2) != ", ") return 0
        j += 2
    }
    return 0
}

function trial_ok(i, f, ctx) {
    if (ctx == "map") return t_map_rest(i, f)
    return t_field_rest(i, f)
}

# ---------------------------------------------------------------------------
# Main parsers. Operate on R/RL at position P.
# Return: 1 = consumed; -1 = buffer ends inside a string (needs more lines);
#         0 = permanent grammar error (ERR set).
# ---------------------------------------------------------------------------

function perr(msg) { ERR = msg; return 0 }

function scan_string(f, ctx,    rest, pos, sp, nxt, off, rstart) {
    rest = substr(R, P + 1)
    off = P + 1
    while (match(rest, /[\\"]/)) {
        rstart = RSTART
        pos = off + rstart - 1
        sp = substr(rest, rstart, 1)
        if (sp == "\"") {
            SRAW = substr(R, P + 1, pos - P - 1)
            P = pos + 1
            return 1
        }
        nxt = substr(rest, rstart + 1, 1)
        if (nxt == "\"") {
            if (trial_ok(pos + 2, f, ctx)) {
                SRAW = substr(R, P + 1, pos - P)
                P = pos + 2
                return 1
            }
            rest = substr(rest, rstart + 2)
            off = pos + 2
            continue
        }
        rest = substr(rest, rstart + 1)
        off = pos + 1
    }
    return -1
}

function scan_value(f, ctx,    c, r, tok) {
    if (P > RL) return -1
    c = substr(R, P, 1)
    if (c == "\"") {
        r = scan_string(f, ctx)
        if (r != 1) return r
        return 1
    }
    if (c == "[") {
        P++
        if (P > RL) return -1
        if (substr(R, P, 1) == "]") { P++; return 1 }
        while (1) {
            r = scan_value(f, ctx)
            if (r != 1) return r
            if (P > RL) return -1
            if (substr(R, P, 1) == "]") { P++; return 1 }
            if (substr(R, P, 2) != ", ") return perr("malformed list")
            P += 2
        }
    }
    if (c == "{") return perr("nested map is not a valid Neo4j property value")
    if (substr(R, P, 4) == "TRUE") { P += 4; return 1 }
    if (substr(R, P, 5) == "FALSE") { P += 5; return 1 }
    if (match(substr(R, P), /^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/)) {
        tok = substr(R, P, RLENGTH)
        if (tok ~ /\./ || tok ~ /[eE]/) { P += RLENGTH; return 1 }
        if (tok ~ /^-?(0|[1-9][0-9]*)$/) { P += RLENGTH; return 1 }
    }
    return perr("unrecognized value at offset " P)
}

function scan_map(f, ctx,    mstart, key, n, r, es, ee) {
    mstart = P
    P++
    if (P > RL) return -1
    if (substr(R, P, 1) == "}") {
        P++
        E_N = 0
        return 1
    }
    delete MAPSEEN
    n = 0
    while (1) {
        if (P > RL) return -1
        if (!match(substr(R, P), /^[A-Za-z_][A-Za-z0-9_]*/)) return perr("invalid map key at offset " P)
        key = substr(R, P, RLENGTH)
        P += RLENGTH
        if (key in MAPSEEN) return perr("duplicate map key " key)
        MAPSEEN[key] = 1
        if (substr(R, P, 1) != ":") return perr("expected colon after map key " key)
        if (substr(R, P + 1, 1) != " ") return perr("expected a space after the colon in map")
        P += 2
        es = P
        r = scan_value(f, ctx)
        if (r != 1) return r
        ee = P - 1
        n++
        EKEY[n] = key
        ESTART[n] = es
        EEND[n] = ee
        if (substr(R, P, 1) == "}") { P++; E_N = n; return 1 }
        if (substr(R, P, 2) != ", ") {
            if (P > RL) return -1
            return perr("malformed map separator at offset " P)
        }
        P += 2
    }
}

# Capture the just-parsed map (EKEY/ESTART/EEND, 1..E_N) as a canonical
# sorted-entries signature string into FMAPSIG[f].
function capture_map_sig(f,    i, s, tk, ts, te, j) {
    for (i = 2; i <= E_N; i++) {
        tk = EKEY[i]; ts = ESTART[i]; te = EEND[i]
        j = i - 1
        while (j >= 1 && EKEY[j] > tk) {
            EKEY[j + 1] = EKEY[j]; ESTART[j + 1] = ESTART[j]; EEND[j + 1] = EEND[j]
            j--
        }
        EKEY[j + 1] = tk; ESTART[j + 1] = ts; EEND[j + 1] = te
    }
    s = ""
    for (i = 1; i <= E_N; i++)
        s = s esc(EKEY[i]) BYTE1F esc(substr(R, ESTART[i], EEND[i] - ESTART[i] + 1)) BYTE1E
    FMAPSIG[f] = s
}

function esc(x) {
    gsub(BYTE01, BYTE01 BYTE01, x)
    gsub(BYTE1E, BYTE01 BYTE1E, x)
    gsub(BYTE1F, BYTE01 BYTE1F, x)
    return x
}

# Parse buf as one complete record. On success fills SIG (nodes) or
# SIGSRC/SIGTGT (rels). Returns 1 / -1 (more lines) / 0 (ERR).
function parse_record(buf, is_nodes,    f, r) {
    R = buf
    RL = length(R)
    P = 1
    ERR = ""
    if (is_nodes) { NFIELDS = 2; K[1] = "S"; K[2] = "M" }
    else {
        NFIELDS = 6
        K[1] = "S"; K[2] = "M"; K[3] = "S"; K[4] = "M"; K[5] = "S"; K[6] = "M"
    }
    for (f = 1; f <= NFIELDS; f++) {
        if (f > 1) {
            if (substr(R, P, 2) != ", ") return perr("expected comma-space between fields at offset " P)
            P += 2
        }
        if (K[f] == "S") {
            if (substr(R, P, 1) != "\"") return perr("expected a quoted string for field " f)
            r = scan_string(f, "field")
            if (r != 1) return r
            FRAW[f] = SRAW
        } else {
            if (substr(R, P, 1) != "{") return perr("expected a map literal for field " f)
            r = scan_map(f, "map")
            if (r != 1) return r
            capture_map_sig(f)
        }
    }
    if (P != RL + 1) return perr("unexpected content after the final field (offset " P ")")
    if (is_nodes) {
        SIG = esc(FRAW[1]) BYTE1E FMAPSIG[2]
    } else {
        SIGSRC = esc(FRAW[1]) BYTE1E FMAPSIG[2]
        SIGTGT = esc(FRAW[5]) BYTE1E FMAPSIG[6]
    }
    return 1
}

# ---------------------------------------------------------------------------
# Line drivers.
# ---------------------------------------------------------------------------

function nodes_line(line,    st) {
    if (NBUF == "") {
        if (line !~ /^"/) fail("nodes file: line " NR " does not start a node record (expected a quoted label)")
        NBUF = line
    } else {
        NBUF = NBUF "\n" line
    }
    st = parse_record(NBUF, 1)
    if (st == 1) {
        node_count++
        sigs[SIG] = 1
        NBUF = ""
    } else if (st == 0) {
        fail("nodes file: " ERR)
    }
}

function rels_line(line,    st) {
    if (RBUF == "") {
        if (line !~ /^"/) fail("rels file: line " NR " does not start a rels record (expected a quoted src_label)")
        RBUF = line
    } else {
        RBUF = RBUF "\n" line
    }
    st = parse_record(RBUF, 0)
    if (st == 1) {
        rel_count++
        if (!(SIGSRC in sigs)) orphan_src++
        if (!(SIGTGT in sigs)) orphan_tgt++
        RBUF = ""
    } else if (st == 0) {
        fail("rels file: " ERR)
    }
}

BEGIN {
    BYTE01 = sprintf("%c", 1)
    BYTE1E = sprintf("%c", 30)
    BYTE1F = sprintf("%c", 31)
    FAILED = 0
    s1 = 0
    s2 = 0
    node_count = 0
    rel_count = 0
    orphan_src = 0
    orphan_tgt = 0
    NBUF = ""
    RBUF = ""
    HAVERT = (PROCINFO["version"] != "")
}

FNR == NR {
    if (FNR == 1) {
        if ($0 != "label, props") fail("nodes file: line 1 is not the exact nodes header (label, props)")
        s1 = 1
        last_rt1 = RT
        next
    }
    if (s1 && HAVERT && last_rt1 == "") fail("nodes file: does not end with a newline (last record truncated?)")
    last_rt1 = RT
    nodes_line($0)
    next
}

{
    if (FNR == 1) {
        if (s1 && HAVERT && last_rt1 == "") fail("nodes file: does not end with a newline (last record truncated?)")
        if ($0 != "src_label, src_props, rel_type, rel_props, tgt_label, tgt_props") fail("rels file: line 1 is not the exact rels header")
        s2 = 1
        last_rt2 = RT
        next
    }
    if (s2 && HAVERT && last_rt2 == "") fail("rels file: does not end with a newline (last record truncated?)")
    last_rt2 = RT
    rels_line($0)
}

END {
    if (FAILED) exit 1
    if (s1 && HAVERT && last_rt1 == "") fail("nodes file: does not end with a newline (last record truncated?)")
    if (s2 && HAVERT && last_rt2 == "") fail("rels file: does not end with a newline (last record truncated?)")
    if (NBUF != "") fail("nodes file: unterminated record at end of input")
    if (RBUF != "") fail("rels file: unterminated record at end of input")
    if (node_count == 0) fail("nodes file has no data records (header-only)")
    if (rel_count == 0) fail("rels file has no data records (header-only)")
    if (orphan_src + orphan_tgt > 0)
        fail(orphan_src + orphan_tgt " rel endpoint(s) do not resolve to any node in the nodes file (orphan src=" orphan_src " tgt=" orphan_tgt "); inconsistent pair, the importer would refuse it")
    exit 0
}
' <(gzip -cd -- "$nodes_file" 2>/dev/null) <(gzip -cd -- "$rels_file" 2>/dev/null)); then
        VERIFY_REASON="Neo4j nodes/rels pair rejected by the importer replay contract (schema v1): ${report:-decompression or parse failure}: ${nodes_file} / ${rels_file}"
        RETENTION_CACHE[pair:${nodes_file}]=0
        return 1
    fi
    RETENTION_CACHE[pair:${nodes_file}]=1
}

is_pg_retention_good() {
    local file="$1"
    local cached="${RETENTION_CACHE[pg:${file}]:-}"
    local saved_reason="$VERIFY_REASON"
    local saved_stats="$PG_DUMP_STATS"

    if [[ -n "$cached" ]]; then
        VERIFY_REASON="$saved_reason"
        PG_DUMP_STATS="$saved_stats"
        if [[ "$cached" == 1 ]]; then
            return 0
        fi
        return 1
    fi
    if is_pg_structurally_good "$file" &&
        (( $(stat -c %s -- "$file") >= PG_MIN_BYTES )) &&
        verify_pg_dump_structure "$file"; then
        RETENTION_CACHE[pg:${file}]=1
    else
        RETENTION_CACHE[pg:${file}]=0
    fi
    VERIFY_REASON="$saved_reason"
    PG_DUMP_STATS="$saved_stats"
    if [[ "${RETENTION_CACHE[pg:${file}]}" == 1 ]]; then
        return 0
    fi
    return 1
}

is_neo4j_pair_retention_good() {
    local nodes_file="$1"
    local rels_file="${nodes_file%.nodes.gz}.rels.gz"
    local cached="${RETENTION_CACHE[neo4j:${nodes_file}]:-}"
    local saved_reason="$VERIFY_REASON"

    if [[ -n "$cached" ]]; then
        VERIFY_REASON="$saved_reason"
        if [[ "$cached" == 1 ]]; then
            return 0
        fi
        return 1
    fi
    if [[ -f "$nodes_file" && -f "$rels_file" ]] &&
        (( $(stat -c %s -- "$nodes_file") >= NEO4J_MIN_BYTES )) &&
        (( $(stat -c %s -- "$rels_file") >= NEO4J_MIN_BYTES )) &&
        verify_neo4j_export_shape "$nodes_file" "Neo4j nodes" "$NEO4J_NODES_HEADER" "$NEO4J_MIN_NODE_ROWS" &&
        verify_neo4j_export_shape "$rels_file" "Neo4j relationships" "$NEO4J_RELS_HEADER" "$NEO4J_MIN_REL_ROWS" &&
        neo4j_pair_same_backup "$nodes_file"; then
        RETENTION_CACHE[neo4j:${nodes_file}]=1
    else
        RETENTION_CACHE[neo4j:${nodes_file}]=0
    fi
    VERIFY_REASON="$saved_reason"
    if [[ "${RETENTION_CACHE[neo4j:${nodes_file}]}" == 1 ]]; then
        return 0
    fi
    return 1
}

label_family() {
    case "$1" in
        weekly*) echo weekly ;;
        daily*) echo daily ;;
        manual*) echo manual ;;
        post-ingest*) echo post-ingest ;;
        pre-ingest*) echo pre-ingest ;;
        *) echo other ;;
    esac
}

artifact_family() {
    local name="${1##*/}"

    case "$name" in
        *_weekly*) echo weekly ;;
        *_daily*) echo daily ;;
        *_manual*) echo manual ;;
        *_post-ingest*) echo post-ingest ;;
        *_pre-ingest*) echo pre-ingest ;;
        *) echo other ;;
    esac
}

family_pattern() {
    local family="$1"
    local suffix="$2"

    case "$family" in
        weekly) echo "*_weekly*${suffix}" ;;
        daily) echo "*_daily*${suffix}" ;;
        manual) echo "*_manual*${suffix}" ;;
        post-ingest) echo "*_post-ingest*${suffix}" ;;
        pre-ingest) echo "*_pre-ingest*${suffix}" ;;
        *) echo "*${suffix}" ;;
    esac
}

newest_good_pg() {
    local pattern="$1"
    local exclude="$2"
    local candidate
    local -a files=( "${BACKUP_DIR}/pg/"${pattern} )

    NEWEST_GOOD=""
    for candidate in "${files[@]}"; do
        [[ "$candidate" != "$exclude" ]] || continue
        if is_pg_retention_good "$candidate" &&
            { [[ -z "$NEWEST_GOOD" ]] || [[ "$candidate" -nt "$NEWEST_GOOD" ]]; }; then
            NEWEST_GOOD="$candidate"
        fi
    done
}

newest_good_neo4j_pair() {
    local pattern="$1"
    local exclude="$2"
    local candidate
    local -a files=( "${BACKUP_DIR}/neo4j/"${pattern} )

    NEWEST_GOOD=""
    for candidate in "${files[@]}"; do
        [[ "$candidate" != "$exclude" ]] || continue
        if is_neo4j_pair_retention_good "$candidate" &&
            { [[ -z "$NEWEST_GOOD" ]] || [[ "$candidate" -nt "$NEWEST_GOOD" ]]; }; then
            NEWEST_GOOD="$candidate"
        fi
    done
}

validated_stat_size() {
    local file="$1"
    local artifact="$2"
    local size

    if ! size=$(stat -c %s -- "$file"); then
        VERIFY_REASON="${artifact} byte-count failure: stat exited non-zero for ${file}"
        return 1
    fi
    if [[ -z "$size" || ! "$size" =~ ^[0-9]+$ ]]; then
        VERIFY_REASON="${artifact} byte-count failure: stat returned a non-decimal value for ${file}"
        return 1
    fi
    VALIDATED_SIZE="$size"
}

size_is_sane() {
    local file="$1"
    local reference="$2"
    local floor="$3"
    local size
    local reference_size

    if ! validated_stat_size "$file" "artifact"; then
        return 1
    fi
    size="$VALIDATED_SIZE"
    if [[ -n "$reference" ]]; then
        if ! validated_stat_size "$reference" "reference artifact"; then
            return 1
        fi
        reference_size="$VALIDATED_SIZE"
        (( size * 2 >= reference_size ))
    else
        (( size >= floor ))
    fi
}

verify_and_log_gzip_artifact() {
    local file="$1"
    local artifact="$2"
    local size

    if [[ ! -f "$file" ]]; then
        VERIFY_REASON="${artifact} artifact is missing: ${file}"
        log_msg "Gzip integrity FAIL: ${file} exists=FAIL bytes=unavailable gzip -t=NOT-RUN (missing)"
        return 1
    fi

    if ! validated_stat_size "$file" "$artifact artifact"; then
        log_msg "Gzip integrity FAIL: ${file} exists=PASS bytes=unavailable gzip -t=NOT-RUN (byte-count failure)"
        return 1
    fi
    size="$VALIDATED_SIZE"
    if ! gzip -t -- "$file" 2>/dev/null; then
        VERIFY_REASON="${artifact} artifact failed gzip integrity check: ${file}"
        log_msg "Gzip integrity FAIL: ${file} exists=PASS bytes=${size} gzip -t=FAIL"
        return 1
    fi
    if (( size == 0 )); then
        VERIFY_REASON="${artifact} artifact is zero-length: ${file}"
        log_msg "Gzip integrity FAIL: ${file} exists=PASS bytes=0 gzip -t=PASS (zero-length)"
        return 1
    fi

    log_msg "Gzip integrity PASS: ${file} exists=PASS bytes=${size} gzip -t=PASS"
}

log_verification_result() {
    local file="$1"
    local result="$2"
    local reason="${3:-}"
    local exists="FAIL"
    local size="unavailable"

    if [[ ! -f "$file" ]]; then
        if [[ "$result" == "PASS" ]]; then
            VERIFY_REASON="artifact byte-count failure: file is missing before terminal verification: ${file}"
            log_msg "Verification FAIL: ${file} exists=${exists} bytes=${size} reason=${VERIFY_REASON}"
            return 1
        fi

        log_msg "Verification FAIL: ${file} exists=${exists} bytes=${size} reason=${reason}"
        return 0
    fi

    exists="PASS"
    if ! validated_stat_size "$file" "artifact"; then
        log_msg "Verification FAIL: ${file} exists=${exists} bytes=unavailable reason=${VERIFY_REASON}"
        return 1
    fi
    size="$VALIDATED_SIZE"

    if [[ "$result" == "PASS" ]]; then
        log_msg "Verification PASS: ${file} exists=${exists} bytes=${size} all-checks=PASS"
        VERIFIED_SIZE="$size"
    fi
}

verify_pg_artifact() {
    local file="$1"
    local family="$2"
    local pattern
    local pg_tables
    local pg_data_rows

    VERIFY_REASON=""
    if ! verify_and_log_gzip_artifact "$file" "PostgreSQL"; then
        log_verification_result "$file" FAIL "$VERIFY_REASON" || return 1
        return 1
    fi
    if ! is_pg_structurally_good "$file"; then
        VERIFY_REASON="PostgreSQL artifact lacks the completion marker: ${file}"
        log_verification_result "$file" FAIL "$VERIFY_REASON" || return 1
        return 1
    fi
    if ! verify_pg_dump_structure "$file"; then
        log_verification_result "$file" FAIL "$VERIFY_REASON" || return 1
        return 1
    fi
    read -r _ pg_tables pg_data_rows _ <<< "${PG_DUMP_STATS}"
    if (( pg_tables < PG_MIN_TABLES )) || (( pg_data_rows < PG_MIN_DATA_ROWS )); then
        VERIFY_REASON="PostgreSQL dump below calibrated floors: ${pg_tables} CREATE TABLE statements (floor ${PG_MIN_TABLES}), ${pg_data_rows} data rows (floor ${PG_MIN_DATA_ROWS}): ${file}"
        log_verification_result "$file" FAIL "$VERIFY_REASON" || return 1
        return 1
    fi
    pattern=$(family_pattern "$family" '.sql.gz')
    newest_good_pg "$pattern" "$file"
    if ! size_is_sane "$file" "$NEWEST_GOOD" "$PG_MIN_BYTES"; then
        if [[ -z "$VERIFY_REASON" ]]; then
            VERIFY_REASON="PostgreSQL artifact failed size sanity against ${NEWEST_GOOD:-${PG_MIN_BYTES}-byte floor}: ${file}"
        fi
        log_verification_result "$file" FAIL "$VERIFY_REASON" || return 1
        return 1
    fi
    log_verification_result "$file" PASS || return 1
    PG_VERIFIED_BYTES="$VERIFIED_SIZE"
}

verify_neo4j_pair() {
    local nodes_file="$1"
    local family="$2"
    local rels_file="${nodes_file%.nodes.gz}.rels.gz"
    local pattern
    local reference_nodes
    local reference_rels=""
    local nodes_gzip_ok=0
    local rels_gzip_ok=0
    local nodes_rows_ok=0
    local rels_rows_ok=0
    local nodes_size_ok=0
    local rels_size_ok=0
    local first_failure=""
    local reason
    local final_verification_failed=0

    VERIFY_REASON=""

    if verify_and_log_gzip_artifact "$nodes_file" "Neo4j nodes"; then
        nodes_gzip_ok=1
    else
        reason="$VERIFY_REASON"
        log_verification_result "$nodes_file" FAIL "$reason"
        first_failure="$reason"
    fi
    if verify_and_log_gzip_artifact "$rels_file" "Neo4j relationships"; then
        rels_gzip_ok=1
    else
        reason="$VERIFY_REASON"
        log_verification_result "$rels_file" FAIL "$reason"
        [[ -n "$first_failure" ]] || first_failure="$reason"
    fi

    if (( nodes_gzip_ok == 0 || rels_gzip_ok == 0 )); then
        if (( nodes_gzip_ok == 1 )); then
            reason="Neo4j nodes artifact cannot complete pair verification because the relationships artifact failed gzip integrity: ${rels_file}"
            log_verification_result "$nodes_file" FAIL "$reason"
            [[ -n "$first_failure" ]] || first_failure="$reason"
        fi
        if (( rels_gzip_ok == 1 )); then
            reason="Neo4j relationships artifact cannot complete pair verification because the nodes artifact failed gzip integrity: ${nodes_file}"
            log_verification_result "$rels_file" FAIL "$reason"
            [[ -n "$first_failure" ]] || first_failure="$reason"
        fi
        VERIFY_REASON="$first_failure"
        return 1
    fi

    if verify_neo4j_export_shape "$nodes_file" "Neo4j nodes" "$NEO4J_NODES_HEADER" "$NEO4J_MIN_NODE_ROWS"; then
        nodes_rows_ok=1
    else
        reason="$VERIFY_REASON"
        log_verification_result "$nodes_file" FAIL "$reason"
        first_failure="$reason"
    fi
    if verify_neo4j_export_shape "$rels_file" "Neo4j relationships" "$NEO4J_RELS_HEADER" "$NEO4J_MIN_REL_ROWS"; then
        rels_rows_ok=1
    else
        reason="$VERIFY_REASON"
        log_verification_result "$rels_file" FAIL "$reason"
        [[ -n "$first_failure" ]] || first_failure="$reason"
    fi
    if (( nodes_rows_ok == 0 || rels_rows_ok == 0 )); then
        if (( nodes_rows_ok == 1 )); then
            reason="Neo4j nodes artifact cannot complete pair verification because the relationships artifact failed the cypher-shell plain shape check: ${rels_file}"
            log_verification_result "$nodes_file" FAIL "$reason"
        fi
        if (( rels_rows_ok == 1 )); then
            reason="Neo4j relationships artifact cannot complete pair verification because the nodes artifact failed the cypher-shell plain shape check: ${nodes_file}"
            log_verification_result "$rels_file" FAIL "$reason"
        fi
        VERIFY_REASON="$first_failure"
        return 1
    fi

    if ! neo4j_pair_same_backup "$nodes_file"; then
        reason="$VERIFY_REASON"
        log_verification_result "$nodes_file" FAIL "$reason"
        log_verification_result "$rels_file" FAIL "$reason"
        VERIFY_REASON="$reason"
        return 1
    fi

    pattern=$(family_pattern "$family" '.dump.nodes.gz')
    newest_good_neo4j_pair "$pattern" "$nodes_file"
    reference_nodes="$NEWEST_GOOD"
    [[ -z "$reference_nodes" ]] || reference_rels="${reference_nodes%.nodes.gz}.rels.gz"
    if size_is_sane "$nodes_file" "$reference_nodes" "$NEO4J_MIN_BYTES"; then
        nodes_size_ok=1
    else
        reason="$VERIFY_REASON"
        [[ -n "$reason" ]] || reason="Neo4j nodes artifact failed size sanity against ${reference_nodes:-${NEO4J_MIN_BYTES}-byte floor}: ${nodes_file}"
        log_verification_result "$nodes_file" FAIL "$reason"
        first_failure="$reason"
    fi
    if size_is_sane "$rels_file" "$reference_rels" "$NEO4J_MIN_BYTES"; then
        rels_size_ok=1
    else
        reason="$VERIFY_REASON"
        [[ -n "$reason" ]] || reason="Neo4j rels artifact failed size sanity against ${reference_rels:-${NEO4J_MIN_BYTES}-byte floor}: ${rels_file}"
        log_verification_result "$rels_file" FAIL "$reason"
        [[ -n "$first_failure" ]] || first_failure="$reason"
    fi
    if (( nodes_size_ok == 0 || rels_size_ok == 0 )); then
        if (( nodes_size_ok == 1 )); then
            reason="Neo4j nodes artifact cannot complete pair verification because the relationships artifact failed size sanity: ${rels_file}"
            log_verification_result "$nodes_file" FAIL "$reason"
        fi
        if (( rels_size_ok == 1 )); then
            reason="Neo4j relationships artifact cannot complete pair verification because the nodes artifact failed size sanity: ${nodes_file}"
            log_verification_result "$rels_file" FAIL "$reason"
        fi
        VERIFY_REASON="$first_failure"
        return 1
    fi
    if ! log_verification_result "$nodes_file" PASS; then
        final_verification_failed=1
    else
        NEO4J_NODES_VERIFIED_BYTES="$VERIFIED_SIZE"
    fi
    if ! log_verification_result "$rels_file" PASS; then
        final_verification_failed=1
    else
        NEO4J_RELS_VERIFIED_BYTES="$VERIFIED_SIZE"
    fi
    (( final_verification_failed == 0 )) || return 1
}

verify_weekly_neo4j_partial_artifact() {
    local file="$1"
    local artifact="$2"
    local peer="$3"
    local reason

    if verify_and_log_gzip_artifact "$file" "$artifact"; then
        reason="${artifact} artifact cannot complete pair verification because ${peer} copy failed"
    else
        reason="$VERIFY_REASON"
    fi
    log_verification_result "$file" FAIL "$reason" || return 1
    VERIFY_REASON="$reason"
    return 1
}

prune_pg_group() {
    local pattern="$1"
    local keep="$2"
    local description="$3"
    local candidate
    local newest_good=""
    local good_count=0
    local count
    local good
    local family
    local protected_file
    local protected_by_family=0
    local -a files=( "${BACKUP_DIR}/pg/"${pattern} )
    local -A family_protected=()

    count=${#files[@]}
    (( count > keep )) || return 0
    [[ "$pattern" != '*.sql.gz' ]] || protected_by_family=1
    for candidate in "${files[@]}"; do
        if is_pg_retention_good "$candidate"; then
            ((good_count += 1))
            if [[ -z "$newest_good" || "$candidate" -nt "$newest_good" ]]; then
                newest_good="$candidate"
            fi
            if (( protected_by_family == 1 )); then
                family=$(artifact_family "$candidate")
                protected_file="${family_protected[$family]:-}"
                if [[ -z "$protected_file" || "$candidate" -nt "$protected_file" ]]; then
                    family_protected[$family]="$candidate"
                fi
            fi
        fi
    done
    for candidate in "${files[@]}"; do
        (( count > keep )) || break
        [[ "$candidate" != "$newest_good" ]] || continue
        if (( protected_by_family == 1 )); then
            family=$(artifact_family "$candidate")
            [[ "$candidate" != "${family_protected[$family]:-}" ]] || continue
        fi
        good=0
        is_pg_retention_good "$candidate" && good=1
        if (( good == 1 && good_count <= 1 )); then
            continue
        fi
        rm -f -- "$candidate"
        log_msg "Pruned ${description}: ${candidate}"
        ((count -= 1))
        (( good == 0 )) || ((good_count -= 1))
    done
}

prune_neo4j_group() {
    local pattern="$1"
    local keep="$2"
    local description="$3"
    local candidate
    local rels_file
    local newest_good=""
    local good_count=0
    local count
    local good
    local family
    local protected_file
    local protected_by_family=0
    local -a files=( "${BACKUP_DIR}/neo4j/"${pattern} )
    local -A family_protected=()

    count=${#files[@]}
    (( count > keep )) || return 0
    [[ "$pattern" != '*.dump.nodes.gz' ]] || protected_by_family=1
    for candidate in "${files[@]}"; do
        if is_neo4j_pair_retention_good "$candidate"; then
            ((good_count += 1))
            if [[ -z "$newest_good" || "$candidate" -nt "$newest_good" ]]; then
                newest_good="$candidate"
            fi
            if (( protected_by_family == 1 )); then
                family=$(artifact_family "$candidate")
                protected_file="${family_protected[$family]:-}"
                if [[ -z "$protected_file" || "$candidate" -nt "$protected_file" ]]; then
                    family_protected[$family]="$candidate"
                fi
            fi
        fi
    done
    for candidate in "${files[@]}"; do
        (( count > keep )) || break
        [[ "$candidate" != "$newest_good" ]] || continue
        if (( protected_by_family == 1 )); then
            family=$(artifact_family "$candidate")
            [[ "$candidate" != "${family_protected[$family]:-}" ]] || continue
        fi
        good=0
        is_neo4j_pair_retention_good "$candidate" && good=1
        if (( good == 1 && good_count <= 1 )); then
            continue
        fi
        rels_file="${candidate%.nodes.gz}.rels.gz"
        rm -f -- "$candidate" "$rels_file"
        log_msg "Pruned ${description} pair: ${candidate} / ${rels_file}"
        ((count -= 1))
        (( good == 0 )) || ((good_count -= 1))
    done
}

apply_retention() {
    log_msg "Applying retention policy..."

    prune_pg_group '*_weekly*.sql.gz' 4 'weekly PG'
    prune_neo4j_group '*_weekly*.dump.nodes.gz' 4 'weekly Neo4j'
    prune_pg_group '*_daily*.sql.gz' 7 'daily PG'
    prune_pg_group '*_manual*.sql.gz' 7 'manual PG'
    prune_neo4j_group '*_daily*.dump.nodes.gz' 7 'daily Neo4j'
    prune_neo4j_group '*_manual*.dump.nodes.gz' 7 'manual Neo4j'
    prune_pg_group '*_post-ingest*.sql.gz' 30 'post-ingest PG'
    prune_pg_group '*_pre-ingest*.sql.gz' 10 'pre-ingest PG'
    prune_neo4j_group '*_post-ingest*.dump.nodes.gz' 30 'post-ingest Neo4j'
    prune_neo4j_group '*_pre-ingest*.dump.nodes.gz' 10 'pre-ingest Neo4j'

    prune_pg_group '*.sql.gz' "$MAX_TOTAL" 'total PG'
    prune_neo4j_group '*.dump.nodes.gz' "$MAX_TOTAL" 'total Neo4j'
}

run_backup() {
    local family
    local pg_file
    local neo4j_base
    local nodes_file
    local rels_file
    local weekly_pg
    local weekly_nodes
    local weekly_rels
    local weekly_pg_copied=0
    local weekly_nodes_copied=0
    local weekly_rels_copied=0
    local reason
    local -a failures=()
    local -a weekly_failures=()
    local failure_summary

    preflight
    mkdir -p "${BACKUP_DIR}/pg" "${BACKUP_DIR}/neo4j"
    log_msg "Starting backup: ${PREFIX}"
    family=$(label_family "$LABEL")

    pg_file="${BACKUP_DIR}/pg/${PREFIX}.sql.gz"
    log_msg "PostgreSQL -> ${pg_file}"
    if ! "${DOCKER_BIN}" exec "${PG_CONTAINER}" pg_dump -U "${PG_USER}" "${PG_DB}" --no-owner --no-acl 2>/dev/null |
        gzip > "$pg_file"; then
        rm -f -- "$pg_file"
        reason="PostgreSQL export command failed"
        log_verification_result "$pg_file" FAIL "$reason" || true
        failures+=("$reason")
    elif ! verify_pg_artifact "$pg_file" "$family"; then
        failures+=("$VERIFY_REASON")
    else
        log_msg "PostgreSQL verified: ${PG_VERIFIED_BYTES} bytes"
    fi

    neo4j_base="${BACKUP_DIR}/neo4j/${PREFIX}.dump"
    nodes_file="${neo4j_base}.nodes.gz"
    rels_file="${neo4j_base}.rels.gz"
    log_msg "Neo4j -> ${nodes_file} / ${rels_file}"
    if ! "${DOCKER_BIN}" exec "${NEO4J_CONTAINER}" cypher-shell -u neo4j -p "${NEO4J_PASSWORD}" --format plain \
        'MATCH (n) RETURN labels(n)[0] as label, properties(n) as props' 2>/dev/null |
        gzip > "$nodes_file"; then
        rm -f -- "$nodes_file"
        failures+=("Neo4j nodes export command failed")
    fi
    if ! "${DOCKER_BIN}" exec "${NEO4J_CONTAINER}" cypher-shell -u neo4j -p "${NEO4J_PASSWORD}" --format plain \
        'MATCH (n)-[r]->(m) RETURN labels(n)[0] as src_label, properties(n) as src_props, type(r) as rel_type, properties(r) as rel_props, labels(m)[0] as tgt_label, properties(m) as tgt_props' 2>/dev/null |
        gzip > "$rels_file"; then
        rm -f -- "$rels_file"
        failures+=("Neo4j relationships export command failed")
    fi
    if ! verify_neo4j_pair "$nodes_file" "$family"; then
        failures+=("$VERIFY_REASON")
    else
        log_msg "Neo4j verified: nodes=${NEO4J_NODES_VERIFIED_BYTES} bytes, rels=${NEO4J_RELS_VERIFIED_BYTES} bytes"
    fi

    if (( ${#failures[@]} > 0 )); then
        failure_summary=$(printf '%s; ' "${failures[@]}")
        report_failure "${failure_summary%; }"
        return 1
    fi

    if [[ "$LABEL" == daily && "$(date +%u)" == 7 ]]; then
        weekly_pg="${BACKUP_DIR}/pg/${TIMESTAMP}_weekly.sql.gz"
        weekly_nodes="${BACKUP_DIR}/neo4j/${TIMESTAMP}_weekly.dump.nodes.gz"
        weekly_rels="${weekly_nodes%.nodes.gz}.rels.gz"
        log_msg "Sunday detected - creating verified weekly copies"
        if cp -- "$pg_file" "$weekly_pg"; then
            weekly_pg_copied=1
        else
            reason="Weekly PostgreSQL promotion copy failed: ${pg_file} -> ${weekly_pg}"
            log_verification_result "$weekly_pg" FAIL "$reason" || true
            weekly_failures+=("$reason")
        fi
        if cp -- "$nodes_file" "$weekly_nodes"; then
            weekly_nodes_copied=1
        else
            reason="Weekly Neo4j nodes promotion copy failed: ${nodes_file} -> ${weekly_nodes}"
            log_verification_result "$weekly_nodes" FAIL "$reason" || true
            weekly_failures+=("$reason")
        fi
        if cp -- "$rels_file" "$weekly_rels"; then
            weekly_rels_copied=1
        else
            reason="Weekly Neo4j relationships promotion copy failed: ${rels_file} -> ${weekly_rels}"
            log_verification_result "$weekly_rels" FAIL "$reason" || true
            weekly_failures+=("$reason")
        fi
        if (( weekly_pg_copied == 1 )) && ! verify_pg_artifact "$weekly_pg" weekly; then
            weekly_failures+=("$VERIFY_REASON")
        fi
        if (( weekly_nodes_copied == 1 && weekly_rels_copied == 1 )) && ! verify_neo4j_pair "$weekly_nodes" weekly; then
            weekly_failures+=("$VERIFY_REASON")
        elif (( weekly_nodes_copied == 1 && weekly_rels_copied == 0 )); then
            if ! verify_weekly_neo4j_partial_artifact "$weekly_nodes" "Neo4j nodes" "the relationships artifact"; then
                weekly_failures+=("$VERIFY_REASON")
            fi
        elif (( weekly_nodes_copied == 0 && weekly_rels_copied == 1 )); then
            if ! verify_weekly_neo4j_partial_artifact "$weekly_rels" "Neo4j relationships" "the nodes artifact"; then
                weekly_failures+=("$VERIFY_REASON")
            fi
        fi
        if (( ${#weekly_failures[@]} > 0 )); then
            failure_summary=$(printf '%s; ' "${weekly_failures[@]}")
            report_failure "weekly promotion failed verification: ${failure_summary%; }"
            return 1
        fi
    fi

    apply_retention
    rm -f -- "$FAILURE_SENTINEL"
    log_msg "Complete. PG backups: $(find "${BACKUP_DIR}/pg" -type f -name '*.sql.gz' | wc -l), Neo4j backups: $(find "${BACKUP_DIR}/neo4j" -type f -name '*.nodes.gz' | wc -l), Total size: $(du -sh "${BACKUP_DIR}" | cut -f1)"
}

shopt -s nullglob
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    run_backup
fi
