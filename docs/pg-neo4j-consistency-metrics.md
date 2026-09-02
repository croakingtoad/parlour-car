# PostgreSQL / Neo4j consistency metrics

These metrics have similar names but select different populations. They must not
be used interchangeably.

## Live snapshot

All measurements below were read-only and were taken from the local live
`parlour-pg` and `parlour-neo4j` services between
**2026-09-02 16:37:50 and 16:39:00 EDT** (`2026-09-02T20:37:50Z` through
`2026-09-02T20:39:00Z`). PostgreSQL and Neo4j each contained 22 works and
86,749 chunk rows/nodes. Neo4j had 86,749 distinct `chunk_id` values.

| Metric | Measured value | What it actually selects | Correct remedy |
|---|---:|---|---|
| `audit_library.pg_neo4j.is_consistent` | `true`; `chunk_delta: []` | Work-ID set equality and per-work chunk-count equality | None for the empty count-delta population; this metric cannot diagnose identity drift |
| `audit_library.works[*].orphaned_neo4j_chunks` | 2,601 total | Neo4j `Chunk` nodes with no outgoing edge of five entity types | None on this evidence alone; absence of an entity edge is not PG/Neo4j drift |
| `/parlour-booklist` PG-only IDs | 34 | PostgreSQL chunk IDs absent from Neo4j | `backfill_graph_and_entities.py` for each affected work |
| `/parlour-booklist` Neo4j-only IDs | 34 | Neo4j chunk IDs absent from PostgreSQL | `cleanup_neo4j_orphans.py`, but its current hard-coded work list must first be corrected and its dry-run reviewed |

The correct answer to “is PostgreSQL in sync with Neo4j?” is therefore **no**:
the stores have equal chunk cardinality but different identities—34 current
PostgreSQL IDs are missing from Neo4j and 34 stale Neo4j IDs are absent from
PostgreSQL. Removing hyphens from both sides before comparison still produces
34 IDs in each direction, so this is identity drift, not UUID formatting alone.

## 1. `audit_library.pg_neo4j.is_consistent`

**Source:** `src/author_library/graph/backfill.py:50-68` and
`src/author_library/graph/backfill.py:357-408`, packaged for the audit at
`src/author_library/tools/meta.py:501-520`.

**Queries (verbatim):**

```sql
SELECT work_id, title, author, source_class, publication_year FROM works ORDER BY work_id
```

```cypher
MATCH (w:Work) RETURN w.work_id AS work_id
```

```sql
SELECT work_id, COUNT(*) AS chunk_count FROM chunks GROUP BY work_id
```

```cypher
MATCH (c:Chunk) RETURN c.work_id AS work_id, COUNT(c) AS chunk_count
```

The function makes sets of the PostgreSQL and Neo4j work IDs, then compares the
two chunk counts for every work ID. `is_consistent` is true when the work-ID sets
are equal and every pair of per-work counts is equal.

**Population:** missing/extra work IDs and works whose aggregate chunk counts
differ.

**It does not measure:** chunk-ID equality, duplicate `chunk_id` values, chunk
content/property equality, or relationship equality. Equal numbers of missing
and stale chunks within one work cancel out.

**Measured value:** `true`, 22 PostgreSQL works, 22 Neo4j works, no missing or
extra work IDs, and `chunk_delta: []`. The three affected works each have equal
counts while hiding equal-and-opposite ID differences:

| Work | PG-only IDs | Neo4j-only IDs | Net count delta |
|---|---:|---:|---:|
| `iain-mcgilchrist--the-master-and-his-emissary` | 8 | 8 | 0 |
| `iain-mcgilchrist--the-matter-with-things-our-brains-our-delusions-and-the-unmaking-of-the-world` | 24 | 24 | 0 |
| `paul-kingsnorth--against-the-machine-on-the-unmaking-of-humanity` | 2 | 2 | 0 |

**Remedy:** none for this metric's reported population, because it reports no
count mismatches. It must not be treated as proof of identity consistency. The
hidden identity drift requires the two directional remedies in section 3.

## 2. `audit_library.works[*].orphaned_neo4j_chunks`

**Source:** `src/author_library/tools/meta.py:449-456`.

**Query (verbatim):**

```cypher
MATCH (c:Chunk)
WHERE NOT (c)-[:EXPLORES_THEME|MAKES_ARGUMENT|ATTRIBUTED_BY_CRITIC|CONCEPT_USED_IN|REFERENCES_PERSON]->()
RETURN c.work_id AS work_id, COUNT(c) AS orphan_count
```

**Population:** Neo4j `Chunk` nodes that have no outgoing relationship of any
of those five entity-extraction types. The audit groups that population by the
node's `work_id` and sums to 2,601.

**It does not measure:** whether the chunk ID exists in PostgreSQL, whether a
PostgreSQL chunk is missing from Neo4j, incoming relationships, or outgoing
relationships of other types. “Orphaned” in this field means uncovered by the
listed entity edges, not orphaned from PostgreSQL.

**Measured value:** 2,601. An identity join divides it into:

- 2,587 IDs that are present in PostgreSQL; for example,
  `00032051-591e-4205-b42d-8e5bd029fea5`.
- 14 IDs absent from PostgreSQL; these are the overlap with the 34 true Neo4j
  orphans listed in section 4.

**Remedy:** **none—this metric alone does not establish a defect.** A chunk may
legitimately yield no entity edge. If a separate extraction audit proves an
entire work was never extracted, `backfill_entities.py` is the relevant script,
but its current logic is work-level and skips a work as soon as that work has
any entity edge (`scripts/backfill_entities.py:38-55`). It is not a blanket
remedy for these 2,601 nodes.

## 3. `/parlour-booklist` identity set differences

The checked-out `parlour` `main` tree does not contain a tracked implementation
of this command. The installed command is
`~/.claude/skills/parlour-booklist/booklist.py`: PostgreSQL IDs at lines 147-150,
Neo4j IDs at lines 202-206, set differences at lines 514-531, and invocation at
lines 666-680.

**Queries (verbatim):**

```sql
SELECT id::text FROM chunks
```

```cypher
MATCH (c:Chunk) RETURN c.chunk_id AS cid
```

The implementation computes:

```python
missing_in_neo4j = pg_cids - neo4j_cids
missing_in_pg = neo4j_cids - pg_cids
```

**Population:** two directional chunk-ID set differences. The PG-only side is
current relational data with no graph node; the Neo4j-only side is graph data
with no current relational source row.

**It does not measure:** entity extraction coverage, chunk properties/text,
relationships, work-node equality, or duplicate IDs (both query results are
converted to sets).

**Measured value:** 34 IDs in each direction. Counts by work are 8, 24, and 2
for the three works shown in section 1. Representative PG-only IDs are
`216f4f4c-e9ff-48e2-a11a-0208d2cdccee`,
`0a830d9b-6870-4d1d-b96d-0767e9e878d1`, and
`a0b00409-18a2-435b-89c0-266d2d3f3df9`. Representative Neo4j-only IDs are
`009e77d96c5043968d52f74851cc470b`,
`08c085c01c7d44648d2dfc148570be2e`, and
`18a4b92ab0ea411d89ebe57fa5c9df1c`.

**Remedies:**

- PG-only 34: run `scripts/backfill_graph_and_entities.py` once for each of the
  three affected work IDs. Its graph step compares IDs and creates the missing
  current nodes (`src/author_library/graph/backfill.py:128-168`).
- Neo4j-only 34: `scripts/cleanup_neo4j_orphans.py` expresses the right
  PG-absence test (`scripts/cleanup_neo4j_orphans.py:46-48,89-103`). However,
  its current `AFFECTED_WORKS` list names three unrelated Coleridge/Holmes works
  (`scripts/cleanup_neo4j_orphans.py:24-28`), so the script must not be executed
  unchanged as a remedy for this snapshot. Stage 4 must scope it to the three
  affected works above, verify a dry-run selects exactly the expected IDs, and
  obtain Marty’s approval before any production deletion.

## 4. Population overlap, with chunk-ID evidence

The populations are **partially overlapping**, not equivalent:

- PG-only 34 vs Neo4j-only 34: intersection 0 by measured ID-set comparison.
- PG-only 34 vs entity-uncovered 2,601: intersection 0; a PG-only ID has no
  Neo4j node on which the edge predicate could match.
- Neo4j-only 34 vs entity-uncovered 2,601: intersection **14**. The other 20
  Neo4j-only nodes do have at least one of the five entity edges. For example,
  `012a0b5cbd8c4149b638966425047fbb` is Neo4j-only but is not in the 2,601.

The 14 IDs in both the Neo4j-only and entity-uncovered populations are:

```text
009e77d96c5043968d52f74851cc470b
08c085c01c7d44648d2dfc148570be2e
0a5bbb5745b74b2998d02ade966ddd97
1d3f16b9890f45d09b3a33f00d3b407a
2c921b02a40c49dfb2d7cf1426ca25d4
3f11339a0ede4c5592ea2c9ac55290c9
6255417921ff4448a565972553d3ea2b
9a1815cd4bbb48858115c5d7a48a4fdc
a0f8eaf7e87c45de968d05b12dd6cde0
a9924af40c2946c59b1bde869556b7ef
b3b08fcd78d04d568c00b09531e5eeb1
e63447d3329f4dc38f0c02dbf9522520
e8dec63b47cf4aea9b12df7b3338e5d6
e92b69bef3c54610b65b5b12c1d8c78c
```

## Safety verdict

Recommending `cleanup_neo4j_orphans.py` for the reported 2,601
`orphaned_neo4j_chunks` is **incorrect and potentially destructive**. That
number is not a PostgreSQL-absence population: 2,587 of its nodes have matching
PostgreSQL chunks. Deleting them would create 2,587 new PostgreSQL-to-Neo4j
gaps. Conversely, 20 of the 34 actual Neo4j orphans are absent from the 2,601
because they still have entity edges.

The cleanup script's actual chunk predicate—normalized Neo4j IDs minus
normalized PostgreSQL IDs—is the correct predicate for stale graph chunks. Its
name does not make the audit field's different predicate equivalent. Cleanup
must be recommended only from an identity set-difference measurement and only
after its exact deletion set has been reviewed and approved.
