# Crossmatching and Deduplication Architecture

## Purpose

This document describes and justifies the execution design used in the CRC
pipeline for crossmatching, graph construction, deduplication, and final catalog
generation.

The central design choice is deliberate:

- LSDB/HATS is used where spatial semantics are required;
- Dask DataFrame is used for distributed relational operations;
- Parquet is used as a distributed materialization boundary;
- pandas is used only inside bounded, per-partition computations or small-data
  fast paths.

This architecture separates spatial and relational responsibilities. It keeps
the HATS spatial model at the core of the pipeline while using simpler
distributed tabular representations for stages where spatial semantics are not
required.

## Executive summary

For this pipeline and workload profile, the large-catalog path uses
`LSDB Catalog.concat()` and `write_catalog()` selectively, and stages the
heavier non-spatial transformations through Dask and Parquet boundaries.

In our production-scale runs, that path kept the full LSDB/NestedFrame lineage
attached through crossmatching, neighbor aggregation, catalog updates,
concatenation, and writing. In this workload, that was associated with higher
scheduler and driver pressure and made serialization and worker recovery more
expensive.

The large-catalog path therefore uses the following strategy:

1. Open and spatially partition catalogs with LSDB/HATS.
2. Project each catalog to the columns required by the spatial crossmatch.
3. Execute the crossmatch with LSDB, preserving HATS pixels and margins.
4. Materialize only the resulting pair identifiers as narrow Parquet datasets.
5. Use Dask to aggregate neighbors and join them back to the complete rows.
6. Concatenate the complete updated rows with Dask.
7. Write a distributed Parquet checkpoint.
8. Rebuild the HATS catalog and margins with `hats-import`.
9. Use LSDB/HATS pixel alignment again for margin-aware deduplication.

The scientific semantics are unchanged. The design only changes how intermediate
data is represented, scheduled, and materialized.

## Responsibilities by technology

| Technology | Responsibility |
| --- | --- |
| LSDB/HATS | Catalog opening, spatial partitioning, margins, crossmatching, and main/margin pixel alignment |
| Dask DataFrame | Distributed projection, filtering, grouping, joins by `CRD_ID`, tabular concatenation, and lazy output filtering |
| Parquet | Distributed checkpoints that cut prior task-graph lineage and provide a stable input to HATS import |
| `hats-import` | Reconstruction of large HATS collections, spatial metadata, pixel layout, and margins from staged rows |
| pandas | Local graph solving inside one aligned main+margin partition and explicitly bounded small-data fast paths |

## The three data representations used during crossmatching

### 1. Complete HATS catalog

The complete catalog contains all scientific and provenance columns, for
example:

```text
CRD_ID, ra, dec, z, z_err, source, survey,
z_flag_homogenized, instrument_type_homogenized,
compared_to, tie_result, ...
```

It is represented as an LSDB catalog backed by a distributed
Dask/NestedFrame collection. It remains the authoritative source for complete
rows throughout the pipeline.

The complete catalog is used to:

- preserve every output column;
- receive the updated `compared_to` values;
- supply tie-breaking attributes during deduplication;
- produce the final consolidated catalog.

### 2. Spatially projected HATS catalog

The angular crossmatch does not require redshift, flags, instrument type, or
other scientific columns. Before calling LSDB crossmatch, the pipeline projects
each input catalog to:

```text
CRD_ID, ra, dec
```

`source` is additionally retained on the left side when neighbor-saturation
diagnostics are enabled.

This projection is still an LSDB/HATS catalog. Its HATS partitioning and
projected margin are preserved. Consequently, LSDB still controls the spatial
search and boundary handling.

Projecting before the crossmatch is important. It keeps the spatial stage narrow
from the start, so the underlying graph does not need to reference and
serialize wide input NestedFrames.

### 3. Narrow pair table

The useful output of the spatial crossmatch is an edge list:

```text
CRD_IDleft, CRD_IDright
```

Optional diagnostic columns are:

```text
_dist_arcsec, sourceleft
```

Inside each worker, the selected NestedFrame partition is converted to a plain
`pandas.DataFrame`. The partitions are then written as a distributed Parquet
dataset and reopened as a Dask DataFrame.

This step has two purposes:

1. Only the required columns cross the materialization boundary.
2. Downstream graph operations no longer retain the LSDB crossmatch graph.

The pair table is used to remove self-matches and duplicate pairs, monitor
neighbor saturation, aggregate adjacency information, and update
`compared_to`.

## End-to-end crossmatch flow

```text
Complete HATS catalog A                     Complete HATS catalog B
          |                                           |
          | project CRD_ID, ra, dec, [source]         | project CRD_ID, ra, dec
          v                                           v
Projected HATS catalog A  ---- LSDB crossmatch ----  Projected HATS catalog B
                                  |
                                  v
                 Narrow distributed pair table
                    CRD_IDleft, CRD_IDright
                                  |
                       Dask groupby / aggregation
                                  |
                                  v
                  CRD_ID -> new compared_to values
                         /                    \
                        v                      v
          Dask join into complete A    Dask join into complete B
                        \                      /
                         v                    v
                     Dask concat of complete rows
                                  |
                                  v
                    Distributed Parquet checkpoint
                                  |
                                  v
                  hats-import rebuilds HATS + margins
```

The complete catalogs are used immediately afterward to restore the complete
scientific rows and attach the graph edges identified by the spatial
operation.

## Why the neighbor update uses Dask joins

After crossmatching, the pipeline constructs a table similar to:

```text
CRD_ID     _new_compared_to
A          B, C
B          A
C          A
```

This is a relational join by `CRD_ID`, rather than a spatial join. The neighbor table
has no coordinates, HATS pixels, margins, or spatial metadata. Converting it
into an artificial HATS catalog would add work without adding spatial
correctness.

At this stage, the natural operation is a distributed Dask merge:

```text
complete catalog rows LEFT JOIN neighbor table ON CRD_ID
```

The merge updates `compared_to` while preserving all other columns from the
complete catalog.

## Why the final crossmatch concatenation uses Dask

After neighbor values have been joined back, the pipeline has two complete
distributed dataframes. At this stage the required operation is row-wise
tabular concatenation. There is no new spatial relationship to calculate.

Using `LSDB Catalog.concat()` here would require wrapping the updated
dataframes back into catalog objects while retaining the lineage of:

- the LSDB crossmatch;
- pair projection and filtering;
- distributed neighbor aggregation;
- joins into both complete catalogs;
- catalog concatenation;
- final catalog writing.

Calling `write_catalog()` on that result would submit or retain a large,
coupled graph through the writing stage. In our production-scale runs, this
path was associated with high scheduler/driver pressure, large serialization
costs, and avoidable worker instability.

Dask concatenation followed by Parquet staging is intentionally simpler:

```text
Dask joins -> Dask concat -> distributed Parquet
```

The output remains complete; only the execution representation changes.

## Why Parquet is an architectural boundary

Parquet serves as a checkpoint between two distributed phases.

Before the checkpoint, the graph describes how rows were produced from prior
crossmatches, aggregations, and joins. After reopening the Parquet dataset, the
graph describes only how to read the materialized partitions.

This boundary provides:

- shorter Dask graph lineage;
- lower scheduler bookkeeping and serialization pressure;
- release of references to earlier NestedFrame graphs;
- independent retry and inspection of staged data;
- a stable, distributed input for `hats-import`;
- explicit schema normalization before rebuilding HATS.

## Why `hats-import` is used after staging

The Dask concat produces complete tabular rows, but it does not, by itself, create a
HATS collection. `hats-import` is then responsible for reconstructing:

- spatial pixel organization;
- HATS catalog metadata;
- catalog properties;
- the configured margin catalog.

This staging step produces an intermediate boundary, after which the next
persistent processing artifact is again a valid HATS collection.

## Deduplication architecture

Crossmatching records graph edges in `compared_to`. Deduplication then solves
the connected components and assigns `tie_result` and canonical `group_id`
values.

This stage returns to LSDB/HATS because partition boundaries and margins are
spatially meaningful.

### HATS main/margin alignment

The pipeline uses LSDB/HATS pixel-tree alignment through
`concat_align_catalogs`, `get_aligned_pixels_from_alignment`, and
`align_and_apply`.

This aligns each main partition with the corresponding margin support,
including pixels represented only in the margin. It avoids assuming that the
raw Dask divisions of the main and margin dataframes are identical.

### Local bounded solver

For each aligned pixel, the worker receives:

```text
main partition + aligned margin partition
```

Only columns required by the graph solver are retained, including:

- `CRD_ID`;
- `compared_to`;
- redshift;
- configured tie-breaking priorities;
- homogenized object type and graph-inclusion policy;
- coordinates required by optional geometry diagnostics.

The local partitions are converted to pandas because the graph algorithm is a
bounded per-pixel computation. This does not materialize the full catalog on
the driver. Many independent partition solvers execute across Dask workers.

The solver labels main and margin rows together but emits labels only for main
rows. Margin rows supply boundary context and are not duplicated in the final
catalog.

### Merge of labels into complete rows

The per-partition outputs contain only the identifiers and labels required by
the complete catalog, principally:

```text
CRD_ID, tie_result, group_id
```

These labels are merged back into the complete distributed dataframe. The
complete dataframe remains lazy for HATS, Parquet, and CSV output paths.

The default diagnostics count rows missing a newly computed label and classify
invalid tie groups by aggregate reason. Optional detailed diagnostics collect a
bounded sample of offending `group_id` and member `CRD_ID` values; this mode is
disabled by default because it requires an additional distributed scan.

`tie_result` has the following stable meanings:

- `0`: participating row that lost within its graph component;
- `1`: selected winner;
- `2`: unresolved hard-tie candidate;
- `3`: row deliberately excluded from the graph by object-type configuration.

Rows with `tie_result=3` remain visible as isolated records in marked outputs.
`concatenate_and_remove_duplicates` filters them out because that mode retains
only `tie_result=1` (plus hard ties according to `tie_treatment_option`). They
neither create nor receive graph edges.

## Scientific semantics preserved by this design

These representation changes do not alter the intended scientific decisions:

- angular edges are still generated by LSDB crossmatch at the configured
  radius;
- HATS margins still provide cross-partition spatial context;
- graph components are still formed from `compared_to` edges;
- object types disabled by configuration remain isolated from the graph;
- configured priority columns still determine winners and hard ties;
- redshift disambiguation and missing-redshift policy remain unchanged;
- canonical groups and tie-result invariants are applied after the same edge
  construction.

The pair table contains identifiers rather than scientific values because its
role is only to represent graph edges. Scientific values remain in the complete
catalog and are read by the deduplication solver.

## Fast paths intentionally retained

The pipeline retains `concat()` and `write_catalog()` where they are a good
fit.

They remain appropriate when the graph is small and simple:

- when a crossmatch finds no new pairs, the pipeline can use LSDB concat and
  `write_catalog()` directly because no neighbor aggregation or join graph was
  added;
- small in-memory HATS outputs can use LSDB `from_dataframe()` and
  `write_catalog()`;
- large and lazy HATS outputs use Parquet staging and `hats-import`.

The choice is based on execution scale and graph complexity. It should be read
as a workload-specific execution tradeoff.

## Why HATS is not used for every intermediate table

HATS provides value when a dataset needs spatial indexing or spatial boundary
semantics. Some intermediate datasets do not:

- a `CRD_IDleft`/`CRD_IDright` edge list;
- a `CRD_ID`/neighbor aggregation table;
- per-object tie labels;
- a tabular concatenation waiting to be reimported.

Representing these structures as HATS datasets would require artificial coordinates,
unnecessary indexing, extra metadata, and additional import/write cycles. It
would not improve the correctness of a key-based groupby or join.

The design principle is therefore:

> Use HATS whenever spatial organization affects correctness or performance;
> use distributed tabular structures when the operation is purely relational.

## Operational trade-offs

The checkpoint-based path has costs:

- additional temporary disk I/O;
- temporary Parquet storage requirements;
- an explicit HATS reimport step;
- cleanup and retry logic for intermediate artifacts.

These costs are accepted because, in this workload, they replace less
predictable scheduler and worker memory pressure with bounded distributed I/O.
At production scale, this tradeoff favors bounded execution, recoverability,
and operational clarity.

## Future opportunities for a more direct LSDB path

The current design remains compatible with future LSDB optimizations. If LSDB gains
a large-catalog concat/write path that materializes partitions incrementally,
cuts lineage before writing, or accepts narrow key-based updates without
retaining the complete preceding graph, this pipeline design can be
reevaluated and potentially simplified.

Any comparison should verify:

- peak scheduler memory;
- peak driver and worker memory;
- serialized graph size;
- retry behavior after worker loss;
- total runtime and temporary storage;
- equivalence of output rows, HATS metadata, margins, and graph labels.

## Final design rule

The pipeline uses the narrowest representation that preserves the semantics of
each stage:

```text
Spatial search and boundary alignment -> LSDB/HATS
Key-based graph operations             -> Dask DataFrame
Bounded per-pixel graph solving        -> pandas inside workers
Distributed execution checkpoint       -> Parquet
Large spatial catalog reconstruction   -> hats-import
```

This separation keeps HATS at the spatial core of the pipeline while allowing
large non-spatial transformations to execute with simpler, more observable,
and more recoverable distributed graphs.
