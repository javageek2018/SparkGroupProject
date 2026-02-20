# SDSC Expanse Spark Setup

## Environment Overview

This project was executed on SDSC Expanse using an interactive compute
node. The dataset contains hundreds of millions of NYC FHV trip records,
so a distributed Spark setup was required to handle large shuffle
operations such as deduplication, aggregations, and feature engineering.

## Cluster Resources Requested

For large-scale preprocessing (deduplication + aggregations), the
following resources were requested:

-   Total Cores: 32
-   Total Memory: 128 GB

These resources were selected to ensure sufficient memory for wide
shuffle stages during deduplication and tipping aggregations.


## Executor Configuration

Spark was configured using the following formula:

Executor Instances = Total Cores - 1\
Executor Memory = (Total Memory - Driver Memory) / Executor Instances

Applied configuration:

-   Total Cores = 32
-   Total Memory = 128 GB
-   Driver Memory = 8 GB

Executor Instances = 32 - 1 = 31

Executor Memory = (128 - 8) / 31\
Executor Memory ≈ 3.87 GB per executor

This configuration ensures: - One core reserved for the driver -
Parallel execution across executor slots - Balanced memory distribution
for shuffle-heavy operations


## SparkSession Configuration

``` python
spark = (
    SparkSession.builder
    .appName("fhvhv-dedup")
    .master("local[*]")
    .config("spark.driver.memory", "90g")
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.sql.shuffle.partitions", "4000")
    .config("spark.sql.files.maxPartitionBytes", "128m")
    .config("spark.local.dir", os.environ["TMPDIR"])
    .getOrCreate()
)
```

### Justification

-   spark.driver.memory = 90g
    Prevents driver-side memory pressure while leaving sufficient memory
    for executors.

-   spark.sql.shuffle.partitions = 4000
    Increased from default to reduce shuffle spill during deduplication
    of hundreds of millions of rows.

-   spark.sql.files.maxPartitionBytes = 128m
    Improves parallelism when loading large parquet files.

-   spark.local.dir = TMPDIR
    Ensures shuffle spill occurs on high-speed scratch disk instead of
    quota-limited home directory.

## Scratch Disk Usage

Shuffle spill and temporary data were directed to:

/scratch/`<username>`{=html}/job\_`<job_id>`{=html}

This prevents disk quota errors and allows large-scale sort and shuffle
operations.


## Spark UI Verification

During large transformations such as:

-   dropDuplicates()
-   groupBy aggregations
-   Parquet write operations

Multiple executors were active concurrently.


## Why This Setup Was Necessary

The dataset contains several hundred million records.

Operations such as: - Deduplication on composite keys - Group-by tipping
aggregations - Temporal feature engineering

require wide transformations and shuffle stages.

The distributed executor configuration allowed: - Stable shuffle
performance - Reduced disk spill failures - Successful persistence of a
cleaned Parquet dataset
