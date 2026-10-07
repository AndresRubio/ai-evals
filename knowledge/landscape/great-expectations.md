---
name: great-expectations
description: Data-quality contracts for pipelines, with expectation suites, checkpoints that stop a run, and Data Docs as the record
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://greatexpectations.io/
    read: 2026-10-07
license: Apache-2.0
related: [landscape-overview, data-contracts-at-ingest]
---
# Great Expectations

## What it is

GX Core, an open-source Python framework for asserting what data should look like and
validating it where a pipeline runs, plus GX Cloud, a hosted layer over it. Its home
page calls it the most popular data quality framework in the world, with a community
of over 11,000 practitioners.

## Core abstraction

An Expectation is one testable assertion about data. An Expectation Suite groups them.
Validation runs a suite against a batch of data. A Checkpoint is the orchestration
point that runs validations and acts on the result. Data Docs are generated,
human-readable pages of what was checked and what passed.

## Where it sits

Data quality for tabular pipelines, not LLM evaluation. Checkpoints run before a
pipeline step to reject bad input and after it to verify the output, stop the run when
a threshold is missed, and notify. Integrations named on the home page: Airflow,
Databricks, BigQuery, Snowflake, PostgreSQL, S3, Azure Blob Storage, Jupyter.

GX Core is local-first and needs no account; GX Cloud is the hosted option.

## Maturity

As of 2026-10-07: GX Core is Apache-2.0 and actively released; GX Cloud is a commercial
product.

## Biggest complaint

Not applicable; not surveyed for practitioner complaints, since it is outside the LLM
eval family.

## Relation to axial

The same loop at the data layer. An expectation suite is the contract, a checkpoint
is the gate, Data Docs are the record. In axial the contract is the labelled split, the
gate is `axial gate`, and `alignment.md` is the record. What axial could take: an
expectation-style check at ingest for empty outputs, duplicate trace ids and timestamp
ranges, so a bad export fails before it reaches a review batch. See
[data-contracts-at-ingest](../procedures/data-contracts-at-ingest.md). No dependency.

## Corrections

None.
