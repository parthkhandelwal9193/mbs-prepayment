# MBS Prepayment Model
Predict monthly prepayment (SMM/CPR) on agency mortgages using Freddie Mac loan-level data,
then reprice an MBS pool under rate shocks to get WAL, effective duration and convexity.

## Structure
- data/raw: downloaded Freddie Mac sample files and PMMS csv (not committed)
- data/processed: parquet files built by our code
- notebooks: exploration and learning
- src: reusable functions (loading, features, model, cash flows)
- outputs: figures and saved models

## Phases
0 setup | 1 load data | 2 survival panel | 3 features | 4 models | 5 validation | 6 S-curve/CPR | 7 cash flows | 8 rate shocks
