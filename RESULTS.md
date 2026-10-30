# Results: Tawseel failed-delivery baseline

## Baseline table

Evaluation on the last 20% of parcels by time, using `make_parcels(4000, seed=0)`.

| Model                          |    ROC AUC |     PR-AUC |      Brier |  Base rate |
| ------------------------------ | ---------: | ---------: | ---------: | ---------: |
| Dummy (prior)                  |     0.5000 |     0.2200 |     0.1717 |     0.2200 |
| Rule                           | Not tested | Not tested | Not tested | Not tested |
| My model (Logistic Regression) |     0.6492 |     0.3837 |     0.1630 |     0.2200 |

**PR-AUC lift over Dummy:** +0.1637 (approximately 74.4% relative improvement).

## Three sentences for Sherif

The model can estimate the probability of a COD parcel failing and ranks parcels by their predicted failure risk, achieving a PR-AUC of 0.3837 compared with 0.2200 for the prior-only baseline. These results show a promising improvement on a time-based test split of 800 synthetic parcels, but they do not yet establish performance on real Tawseel operations. The model cannot guarantee that an individual parcel will fail, explain the cause of a failure, or replace operational judgment; it needs validation on real historical delivery data before being used for decisions.
