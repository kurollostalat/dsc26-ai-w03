# AI usage

## Tools I used, and for what

I used ChatGPT to understand the task requirements, plan the implementation, write and review parts of the code, debug errors, and interpret model evaluation results. I implemented and ran the code, shared test results, and worked through corrections with ChatGPT.

## AI-suggested code I rejected or fixed in add_history_features or time_split

I reviewed and refined `add_history_features` to ensure that parcels belonging to the same customer with identical timestamps do not count as prior parcels for one another. I also ensured that the original row order and index are preserved. In `time_split`, I checked that all parcels sharing the boundary timestamp are assigned to the test set. I verified the implementation using the public tests, which passed.

## How I checked that my model has no leak

I excluded `courier_notes`, `delivered_at`, `failed`, and `parcel_id` from the model features because they are unavailable or inappropriate at prediction time, and used a chronological train/test split with history features based only on strictly earlier timestamps.
