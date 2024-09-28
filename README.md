# Dialectic-Preference-Bias-LLM
Code repository for the dialectic preference bias paper


## TwitterAAE release v1, 2016-09-01
http://slanglab.cs.umass.edu/TwitterAAE/

If you use this corpus, please cite the following paper which describes it:

S. Blodgett, L. Green, and B. O'Connor. Demographic Dialectal Variation in Social Media: A Case Study of African-American English. Proceedings of EMNLP. Austin. 2016.

************************

The corpus contains four tab-delimited files; every item except the demographic inferences is a JSON formatted value.

- emnlp_release_limited: This file contains 59.2 million tweet IDs, tweet timestamps, and model demographic inferences (AA, Hispanic, Other, and White, in order).

- emnlp_release_limited_aa: This file contains 1.1 million tweet IDs, tweet timestamps, and model demographic inferences (AA, Hispanic, Other, and White, in order) from users whose posterior probability of using AA-associated terms under the model was greater than 0.8.

- emnlp_release_all: This file contains the same tweets as emnlp_release_limited, and includes tweet IDs, tweet timestamps, user IDs, tweet longitudes and latitudes, tweet Census blockgroups, full tweet texts, and model demographic inferences.

- emnlp_release_all_aa: This file contains the same tweets as emnlp_release_limited_aa, and includes tweet IDs, tweet timestamps, user IDs, tweet longitudes and latitudes, tweet Census blockgroups, full tweet texts, and model demographic inferences.

Warning: a small number of messages have posterior probabilities of nan.  These are messages that have no in-vocabulary words under the model.
