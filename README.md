# Predicting individual differences in digital alcohol intervention effectiveness through multimodal data

This repository contains the analysis code for the publication:

Fuchs, M., Boyd, Z.M., Schwarze, A. *et al*. Predicting individual differences in digital alcohol intervention effectiveness through multimodal data. *npj Digit. Med*. (2026). https://doi.org/10.1038/s41746-026-02356-4


## Code 
- `main_analysis.ipynb` contains the main analysis, which generates the `results` folder
    - producing sample characteristics
    - storing all models
    - all CV test/train data
    - all CV test/train results
    - run test on out-of-sample follow-up test data
    - create SHAP and PDP plots (in `results/img`)
- `visualization.ipynb` generates the main figure with significance tests
- `negative_control.ipynb` runs the negative control checks where participants from the control group are assigned to either control ("off") or active ("on") weeks to check if the variation predicted in `main_analysis` is due to the intervention or to natural fluctuation.
- `added_checks.ipynb` includes further sensitivity analyses, mostly added in response to reviewer comments.
- `model_free_evidence.ipynb` includes standard statistical tests on the baseline variables for responders and non-responders to the intervention.


## Public Data
### Main Analysis
For the main prediction, 6 data domains collected at baseline were used:
- `data/baseline/alcoholself_bucket280225.csv`
- `data/baseline/subjective_grouperceptions_280225.csv`
- `data/baseline/data_social.csv`
- `data/baseline/brain_bucket_280225.csv`
- `data/baseline/demographic_bucket280225.csv`
- `data/baseline/psychometrics_bucket280225.csv`

The intervention outcome was computed from the logged drinking behavior:
- `data/intervention_time/osf_study1.csv`
- `data/intervention_time/osf_study2.csv`

### Added Analyses
Several other data files (collected during the SHINE study but not used for the main analysis) were used for sensitivity and robustness checks:
- `data/added_analysis/baseline_alc_self.csv` contains self-reported drinking frequency and amount for to establish test-retest stability of drinking perceptions between baseline and followup
- `data/intervention_time/EMA_study1.csv` & `data/intervention_time/EMA_study2.csv` are granular data from the participants' self-logging during the intervention period and are used in `src/added_checks.ipynb` to determine the settings (with friends, alone, etc.) that participants drank in.
- `data/added_analysis/social_group_drinking.csv` reports the average peer group drink frequency to evaluate whether the actual group behavior rather than peer perceptions influence an individual's response to the intervention.
- `data/added_analysis/followup1_peers_and_self_drinking.csv` contains participants' self-reported drinking data at followup, to establish test-retest stability of drinking perceptions
- `data/added_analysis/peer_perceptions_vs_peer_selfreports.csv` was used to visualize peer perceptions vs. self-reported drinking
- `data/added_analysis/underestimators_study_1.csv` contains the IDs of participants who underestimated their peer's drinking to check for systematic differences between this group and the overestimators

## Private Data
Due to risk of participant re-identification, social network data could not be published but is available from the authors upon reasonable request. File paths for these data sources have been left in the scripts so as to show where and how they were used. Below is a description of this data and the included features:

- `data/baseline/data_social.csv` includes features: `id` (participant ID),`like_deg_in` (in-degree based on how often this person was nominated by their peers as being "liked"), `alcLeast_deg_in`, `alcMost_deg_in`, `closest_deg_in`, `influence_deg_in`, `leaders_deg_in`, `goToBad_deg_in`, `goToGood_deg_in` - more detailed descriptions of these features can be found in the manuscript, the supplemental materials, or the SHINE study protocol (linked below).

- `data/added_analysis/alcmost_nominations_baseline_followup1.csv` was constructed from peer nominations at baseline and peer nominations at the 6-month followup to assess whether and how participants' nominations of their highest drinking peers had changed over time. It includes `pID` (participant ID), `cleaned_baseline` (list of peer IDs nominated at baseline, e.g., "{'muri00', 'muri01', 'muric02'}"), `cleaned_followup1` (list of peer IDs nominated at followup), `jaccard_similarity` (calculated similartiy between the baseline and followup nominees)


## Links
- Publication: https://doi.org/10.1038/s41746-026-02356-4 
- Supplements: https://static-content.springer.com/esm/art%3A10.1038%2Fs41746-026-02356-4/MediaObjects/41746_2026_2356_MOESM1_ESM.pdf
- SHINE protocol: https://osf.io/preprints/psyarxiv/cj2nx_v1

## Contact
For questions or clarifications you can contact the author at: magdalena.fuchs@mtec.ethz.ch 