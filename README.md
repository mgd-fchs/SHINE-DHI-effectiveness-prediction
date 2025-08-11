# Prediction of Responsive Individuals to an Alcohol Intervention
This notebook holds the code and data for used to obtain the results presented in "Peer Perceptions as Key Predictors in Multimodal Models of Digital Alcohol Intervention Effectiveness" submitted to npj Digital Medicine.

Digital interventions can change behaviors like alcohol use, but predicting who will benefit remains difficult. We present a novel approach integrating multimodal data, across theory-driven domains—including psychological assessments, social network data, and neural responses to alcohol cues—to make ex-ante predictions about the effectiveness of smartphone-delivered alcohol interventions, targeting psychological distancing, in young adults.

Please note that the social network data (`b3_group_sociometric`) cannot be publicly provided due to privacy concerns. All other data used to provide our analyses is included in this repository.

## Code Execution
- Run `src/main_analysis.ipynb` for the main analysis, which generates the `results` folder
    - storing all models
    - all CV test/train data
    - all CV test/train results
    - run test on out-of-sample follow-up test data
    - create SHAP and PDP plots (in `results/img`)
- Run `visualization.ipynb` to generate the main results figure
- Run `negative_control.ipynb` to run the negative control checks where participants from the control group are assigned to either control ("off") or active ("on") weeks to check if the variation predicted in `main_analysis` is due to the intervention or to natural fluctuation.

