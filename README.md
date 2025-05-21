# Prediction of Responsive Individuals to an Alcohol Intervention

## Code Execution
- Run `main_analysis.ipynb` for the main analysis, which generates the `results` folder
    - storing all models
    - all CV test/train data
    - all CV test/train results
    - run test on out-of-sample follow-up test data
    - create SHAP and PDP plots (in `results/img`)
- Run `visualization.ipynb` to generate the main figure with significance tests
- Run `negative_control.ipynb` to run the negative control checks where participants from the control group are assigned to either control ("off") or active ("on") weeks to check if the variation predicted in `main_analysis` is due to the intervention or to natural fluctuation.
