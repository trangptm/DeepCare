# DeepCare
DeepCare is a deep dynamic model that reads EMR data, infer disease progression and predict future outcome.
4 tasks are implemented:
  - Disease progression: predict diagnoses of the next readmission
  - Intervention recommendation: predict procedures/medications for a set of diagnoses
  - Readmission prediction: predict if a patient will re-admit within a period
  - High-risk prediction: predict if a patient is in high risk (3 unplanned readmissions within a period)
  
DeepCare uses a LSTM to model the patient's history. It treats each patient as a sequence of admissions. Unlike a typical LSTM model, DeepCare reads input from multiple sources: diagnoses, interventions (procedures/medications), admission time and admission type (unplanned or planned).

Link to the paper:
https://arxiv.org/abs/1602.00357
