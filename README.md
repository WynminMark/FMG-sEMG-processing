# FMG-sEMG-processing
[toc]
## for signal processing and muscle strength estimation


### iFEMG_feature.py
- Class SignalFeature()
- Class FMGFeature(SignalFeature)
- Class sEMGFeature(SignalFeature)
- Class AntagonisticFMGFeature()
  - based on Class FMGFeature
- Class AntagonisticsEMGFeature()
  - based on Class sEMGFeature()
- Class LabeledSignalFeature()
- Class LabeledFMGFeature(LabeledSignalFeature)
- Class LabeledsEMGFeature(LabeledSignalFeature)
- Function fea_df_norm(features_df, *col_name)
