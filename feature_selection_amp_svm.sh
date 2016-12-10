#!/bin/bash

#!/bin/bash

python3 feature_selection_amp_svm.py EASY 0 > log/AMP_EASY_0.log 2> log/AMP_EASY_0.err
python3 feature_selection_amp_svm.py EASY 1000 > log/AMP_EASY_1000.log 2> log/AMP_EASY_1000.err
python3 feature_selection_amp_svm.py EASY 1500 > log/AMP_EASY_1500.log 2> log/AMP_EASY_1500.err
python3 feature_selection_amp_svm.py EASY 2000 > log/AMP_EASY_2000.log 2> log/AMP_EASY_2000.err
python3 feature_selection_amp_svm.py MODERATE 0 > log/AMP_MODERATE_0.log 2> log/AMP_MODERATE_0.err
python3 feature_selection_amp_svm.py MODERATE 500 > log/AMP_MODERATE_500.log 2> log/AMP_MODERATE_500.err
python3 feature_selection_amp_svm.py MODERATE 1000 > log/AMP_MODERATE_1000.log 2> log/AMP_MODERATE_1000.err
python3 feature_selection_amp_svm.py MODERATE 1500 > log/AMP_MODERATE_1500.log 2> log/AMP_MODERATE_1500.err
python3 feature_selection_amp_svm.py MODERATE 2000 > log/AMP_MODERATE_2000.log 2> log/AMP_MODERATE_2000.err
python3 feature_selection_amp_svm_difficult.py 500 > log/AMP_DIFFICULT_500.log 2> log/AMP_DIFFICULT_500.err
python3 feature_selection_amp_svm_difficult.py 1000 > log/AMP_DIFFICULT_1000.log 2> log/AMP_DIFFICULT_1000.err
python3 feature_selection_amp_svm_difficult.py 1500 > log/AMP_DIFFICULT_1500.log 2> log/AMP_DIFFICULT_1500.err
python3 feature_selection_amp_svm_difficult.py 2000 > log/AMP_DIFFICULT_2000.log 2> log/AMP_DIFFICULT_2000.err