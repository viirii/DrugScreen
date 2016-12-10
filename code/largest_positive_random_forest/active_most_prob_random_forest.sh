#!/bin/bash

python3 active_most_proba_random_forest.py EASY 0 > log/AMRF_EASY_0.log 2> log/AMRF_EASY_0.err
python3 active_most_proba_random_forest.py EASY 1000 > log/AMRF_EASY_1000.log 2> log/AMRF_EASY_1000.err
python3 active_most_proba_random_forest.py EASY 1500 > log/AMRF_EASY_1500.log 2> log/AMRF_EASY_1500.err
python3 active_most_proba_random_forest.py EASY 2000 > log/AMRF_EASY_2000.log 2> log/AMRF_EASY_2000.err
python3 active_most_proba_random_forest.py MODERATE 0 > log/AMRF_MODERATE_0.log 2> log/AMRF_MODERATE_0.err
python3 active_most_proba_random_forest.py MODERATE 500 > log/AMRF_MODERATE_500.log 2> log/AMRF_MODERATE_500.err
python3 active_most_proba_random_forest.py MODERATE 1000 > log/AMRF_MODERATE_1000.log 2> log/AMRF_MODERATE_1000.err
python3 active_most_proba_random_forest.py MODERATE 1500 > log/AMRF_MODERATE_1500.log 2> log/AMRF_MODERATE_1500.err
python3 active_most_proba_random_forest.py MODERATE 2000 > log/AMRF_MODERATE_2000.log 2> log/AMRF_MODERATE_2000.err
python3 active_most_proba_random_forest_difficult.py 0 > log/AMRF_DIFFICULT_0.log 2> log/AMRF_DIFFICULT_0.err
python3 active_most_proba_random_forest_difficult.py 500 > log/AMRF_DIFFICULT_500.log 2> log/AMRF_DIFFICULT_500.err
python3 active_most_proba_random_forest_difficult.py 1000 > log/AMRF_DIFFICULT_1000.log 2> log/AMRF_DIFFICULT_1000.err
python3 active_most_proba_random_forest_difficult.py 1500 > log/AMRF_DIFFICULT_1500.log 2> log/AMRF_DIFFICULT_1500.err
python3 active_most_proba_random_forest_difficult.py 2000 > log/AMRF_DIFFICULT_2000.log 2> log/AMRF_DIFFICULT_2000.err