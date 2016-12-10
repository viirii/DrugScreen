#!/bin/bash

python3 dhm.py EASY 0 > log/EASY_0.log 2> log/EASY_0.err
python3 dhm.py EASY 500 > log/EASY_500.log 2> log/EASY_500.err
python3 dhm.py EASY 1000 > log/EASY_1000.log 2> log/EASY_1000.err
python3 dhm.py EASY 1500 > log/EASY_1500.log 2> log/EASY_1500.err
python3 dhm.py EASY 2000 > log/EASY_2000.log 2> log/EASY_2000.err
python3 dhm.py MODERATE 0 > log/MODERATE_0.log 2> log/MODERATE_0.err
python3 dhm.py MODERATE 500 > log/MODERATE_500.log 2> log/MODERATE_500.err
python3 dhm.py MODERATE 1000 > log/MODERATE_1000.log 2> log/MODERATE_1000.err
python3 dhm.py MODERATE 1500 > log/MODERATE_1500.log 2> log/MODERATE_1500.err
python3 dhm.py MODERATE 2000 > log/MODERATE_2000.log 2> log/MODERATE_2000.err
python3 dhm_difficult.py 0 > log/DIFFICULT_0.log 2> log/DIFFICULT_0.err
python3 dhm_difficult.py 500 > log/DIFFICULT_500.log 2> log/DIFFICULT_500.err
python3 dhm_difficult.py 1000 > log/DIFFICULT_1000.log 2> log/DIFFICULT_1000.err
python3 dhm_difficult.py 1500 > log/DIFFICULT_1500.log 2> log/DIFFICULT_1500.err
python3 dhm_difficult.py 2000 > log/DIFFICULT_2000.log 2> log/DIFFICULT_2000.err