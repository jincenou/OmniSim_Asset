#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python scripts/run_variant.py --case random10_02_white_pickup --variant R1 --seed 42 --gpu 2 > logs/random10_02_white_pickup_R1_seed42_extra.log 2>&1
echo $? > logs/random10_02_white_pickup_R1_seed42_extra.exit
python scripts/run_variant.py --case random10_02_white_pickup --variant R2 --seed 42 --gpu 2 > logs/random10_02_white_pickup_R2_seed42_extra.log 2>&1
echo $? > logs/random10_02_white_pickup_R2_seed42_extra.exit
python scripts/run_variant.py --case random10_02_white_pickup --variant R --seed 7 --gpu 2 > logs/random10_02_white_pickup_R_seed7_extra.log 2>&1
echo $? > logs/random10_02_white_pickup_R_seed7_extra.exit
python scripts/run_variant.py --case random10_02_white_pickup --variant B0_real1 --seed 7 --gpu 2 > logs/random10_02_white_pickup_B0_real1_seed7_extra.log 2>&1
echo $? > logs/random10_02_white_pickup_B0_real1_seed7_extra.exit
python scripts/run_variant.py --case random10_02_white_pickup --variant B0_real4 --seed 7 --gpu 2 > logs/random10_02_white_pickup_B0_real4_seed7_extra.log 2>&1
echo $? > logs/random10_02_white_pickup_B0_real4_seed7_extra.exit
python scripts/run_variant.py --case random10_02_white_pickup --variant B0_repeat --seed 42 --gpu 2 > logs/random10_02_white_pickup_B0_repeat_seed42_extra.log 2>&1
echo $? > logs/random10_02_white_pickup_B0_repeat_seed42_extra.exit
