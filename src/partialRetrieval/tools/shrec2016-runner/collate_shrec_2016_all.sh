#! /bin/bash

g++ metrics/metrics.cpp -o metrics/metrics -O3

echo "Artificial Q25"
python3 collate_results_shrec2016_style.py /mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/Artificial/Q25 \
/mnt/NEXUS/datasets/SHREC2016-Partial-Shape-Queries/SHREC_material/Artificial_labels \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Artificial/Q25
echo ""
metrics/metrics metrics/Hampson_query_cla_6.txt \
metrics/Hampson_target_cla_6.txt \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Artificial/Q25 \
21 \
metrics/metrics_Artificial_Q25.txt
echo ""

echo "Artificial Q40"
python3 collate_results_shrec2016_style.py /mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/Artificial/Q40 \
/mnt/NEXUS/datasets/SHREC2016-Partial-Shape-Queries/SHREC_material/Artificial_labels \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Artificial/Q40
echo ""
metrics/metrics metrics/Hampson_query_cla_6.txt \
metrics/Hampson_target_cla_6.txt \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Artificial/Q40 \
21 \
metrics/metrics_Artificial_Q40.txt
echo ""

echo "Breuckmann View1"
python3 collate_results_shrec2016_style.py /mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/Breuckmann/View1 \
/mnt/NEXUS/datasets/SHREC2016-Partial-Shape-Queries/SHREC_material/Breuckmann_labels \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Breuckmann/View1
echo ""
metrics/metrics metrics/Breuckmann_query_cla_6.txt \
metrics/Hampson_target_cla_6.txt \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Breuckmann/View1 \
25 \
metrics/metrics_Breuckmann_View1.txt
echo ""

echo "Breuckmann View2"
python3 collate_results_shrec2016_style.py /mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/Breuckmann/View2 \
/mnt/NEXUS/datasets/SHREC2016-Partial-Shape-Queries/SHREC_material/Breuckmann_labels \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Breuckmann/View2
echo ""
metrics/metrics metrics/Breuckmann_query_cla_6.txt \
metrics/Hampson_target_cla_6.txt \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Breuckmann/View2 \
25 \
metrics/metrics_Breuckmann_View2.txt
echo ""

echo "Breuckmann View3"
python3 collate_results_shrec2016_style.py /mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/Breuckmann/View3 \
/mnt/NEXUS/datasets/SHREC2016-Partial-Shape-Queries/SHREC_material/Breuckmann_labels \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Breuckmann/View3
echo ""
metrics/metrics metrics/Breuckmann_query_cla_6.txt \
metrics/Hampson_target_cla_6.txt \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Breuckmann/View3 \
25 \
metrics/metrics_Breuckmann_View3.txt
echo ""

echo "Kinect View1"
python3 collate_results_shrec2016_style.py /mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/Kinect/View1 \
/mnt/NEXUS/datasets/SHREC2016-Partial-Shape-Queries/SHREC_material/Kinect_labels \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Kinect/View1
echo ""
metrics/metrics metrics/Kinect_query_cla_6.txt \
metrics/Hampson_target_cla_6.txt \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Kinect/View1 \
25 \
metrics/metrics_Kinect_View1.txt
echo ""

echo "Kinect View2"
python3 collate_results_shrec2016_style.py /mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/Kinect/View2 \
/mnt/NEXUS/datasets/SHREC2016-Partial-Shape-Queries/SHREC_material/Kinect_labels \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Kinect/View2
echo ""
metrics/metrics metrics/Kinect_query_cla_6.txt \
metrics/Hampson_target_cla_6.txt \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Kinect/View2 \
25 \
metrics/metrics_Kinect_View2.txt
echo ""

echo "Kinect View3"
python3 collate_results_shrec2016_style.py /mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/Kinect/View3 \
/mnt/NEXUS/datasets/SHREC2016-Partial-Shape-Queries/SHREC_material/Kinect_labels \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Kinect/View3
echo ""
metrics/metrics metrics/Kinect_query_cla_6.txt \
metrics/Hampson_target_cla_6.txt \
/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/shrec2016_rankedLists/Kinect/View3 \
25 \
metrics/metrics_Kinect_View3.txt
echo ""