# To do profiling:
# python -m cProfile -o profile_output.txt particle_filter_icp.py
# And then run this script to print results
import pstats
p = pstats.Stats("profile_output.txt")
p.strip_dirs().sort_stats('cumulative').print_stats(50)