import pandas as pd
import os
import numpy as np
import scipy.stats as sc


PHYSICAL_DIST_THR = 50     # Physical distance threshold
WHOLE_GENOME = 3088286401


# def calc_distance(site1, site2):
#     chr1 = site1.iloc[1]
#     chr2 = site2.iloc[1]
#     if chr1 != chr2:
#         return WHOLE_GENOME
#     loc1 = site1.iloc[2]
#     loc2 = site2.iloc[2]
#     dist = np.abs(int(loc1)-int(loc2))
#     return dist


def get_close_sites():
    file = "sites_locations.csv"
    locations_df = pd.read_csv(file, header=0)
    sorted_df = locations_df.sort_values(by=["chrCHR", "MAPINFO"])
    size = sorted_df.shape[0]
    close_sites = pd.DataFrame(columns=["chr", "site1", "pos1", "site2", "pos2", "dist"])
    count = -1
    meta_data_chr = ""  # TODO - remove
    for i in range(size):
        curr_chr = sorted_df.iloc[i, 1]
        if curr_chr != meta_data_chr:  # TODO - remove
            print(curr_chr)
            meta_data_chr = curr_chr
        curr_loc = sorted_df.iloc[i, 2]
        j = i+1
        next_site_chr = sorted_df.iloc[j, 1]
        next_site_loc = sorted_df.iloc[j, 2]
        dist = next_site_loc - curr_loc
        while dist < 100 and curr_chr == next_site_chr:
            count += 1
            close_sites.loc[count] = [curr_chr, sorted_df.iloc[i, 0], curr_loc, sorted_df.iloc[j, 0], next_site_loc, dist]
            j += 1
            next_site_chr = sorted_df.iloc[j, 1]
            next_site_loc = sorted_df.iloc[j, 2]
            dist = next_site_loc - curr_loc
    np.savetxt("close_sites.csv", close_sites, delimiter=",")

#
# def get_corr_matrix(tissue_list):
#     for tissue in tissue_list:
#         file = f"{tissue}.horvath.csv"
#         tissue_df = pd.read_csv(file, header=0)
#         tissue_df = tissue_df.sort()
#         size = tissue_df.shape[0]
#         corr_matrix = np.zeros((size, size))
#         for i in range(size):
#             for j in range(size):
#                 site_a = tissue_df.iloc[i, :]
#                 site_b = tissue_df.iloc[j, :]
#                 corr = sc.pearsonr(site_a, site_b)
#                 corr_matrix[i, j], corr_matrix[j, i] = corr, corr
#         np.savetxt(f"{tissue}_corr_matrix.csv", corr_matrix, delimiter=",")


if __name__ == '__main__':
    # tissue_list = ["TRY"]
    tissue_list = ["BRCA", "HNSC", "LUSC", "KIRC", "PRAD", "COAD", "KIRP", "LIHC", "THCA", "LUAD", "UCEC"]
    # path = os.path.join("path")
    loc_matrix = get_close_sites()
    # get_corr_matrix(tissue_list)
    

