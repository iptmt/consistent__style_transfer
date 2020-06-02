import csv
import numpy as np


def read_csv(file, reverse=False, reserve_index=False):
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        sys_tuples = ([], [], [])
        line_count = 0
        for row in csv_reader:
            sti, cp, nt = int(row["STI"]), int(row["CP"]), int(row["NT"])
            if reverse:
                sti, cp, nt = 4 - sti, 4 - cp, 4 - nt
            sys_id = line_count % 3
            if reserve_index:
                sys_tuples[sys_id].append([sti, cp, nt]) 
            else:
                sys_tuples[sys_id].append([1/sti, 1/cp, 1/nt]) 
            line_count += 1
    return np.array(sys_tuples)


def fleissKappa(rate,n):
    """ 
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category 
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters   
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)

    #mean of the extent to which raters agree for the ith subject 
    PA = sum([(sum([i**2 for i in row])- n) / (n * (n - 1)) for row in rate])/N
    print("PA = ", PA)
    
    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j**2 for j in [sum([rows[i] for rows in rate])/(N*n) for i in range(k)]])
    print("PE =", PE)
    
    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)
    
    return kappa

def create_kappa_mat(seq0, seq1, seq2):
    full_mat = []
    for v0, v1, v2 in zip(seq0, seq1, seq2):
        single_mat = [0, 0, 0]
        single_mat[int(v0) - 1] += 1
        single_mat[int(v1) - 1] += 1
        single_mat[int(v2) - 1] += 1
        full_mat.append(single_mat)
    return full_mat


base = "result/"
reserve_idx = False# True for flesiss' kappa / False for calculating scores
# yelp
y_res_0 = read_csv(base + "yelp_0.csv", True, reserve_idx)
y_res_1 = read_csv(base + "yelp_1.csv", False, reserve_idx)
y_res_2 = read_csv(base + "yelp_2.csv", False, reserve_idx)
if not reserve_idx:
    print((y_res_0.mean(1) + y_res_1.mean(1) + y_res_2.mean(1))/3)
# amazon
a_res_0 = read_csv(base + "amazon_0.csv", True, reserve_idx)
a_res_1 = read_csv(base + "amazon_1.csv", False, reserve_idx)
a_res_2 = read_csv(base + "amazon_2.csv", False, reserve_idx)
if not reserve_idx:
    print((a_res_0.mean(1) + a_res_1.mean(1) + a_res_2.mean(1))/3)

if reserve_idx:
    kappa_mat = create_kappa_mat(
        np.concatenate((y_res_0, a_res_0), axis=1).flatten(),
        np.concatenate((y_res_1, a_res_1), axis=1).flatten(),
        np.concatenate((y_res_2, a_res_2), axis=1).flatten()
    )
    kappa_v = fleissKappa(kappa_mat, n=3)