import numpy as np
import kmeans
import common
import naive_em
import em

# Read a 2D dataset
# path = "30. Project 4. Collaborative Filtering/netflix/"
X = np.loadtxt("toy_data.txt")

# TODO: Your code here

seed = 1
for K in range(1,5):
    mixture, post = common.init(X, K, seed)
    # print(mixture)

# ======
toy_X = X

def test_seeds(X, K):
	print("\n############## KMEAN K=" + str(K) + " ###############")

	mixture0, post0 = common.init(X,K,0)
	mixture1, post1 = common.init(X,K,1)
	mixture2, post2 = common.init(X,K,2)
	mixture3, post3 = common.init(X,K,3)
	mixture4, post4 = common.init(X,K,4)

	cost0 = kmeans.run(X,mixture0,post0)[2]
	cost1 = kmeans.run(X,mixture1,post1)[2]
	cost2 = kmeans.run(X,mixture2,post2)[2]
	cost3 = kmeans.run(X,mixture3,post3)[2]
	cost4 = kmeans.run(X,mixture4,post4)[2]

	print("K=" + str(K) + " seed=0 : cost=" + str(cost0))
	print("K=" + str(K) + " seed=1 : cost=" + str(cost1))
	print("K=" + str(K) + " seed=2 : cost=" + str(cost2))
	print("K=" + str(K) + " seed=3 : cost=" + str(cost3))
	print("K=" + str(K) + " seed=4 : cost=" + str(cost4))

	naive_em_estimate0 = naive_em.run(X,mixture0,post0)
	naive_em_estimate1 = naive_em.run(X,mixture1,post1)
	naive_em_estimate2 = naive_em.run(X,mixture2,post2)
	naive_em_estimate3 = naive_em.run(X,mixture3,post3)
	naive_em_estimate4 = naive_em.run(X,mixture4,post4)

	print("K=" + str(K) + " seed=0 : likelihood=" + str(naive_em_estimate0[2]))
	print("K=" + str(K) + " seed=1 : likelihood=" + str(naive_em_estimate1[2]))
	print("K=" + str(K) + " seed=2 : likelihood=" + str(naive_em_estimate2[2]))
	print("K=" + str(K) + " seed=3 : likelihood=" + str(naive_em_estimate3[2]))
	print("K=" + str(K) + " seed=4 : likelihood=" + str(naive_em_estimate4[2]))


def test_em_seeds(X, K):
	print("\n############## EM K=" + str(K) + " ###############")

	mixture0, post0 = common.init(X,K,0)
	mixture1, post1 = common.init(X,K,1)
	mixture2, post2 = common.init(X,K,2)
	mixture3, post3 = common.init(X,K,3)
	mixture4, post4 = common.init(X,K,4)

	cost0 = em.run(X,mixture0,post0)[2]
	cost1 = em.run(X,mixture1,post1)[2]
	cost2 = em.run(X,mixture2,post2)[2]
	cost3 = em.run(X,mixture3,post3)[2]
	cost4 = em.run(X,mixture4,post4)[2]

	print("K=" + str(K) + " seed=0 : likelihood=" + str(cost0))
	print("K=" + str(K) + " seed=1 : likelihood=" + str(cost1))
	print("K=" + str(K) + " seed=2 : likelihood=" + str(cost2))
	print("K=" + str(K) + " seed=3 : likelihood=" + str(cost3))
	print("K=" + str(K) + " seed=4 : likelihood=" + str(cost4))


# K mean initialization

test_seeds(toy_X, 1)
test_seeds(toy_X, 2)
test_seeds(toy_X, 3)
test_seeds(toy_X, 4)

# EM algo
print("############## EM Algorythme implemented ###############")
mixture, post = common.init(toy_X,3,0)
naive_em_estimate = naive_em.run(toy_X,mixture,post)[2]
print("naive EM log likelihood : " + str(naive_em_estimate))

print("############## Some Tests ######################")
initialMixture, initialPost = common.init(toy_X,1,0)
mixtureEM1, postEM1, ll1 = naive_em.run(toy_X,initialMixture,initialPost)

initialMixture, initialPost = common.init(toy_X,2,0)
mixtureEM2, postEM2, ll2 = naive_em.run(toy_X,initialMixture,initialPost)

initialMixture, initialPost = common.init(toy_X,3,0)
mixtureEM3, postEM3, ll3 = naive_em.run(toy_X,initialMixture,initialPost)

initialMixture, initialPost = common.init(toy_X,4,0)
mixtureEM4, postEM4, ll4 = naive_em.run(toy_X,initialMixture,initialPost)

print("BIC K1 : " + str(common.bic(toy_X, mixtureEM1, ll1)))
print("BIC K2 : " + str(common.bic(toy_X, mixtureEM2, ll2)))
print("BIC K3 : " + str(common.bic(toy_X, mixtureEM3, ll3)))
print("BIC K4 : " + str(common.bic(toy_X, mixtureEM4, ll4)))

X_netflix = np.loadtxt("netflix_incomplete.txt")
test_em_seeds(X_netflix, 1)
test_em_seeds(X_netflix, 12)

X_gold = np.loadtxt('netflix_complete.txt')
mixture4, post4 = common.init(X_netflix,12,1)
mixture, post, cost4 = em.run(X_netflix,mixture4,post4)
X_pred = em.fill_matrix(X_netflix,mixture)

rmse_result = common.rmse(X_gold, X_pred)
print("RMSE between prediction and GOLD is : " + str(rmse_result))



# ==========

# K-means: determining the centroids by comparing the cost
k_dict = dict()
total_cost_dict = dict()
for seed in range(5):
    total_cost = 0
    for k in range(1, 5):
        mixture, post = common.init(X=X, K=k, seed=seed)
        cost = kmeans.run(X, mixture, post)[2]
        total_cost += cost
        k_dict.update({(seed, k): cost})
    total_cost_dict.update({seed: total_cost})

### get the best seed and the best k size that minimizes the cost

## Best seed
# Get the lowest cost
optimal_seed_cost = total_cost_dict[0]
for k, v in total_cost_dict.items():
    if v < optimal_seed_cost:
        optimal_seed_cost = v
    else:
        optimal_seed_cost = optimal_seed_cost
# Get the seed associated with the lowest cost
for k, v in total_cost_dict.items():
    if v == optimal_seed_cost:
        optimal_seed = k

## Best k size
# Get the lowest cost
optimal_k_cost = k_dict[(optimal_seed, 1)]
# Create a new dictionary for k size
optimal_k_dict = dict()
for i in range(1, 5):
    optimal_k_dict.update({(optimal_seed, i): k_dict[(optimal_seed, i)]})
for k, v in optimal_k_dict.items():
    if v < optimal_k_cost:
        optimal_k_cost = v
    else:
        optimal_k_cost = optimal_k_cost
# Get the seed associated with the lowest cost
for k, v in optimal_k_dict.items():
    if v == optimal_k_cost:
        optimal_k = k[1]

### Plotting the k clusters
optimal_seed_k = list()
optimal_seed_k_post = list()
title_list = list()
for i in range(1, 5):
    initial_mixture, initial_post = common.init(X, i, seed = optimal_seed)
    mixture, post, cost = kmeans.run(X, initial_mixture, initial_post)
    optimal_seed_k.append(mixture)
    optimal_seed_k_post.append(post)
    title_list.append(("K-means: The mixture plot when k = {}".format(i)))

for i in range(4):
    common.plot(X, optimal_seed_k[i], optimal_seed_k_post[i], title_list[i])

####### Compare k-means with EM

# K-means: determining the centroids by comparing the cost
em_k_dict = dict()
em_total_likelihood_dict = dict()
for seed in range(5):
    em_total_likelihood = 0
    for k in range(1, 5):
        mixture, post = common.init(X=X, K=k, seed=seed)
        likelihood = naive_em.run(X, mixture, post)[2]
        em_total_likelihood += likelihood
        em_k_dict.update({(seed, k): likelihood})
    em_total_likelihood_dict.update({seed: em_total_likelihood})

### get the best seed and the best k size that minimizes the cost

## Best seed
# Get the lowest cost
optimal_seed_cost = em_total_likelihood_dict[0]
for k, v in em_total_likelihood_dict.items():
    if v > optimal_seed_cost:
        optimal_seed_cost = v
    else:
        optimal_seed_cost = optimal_seed_cost
# Get the seed associated with the lowest cost
for k, v in em_total_likelihood_dict.items():
    if v == optimal_seed_cost:
        optimal_seed = k
print(em_k_dict)

### Plotting the k clusters
em_optimal_seed_k = list()
em_optimal_seed_k_post = list()
em_title_list = list()
for i in range(1, 5):
    initial_mixture, initial_post = common.init(X, i, seed = optimal_seed)
    mixture, post, likelihood = naive_em.run(X, initial_mixture, initial_post)
    em_optimal_seed_k.append(mixture)
    em_optimal_seed_k_post.append(post)
    em_title_list.append(("Gaussian Mixture: The mixture plot when k = {}".format(i)))

for i in range(4):
    common.plot(X, em_optimal_seed_k[i], em_optimal_seed_k_post[i], title_list[i])

### Seed with best BIC
BIC_k_dict = dict()
BIC_total_likelihood_dict = dict()
for seed in range(5):
    BIC_total_likehood = 0
    for k in range(1, 5):
        mixture, post = common.init(X=X, K=k, seed=seed)
        log_likelihood = naive_em.run(X, mixture, post)[2]
        BIC = common.bic(X, mixture, log_likelihood)
        BIC_total_likehood += BIC
        BIC_k_dict.update({(seed, k): BIC})
    total_cost_dict.update({seed: BIC_total_likehood})
print(BIC_k_dict)


### Determining the initialization



### run EM algorithm on X