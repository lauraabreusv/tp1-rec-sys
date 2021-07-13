import pandas as pd
import sys
import numpy as np
import time

def sort_tups(a, b):
    if a > b:
        return(a, b)
    return(b,a)

def pearson(item1, item2, item_matrix, item_sqrd):
    if not item_sqrd[item1] or not item_sqrd[item2]:
        return 0
    
    common = 0
    for user in item_matrix[item1].keys():
        try:
            rating2 = item_matrix[item2][user]
            rating1 = item_matrix[item1][user]
            common += (rating1*rating2)
            
        except:
            continue
            
    return round(common/(np.sqrt(item_sqrd[item1])*np.sqrt(item_sqrd[item2])), 4) 

def create_utility_matrix(df, item_means):
    user_matrix = {}
    item_matrix = {}
    item_sqrd = {}
    for row in df.itertuples():
        item = user_matrix.get(row.UserId, {})
        user = item_matrix.get(row.ItemId, {})
        sqrd = item_sqrd.get(row.ItemId, 0)
        
        item[row.ItemId] = row.Prediction - item_means[row.ItemId]
        user[row.UserId] = row.Prediction - item_means[row.ItemId]
        sqrd += round(user[row.UserId]*user[row.UserId],4)
        
        user_matrix[row.UserId] = item
        item_matrix[row.ItemId] = user
        item_sqrd[row.ItemId] = round(sqrd, 4)
        
    return user_matrix, item_matrix, item_sqrd

def run(ratings, targets):
    train_df = pd.read_csv(ratings, sep='[:,]')
    test_df = pd.read_csv(targets, sep=':')

    abs_mean = round(train_df['Prediction'].mean(), 4)
    
    user_means = {c:round(v,4) for c,v in train_df.groupby('UserId')['Prediction'].mean().items()}
    item_means = {c:round(v,4) for c,v in train_df.groupby('ItemId')['Prediction'].mean().items()}
    user_counts = {c: v for c,v in train_df.groupby('UserId')['Prediction'].count().items()}
    item_counts = {c: v for c,v in train_df.groupby('ItemId')['Prediction'].count().items()}

    user_matrix, item_matrix, item_sqrd = create_utility_matrix(train_df, item_means)

    results = []
    pre_calc = {}
    count = 0
    count_s = 0
    count_n = 0
    for row in test_df.itertuples():
        try:
            item_means[row.ItemId]
        except: 
            try:
                results.append((user_means[row.UserId]*user_counts[row.UserId]+abs_mean)/(user_counts[row.UserId]+1))
            except:
                results.append(abs_mean)
            continue
            
        # pred = 0
        # s = 0
        sims = []
        try:
            for item in user_matrix[row.UserId].keys():            
                tup = sort_tups(item, row.ItemId)

                if tup not in pre_calc:
                    sim = pearson(item, row.ItemId, item_matrix, item_sqrd)
                    pre_calc[tup] = sim
                else:
                    sim = pre_calc[tup]

                if sim != 0:
                    sims.append((sim, item))
                    

            #top-200
            if len(sims) > 50:
                sims.sort(reverse=True)
                pred = sum(sims[i][0]*user_matrix[row.UserId][sims[i][1]] for i in range(50))
                s = sum(abs(x[0]) for x in sims[:50])
                results.append((pred/s) + item_means[row.ItemId])
                count_n +=1
                # pred += sim*user_matrix[row.UserId][item]
                # s += abs(sim)
            # if s:
            elif len(sims) > 0:
                pred = sum(sims[i][0]*user_matrix[row.UserId][sims[i][1]] for i in range(len(sims)))
                s = sum(abs(x[0]) for x in sims)
                results.append((pred/s) + item_means[row.ItemId])
                count_s +=1
            else:
                results.append((item_means[row.ItemId]*item_counts[row.ItemId]+abs_mean)/(item_counts[row.ItemId]+1))
                count+=1
        except:
            results.append((item_means[row.ItemId]*item_counts[row.ItemId]+abs_mean)/(item_counts[row.ItemId]+1))

    results = [10 if x > 10 else 0 if x < 0 else x for x in results]

    sub_df = pd.read_csv(targets)
    sub_df['Prediction'] = results

    print(sub_df.to_csv(index=False))

    print(count)
    print(count_s)
    print(count_n)

if __name__ == "__main__":
    start = time.time()
    run(sys.argv[1], sys.argv[2])
    stop = time.time()
    print(stop - start)