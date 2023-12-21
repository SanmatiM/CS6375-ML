import numpy as np
import random
import math
import pandas as pd
import re
import string
import csv
from io import StringIO
import requests


"""Read file from url"""
def read_file(file_path):
    response = requests.get(file_path)
    content = response.text
    return StringIO(content)

def preprocess_tweets(tweets):
    """Preprocesses a list of tweets."""
    for index in range(len(tweets)):
        tweets[index] = tweets[index].lower()
        tweets[index] = " ".join(filter(lambda x: x[0] != '@', tweets[index].split()))
        tweets[index] = re.sub(r"http\S+", "", tweets[index])
        tweets[index] = re.sub(r"www\S+", "", tweets[index])
        tweets[index] = tweets[index].strip()
        length = len(tweets[index])
        if length > 0 and tweets[index][length - 1] == ':':
            tweets[index] = tweets[index][:length - 1]
        tweets[index] = tweets[index].replace('#', '')
        tweets[index] = tweets[index].translate(str.maketrans('', '', string.punctuation))
        tweets[index] = " ".join(tweets[index].split())
    return tweets

def initialize_centroids(tweets, k):
    """Initializes k centroids randomly."""
    centroids = []
    init_centroid_map = dict()
    j = 0

    if k >= len(tweets):
        raise ValueError("Number of centroids (k) should be less than the number of tweets.")

    while j < k:
        centroid_index = random.randint(0, len(tweets) - 1)
        if centroid_index not in init_centroid_map:
            centroids.append(tweets[centroid_index])
            init_centroid_map[centroid_index] = True
            j += 1

    return centroids

def calculate_jaccard_distance(tweet1, tweet2):
    """Calculates Jaccard distance between two tweets."""
    intersection = set(tweet1).intersection(tweet2)
    union = set().union(tweet1, tweet2)
    return 1 - (len(intersection) / len(union))

def update_centroids(clusters):
    """Updates centroids based on the mean of each cluster."""
    new_centroids = []
    for cluster_index in range(len(clusters)):
        min_sum_distance = math.inf
        centroid_index = -1
        for point_index in range(len(clusters[cluster_index])):
            current_distance = sum(clusters[cluster_index][point_index][1:])
            if current_distance < min_sum_distance:
                min_sum_distance = current_distance
                centroid_index = point_index
        new_centroids.append(clusters[cluster_index][centroid_index][0])
    return new_centroids

def k_means(tweets, k, max_iterations=50):
    """Performs k-means clustering."""
    centroids = initialize_centroids(tweets, k)
    iteration_count = 0
    previous_centroids = []

    while (previous_centroids != centroids) and (iteration_count < max_iterations):
        clusters = dict()

        for tweet_index in range(len(tweets)):
            min_distance = math.inf
            cluster_index = -1

            for centroid_index in range(len(centroids)):
                distance = calculate_jaccard_distance(centroids[centroid_index], tweets[tweet_index])
                if distance < min_distance:
                    cluster_index = centroid_index
                    min_distance = distance

            clusters.setdefault(cluster_index, []).append([tweets[tweet_index], min_distance])

        previous_centroids = centroids
        centroids = update_centroids(clusters)
        iteration_count += 1

    return clusters, calculate_sse(clusters)

def calculate_sse(clusters):
    """Calculates Sum of Squared Errors (SSE)."""
    sse = 0
    for cluster_index in range(len(clusters)):
        for point_index in range(len(clusters[cluster_index])):
            sse += clusters[cluster_index][point_index][1] ** 2
    return sse

# Read file and preprocess tweets
file_path = "https://raw.githubusercontent.com/SanmatiM/CS6375-Machine-Learning/main/Assignment3/health%2Bnews%2Bin%2Btwitter/Health-Tweets/usnewshealth.txt"
dataset = read_file(file_path)
df = pd.read_csv(dataset, sep='|', header=None, names=['ID', 'Timestamp', 'Tweet'])
df.drop(['ID', 'Timestamp'], axis=1, inplace=True)
df.Tweet = df.Tweet.astype(str)
df.Tweet = preprocess_tweets(df.Tweet)

# Perform k-means for different values of k
k_values = list(range(3, 13))
table = [["Value of K", "SSE", "Size of each cluster"]]

for k in k_values:
    print(f"Running k-means for k = {k}")
    clusters, sse = k_means(df.Tweet, k)
    cluster_info = [f"Cluster {i + 1}: {len(clusters[i])} tweets" for i in range(len(clusters))]
    table.append([k, sse, cluster_info])

# Save results to a CSV file
csv_file_path = "k_means_results.csv"
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(table)
