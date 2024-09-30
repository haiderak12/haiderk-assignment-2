# app.py

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import numpy as np
import random

app = Flask(__name__)
app.secret_key = 'cs506'  # Replace with a secure key in production

# Configure server-side session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

class KMeans:
    def __init__(self, data, k, init_method='random'):
        self.data = data
        self.k = k
        self.init_method = init_method
        self.centroids = []
        self.assignments = [0] * len(data)
        self.previous_assignments = None
        self.iteration = 0

        if self.init_method != 'manual':
            self.initialize_centroids()

    def initialize_centroids(self):
        if self.init_method == 'random':
            self.centroids = random.sample(self.data.tolist(), self.k)
        elif self.init_method == 'farthest':
            self.centroids = self.farthest_first_initialization()
        elif self.init_method == 'kmeans++':
            self.centroids = self.kmeans_plus_plus_initialization()
        else:
            raise ValueError("Unknown initialization method")

    def farthest_first_initialization(self):
        centroids = []
        centroids.append(random.choice(self.data.tolist()))
        while len(centroids) < self.k:
            distances = []
            for point in self.data.tolist():
                min_distance = min([np.linalg.norm(np.array(point) - np.array(c)) for c in centroids])
                distances.append((point, min_distance))
            next_centroid = max(distances, key=lambda x: x[1])[0]
            centroids.append(next_centroid)
        return centroids

    def kmeans_plus_plus_initialization(self):
        centroids = []
        centroids.append(random.choice(self.data.tolist()))
        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(np.array(x) - np.array(c)) ** 2 for c in centroids]) for x in self.data.tolist()])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = random.random()
            for idx, c_prob in enumerate(cumulative_probabilities):
                if r < c_prob:
                    centroids.append(self.data.tolist()[idx])
                    break
        return centroids

    def assign_clusters(self):
        self.previous_assignments = self.assignments.copy()
        for idx, point in enumerate(self.data.tolist()):
            distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in self.centroids]
            self.assignments[idx] = int(np.argmin(distances))

    def update_centroids(self):
        for i in range(self.k):
            cluster_points = self.data[np.array(self.assignments) == i]
            if len(cluster_points) > 0:
                self.centroids[i] = cluster_points.mean(axis=0).tolist()

    def has_converged(self):
        return self.previous_assignments == self.assignments

    def step(self):
        self.assign_clusters()
        self.update_centroids()
        self.iteration += 1
        return {
            'centroids': self.centroids,
            'assignments': self.assignments,
            'iteration': self.iteration
        }

    def run(self):
        while True:
            self.assign_clusters()
            self.update_centroids()
            self.iteration += 1
            if self.has_converged():
                break
        return {
            'centroids': self.centroids,
            'assignments': self.assignments,
            'iteration': self.iteration
        }

def kmeans_to_dict(kmeans):
    return {
        'data': kmeans.data.tolist(),
        'k': kmeans.k,
        'init_method': kmeans.init_method,
        'centroids': kmeans.centroids,
        'assignments': kmeans.assignments,
        'previous_assignments': kmeans.previous_assignments,
        'iteration': kmeans.iteration
    }

def dict_to_kmeans(d):
    kmeans = KMeans(np.array(d['data']), d['k'], d['init_method'])
    kmeans.centroids = d['centroids']
    kmeans.assignments = d['assignments']
    kmeans.previous_assignments = d['previous_assignments']
    kmeans.iteration = d['iteration']
    return kmeans

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    n_samples = 200
    centers = 4
    cluster_std = 1.0

    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)
    session['data'] = data.tolist()
    return jsonify({'data': session['data']})

@app.route('/initialize', methods=['POST'])
def initialize():
    k = int(request.form['k'])
    init_method = request.form['init_method']
    data = np.array(session.get('data'))

    kmeans = KMeans(data, k, init_method)
    session['kmeans'] = kmeans_to_dict(kmeans)
    return jsonify({
        'centroids': kmeans.centroids,
        'assignments': kmeans.assignments
    })

@app.route('/set_centroids', methods=['POST'])
def set_centroids():
    kmeans_dict = session.get('kmeans')
    if not kmeans_dict:
        return jsonify({'error': 'KMeans not initialized'}), 400

    centroids = request.json.get('centroids')
    if not centroids:
        return jsonify({'error': 'No centroids provided'}), 400

    kmeans = dict_to_kmeans(kmeans_dict)
    kmeans.centroids = centroids
    session['kmeans'] = kmeans_to_dict(kmeans)
    return jsonify({'status': 'centroids updated'})

@app.route('/step', methods=['POST'])
def step():
    kmeans_dict = session.get('kmeans')
    if not kmeans_dict:
        return jsonify({'error': 'KMeans not initialized'}), 400

    kmeans = dict_to_kmeans(kmeans_dict)
    result = kmeans.step()
    session['kmeans'] = kmeans_to_dict(kmeans)
    return jsonify(result)

@app.route('/run', methods=['POST'])
def run():
    kmeans_dict = session.get('kmeans')
    if not kmeans_dict:
        return jsonify({'error': 'KMeans not initialized'}), 400

    kmeans = dict_to_kmeans(kmeans_dict)
    result = kmeans.run()
    session['kmeans'] = kmeans_to_dict(kmeans)
    return jsonify(result)

@app.route('/reset', methods=['POST'])
def reset():
    session.pop('kmeans', None)
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(port=3000)
