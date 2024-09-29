from flask import Flask, render_template, request, jsonify
import numpy as np
from kmeans import KMeans

app = Flask(__name__)

# Route to display the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to generate new random data
@app.route('/generate_data', methods=['POST'])
def generate_data():
    np.random.seed()  # Remove seed to ensure randomness
    num_points = 300
    data = np.random.randn(num_points, 2) * 5  # Scaled up to make centroids more spread out
    return jsonify(data.tolist())

# Route to run KMeans step-by-step
@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    data = np.array(request.json['data'])
    n_clusters = int(request.json['n_clusters'])
    init_method = request.json['init_method']
    manual_centroids = request.json.get('manual_centroids')
    
    kmeans = KMeans(n_clusters=n_clusters, init_method=init_method)
    centroids, labels = kmeans.fit(data, manual_centroids=manual_centroids)
    
    return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=3000)
