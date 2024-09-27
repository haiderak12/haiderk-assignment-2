from flask import Flask, render_template, request, jsonify
import numpy as np
from kmeans import KMeans

app = Flask(__name__)

# Route to display the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to generate new data
@app.route('/generate_data', methods=['POST'])
def generate_data():
    np.random.seed(42)
    num_points = 300
    x = np.random.randn(num_points, 2)
    return jsonify(x.tolist())

# Route to run KMeans step-by-step
@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    data = np.array(request.json['data'])
    kmeans = KMeans(n_clusters=int(request.json['n_clusters']), init_method=request.json['init_method'])
    centroids, labels = kmeans.fit(data)
    return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=3000)
