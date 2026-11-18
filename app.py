"""
RoadGuard AI - Main Flask Application Entry Point
Intelligent Road Sign Occlusion Detection System for Autonomous Driving Research.
Serves both REST API endpoints and futuristic AI Dashboard web views.
"""

import os
import sys

# Ensure project root in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, send_from_directory, render_template, jsonify
from backend.routes import api_bp, get_model

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)

# Configuration
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "roadguard-ai-secret-key-2026")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Register API Blueprint
app.register_blueprint(api_bp)


# ------------------ Static & UI Routes ------------------ #

@app.route("/")
def index():
    """Serves the main RoadGuard AI Futuristic Dashboard."""
    return send_from_directory("frontend", "index.html")


@app.route("/dashboard")
def dashboard_page():
    """Direct route for Dashboard Module."""
    return send_from_directory("frontend", "dashboard.html")


@app.route("/scanner")
def scanner_page():
    """Direct route for AI Road Sign Scanner Module."""
    return send_from_directory("frontend", "scanner.html")


@app.route("/occlusion-lab")
def occlusion_lab_page():
    """Direct route for Occlusion Lab Simulator Module."""
    return send_from_directory("frontend", "occlusion_lab.html")


@app.route("/comparison")
def comparison_page():
    """Direct route for Before vs After Analysis Module."""
    return send_from_directory("frontend", "comparison.html")


@app.route("/performance")
def performance_page():
    """Direct route for Model Performance Metrics Module."""
    return send_from_directory("frontend", "performance.html")


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serves generated plots, matrices, and reports."""
    return send_from_directory("results", filename)


@app.route("/dataset-sample/<int:class_id>")
def sample_image(class_id):
    """Provides a sample road sign image for one-click testing in the scanner or lab."""
    import glob
    pattern = os.path.join("dataset", "Images", f"{class_id:05d}", "*.png")
    files = glob.glob(pattern)
    if not files:
        pattern2 = os.path.join("dataset", "*", f"{class_id}", "*.png")
        files = glob.glob(pattern2)
    if files:
        return send_from_directory(os.path.dirname(files[0]), os.path.basename(files[0]))
    return jsonify({"error": "No sample found for class"}), 404


# ------------------ Error Handlers ------------------ #

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource Not Found", "status": 404}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal Server Error", "status": 500, "message": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ROADGUARD AI - AUTONOMOUS DRIVING RESEARCH DASHBOARD")
    print("  Server launching on http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Preload model in background if ready
    try:
        get_model()
    except Exception as ex:
        print(f"[Warning] Model preload deferred: {ex}")
        
    app.run(host="0.0.0.0", port=5000, debug=False)
