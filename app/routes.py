from flask import Blueprint, jsonify, render_template

from agent.researcher import research

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/research/<symbol>")
def api_research(symbol):
    symbol = symbol.strip().upper()
    if not symbol:
        return jsonify({"error": "A ticker symbol is required."}), 400
    try:
        report = research(symbol)
    except Exception as exc:  # surface failures to the frontend
        return jsonify({"error": str(exc)}), 500
    return jsonify({"symbol": symbol, "report": report})
