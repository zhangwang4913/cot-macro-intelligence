"""
COT Macro Intelligence — Flask web server.
Serves the HTML dashboard and exposes REST API endpoints.
"""

import asyncio
import logging
import os
import sys
import webbrowser
from pathlib import Path
from threading import Timer

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

import analytics
import database
from data_pipeline import COTPipeline, TRACKED_MARKETS

load_dotenv()
log = logging.getLogger("cot.server")

BASE_DIR = Path(__file__).parent
DB_PATH  = Path(os.environ.get("COT_DB_PATH",  str(BASE_DIR / "cot_data.db")))
DATA_DIR = Path(os.environ.get("COT_DATA_DIR", str(BASE_DIR / "data")))

app = Flask(__name__, static_folder=str(BASE_DIR))


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _serialize_cat(cat) -> dict:
    return {
        "category":       cat.category,
        "long":           cat.long_contracts,
        "short":          cat.short_contracts,
        "net":            cat.net_contracts,
        "oi":             cat.open_interest,
        "net_pct_oi":     cat.net_pct_oi,
        "wow":            cat.wow_change,
        "pct_1y":         cat.pct_1y,
        "pct_3y":         cat.pct_3y,
        "pct_5y":         cat.pct_5y,
        "z_1y":           cat.z_1y,
        "z_3y":           cat.z_3y,
        "z_5y":           cat.z_5y,
    }


def _serialize_metrics(all_metrics: dict) -> list:
    out = []
    for code, m in all_metrics.items():
        pcat = analytics.PRIMARY_CATEGORY.get(m.report_type, "managed_money")
        cat  = m.categories.get(pcat)
        row = {
            "code":            code,
            "display":         m.display_name,
            "asset_class":     m.asset_class,
            "report_type":     m.report_type,
            "latest_date":     str(m.latest_date) if m.latest_date else None,
            "primary_cat":     pcat,
            "alignment_score": m.alignment_score,
            "alignment_label": m.alignment_label,
            "categories":      {k: _serialize_cat(v) for k, v in m.categories.items()},
        }
        if cat:
            row.update({
                "pct_1y":    cat.pct_1y,
                "pct_3y":    cat.pct_3y,
                "pct_5y":    cat.pct_5y,
                "z_1y":      cat.z_1y,
                "z_3y":      cat.z_3y,
                "z_5y":      cat.z_5y,
                "wow":       cat.wow_change,
                "net_pct_oi":cat.net_pct_oi,
                "net":       cat.net_contracts,
            })
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "dashboard.html")


@app.route("/api/metrics")
def get_metrics():
    try:
        all_metrics = analytics.compute_all_metrics(DB_PATH)
        db_stats    = database.get_db_stats(DB_PATH)
        return jsonify({
            "markets":   _serialize_metrics(all_metrics),
            "db_stats":  {k: str(v) for k, v in db_stats.items()},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/<code>/<category>")
def get_history(code, category):
    try:
        df = database.get_history(DB_PATH, code, category, weeks=104)
        df = df.sort_values("report_date", ascending=True)
        return jsonify({
            "dates":          df["report_date"].astype(str).tolist(),
            "net_contracts":  df["net_contracts"].fillna(0).astype(int).tolist(),
            "net_pct_oi":     df["net_pct_oi"].fillna(0).round(3).tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/insight/<code>", methods=["POST"])
def get_insight(code):
    try:
        from insight_engine import InsightEngine
        all_metrics = analytics.compute_all_metrics(DB_PATH)
        m = all_metrics.get(code)
        if not m:
            return jsonify({"error": f"No data for market {code}"}), 404
        engine = InsightEngine(DB_PATH)
        force  = request.json.get("force", False) if request.is_json else False
        text   = _run_async(engine.get_market_insight(code, m, force_refresh=force))
        return jsonify({"text": text})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/macro", methods=["POST"])
def get_macro():
    try:
        from insight_engine import InsightEngine
        all_metrics = analytics.compute_all_metrics(DB_PATH)
        engine = InsightEngine(DB_PATH)
        force  = request.json.get("force", False) if request.is_json else False
        text   = _run_async(engine.get_macro_insight(all_metrics, force_refresh=force))
        return jsonify({"text": text})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh", methods=["POST"])
def refresh_data():
    try:
        pipe = COTPipeline(DATA_DIR, DB_PATH)
        rows, latest = pipe.refresh()
        return jsonify({"rows_added": rows, "latest_date": str(latest)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    database.init_db(DB_PATH)
    logging.basicConfig(level=logging.WARNING)
    port = int(os.environ.get("COT_PORT", 5050))
    Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    gemini_key    = os.environ.get("GEMINI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if gemini_key:
        print(f"LLM provider  : Gemini  (key: {gemini_key[:8]}...{gemini_key[-4:]})")
    elif anthropic_key:
        print(f"LLM provider  : Claude  (key: {anthropic_key[:12]}...{anthropic_key[-4:]})")
    else:
        print("LLM provider  : NONE — insight buttons will error")
    print(f"COT Dashboard -> http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
