"""
Exporta un snapshot diario de la BD a public/data.json
para el dashboard estático en Netlify.
"""

import json
import os
import sqlite3
from datetime import datetime, date, timedelta

from .setup_db import get_connection, DB_PATH

PUBLIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'public')
OUTPUT_PATH = os.path.join(PUBLIC_DIR, 'data.json')


def export_json(db_path: str = DB_PATH) -> None:
    today = date.today().isoformat()

    with get_connection(db_path) as conn:
        # Métricas OOS acumuladas
        metrics_row = conn.execute("""
            SELECT
                COUNT(*)                                          AS n_games,
                AVG(ABS(actual_home_pts - predicted_home_pts))    AS mae_home,
                AVG(ABS(actual_away_pts - predicted_away_pts))    AS mae_away,
                AVG(ABS((actual_home_pts+actual_away_pts) - predicted_total)) AS mae_total,
                AVG((actual_home_pts+actual_away_pts - predicted_total)
                    * (actual_home_pts+actual_away_pts - predicted_total)) AS mse_total
            FROM predictions
            WHERE actual_home_pts IS NOT NULL
        """).fetchone()

        vegas_row = conn.execute("""
            SELECT AVG(ABS(gl.over_under - (p.actual_home_pts + p.actual_away_pts))) AS mae_vegas
            FROM predictions p
            JOIN game_lines gl ON gl.game_id = p.game_id
            WHERE p.actual_home_pts IS NOT NULL AND gl.over_under IS NOT NULL
        """).fetchone()

        # Predicciones de hoy
        today_preds = conn.execute("""
            SELECT p.game_id, gl.home_team, gl.away_team,
                   p.predicted_home_pts, p.predicted_away_pts, p.predicted_total,
                   gl.over_under,
                   p.actual_home_pts, p.actual_away_pts
            FROM predictions p
            JOIN game_lines gl ON gl.game_id = p.game_id
            WHERE gl.date = ?
            ORDER BY gl.game_id
        """, (today,)).fetchall()

        # Últimos 15 resultados resueltos
        recent = conn.execute("""
            SELECT p.game_id, gl.home_team, gl.away_team, gl.date,
                   p.predicted_home_pts, p.predicted_away_pts, p.predicted_total,
                   p.actual_home_pts, p.actual_away_pts,
                   (p.actual_home_pts + p.actual_away_pts) AS actual_total,
                   gl.over_under,
                   ABS(p.predicted_total - (p.actual_home_pts + p.actual_away_pts)) AS error_total
            FROM predictions p
            JOIN game_lines gl ON gl.game_id = p.game_id
            WHERE p.actual_home_pts IS NOT NULL
            ORDER BY gl.date DESC
            LIMIT 15
        """).fetchall()

        # Injury report del día
        injuries = conn.execute("""
            SELECT team, player_name, status
            FROM player_availability
            WHERE DATE(recorded_at) = ?
              AND LOWER(status) IN ('out', 'doubtful', 'day-to-day')
            ORDER BY team, status, player_name
        """, (today,)).fetchall()

    def _r(v, decimals=1):
        return round(float(v), decimals) if v is not None else None

    rmse_total = (float(metrics_row['mse_total']) ** 0.5
                  if metrics_row['mse_total'] else None)

    payload = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "season": "2026-27",
        "metrics": {
            "n_games": int(metrics_row['n_games'] or 0),
            "mae_home": _r(metrics_row['mae_home']),
            "mae_away": _r(metrics_row['mae_away']),
            "mae_total": _r(metrics_row['mae_total']),
            "rmse_total": _r(rmse_total),
            "mae_vegas_total": _r(vegas_row['mae_vegas']),
        },
        "predictions_today": [
            {
                "game_id": r['game_id'],
                "home": r['home_team'],
                "away": r['away_team'],
                "pred_home": _r(r['predicted_home_pts']),
                "pred_away": _r(r['predicted_away_pts']),
                "pred_total": _r(r['predicted_total']),
                "vegas_ou": _r(r['over_under']),
                "actual_home": r['actual_home_pts'],
                "actual_away": r['actual_away_pts'],
            }
            for r in today_preds
        ],
        "recent_results": [
            {
                "date": r['date'],
                "home": r['home_team'],
                "away": r['away_team'],
                "pred_home": _r(r['predicted_home_pts']),
                "pred_away": _r(r['predicted_away_pts']),
                "pred_total": _r(r['predicted_total']),
                "actual_home": r['actual_home_pts'],
                "actual_away": r['actual_away_pts'],
                "actual_total": r['actual_total'],
                "vegas_ou": _r(r['over_under']),
                "error_total": _r(r['error_total']),
            }
            for r in recent
        ],
        "injury_report": [
            {"team": r['team'], "player": r['player_name'], "status": r['status']}
            for r in injuries
        ],
    }

    os.makedirs(PUBLIC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"data.json exportado → {len(today_preds)} predicciones hoy, "
          f"{len(recent)} resultados recientes, {len(injuries)} lesiones")


if __name__ == '__main__':
    export_json()
