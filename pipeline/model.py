"""
Entrenamiento y predicción — Random Forest para puntos por equipo.

Flujo:
  train()    → entrena RF home_pts y away_pts, guarda artefactos + log en model_versions
  predict()  → genera predicciones para hoy, loguea con timestamp en predictions
  evaluate() → calcula MAE/RMSE sobre predicciones ya resueltas (para paper)
"""

import os
import json
import joblib
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline

from .setup_db import get_connection, DB_PATH

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURE_COLS = [
    'home_off_rtg_5', 'home_off_rtg_10', 'home_def_rtg_5', 'home_def_rtg_10',
    'home_pace_5', 'home_pace_10', 'home_pts_avg_5', 'home_pts_allowed_avg_5',
    'home_days_rest', 'home_is_b2b', 'home_win_streak',
    'away_off_rtg_5', 'away_off_rtg_10', 'away_def_rtg_5', 'away_def_rtg_10',
    'away_pace_5', 'away_pace_10', 'away_pts_avg_5', 'away_pts_allowed_avg_5',
    'away_days_rest', 'away_is_b2b', 'away_win_streak',
    'h2h_avg_home_pts', 'h2h_avg_away_pts',
    'home_active_pct', 'away_active_pct',
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _models_path(version_id: str) -> tuple[str, str]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return (
        os.path.join(MODELS_DIR, f'{version_id}_home.pkl'),
        os.path.join(MODELS_DIR, f'{version_id}_away.pkl'),
    )


def _latest_version(db_path: str) -> str | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT version_id FROM model_versions ORDER BY trained_at DESC LIMIT 1"
        ).fetchone()
    return row['version_id'] if row else None


def _load_models(version_id: str) -> tuple:
    h_path, a_path = _models_path(version_id)
    return joblib.load(h_path), joblib.load(a_path)


def _build_training_df(db_path: str) -> pd.DataFrame:
    """
    Construye el DataFrame de entrenamiento uniendo rolling_stats de ambos equipos
    con los resultados reales de cada partido.
    Las features son PRE-partido (rolling_stats.date = fecha del partido,
    computadas sobre partidos ANTERIORES a esa fecha).
    """
    query = """
    SELECT
        gl.game_id,
        gl.date,
        gl.home_team,
        gl.away_team,

        h.off_rtg_5   AS home_off_rtg_5,
        h.off_rtg_10  AS home_off_rtg_10,
        h.def_rtg_5   AS home_def_rtg_5,
        h.def_rtg_10  AS home_def_rtg_10,
        h.pace_5      AS home_pace_5,
        h.pace_10     AS home_pace_10,
        h.pts_avg_5         AS home_pts_avg_5,
        h.pts_allowed_avg_5 AS home_pts_allowed_avg_5,
        h.days_rest         AS home_days_rest,
        h.is_back_to_back   AS home_is_b2b,
        h.win_streak        AS home_win_streak,

        a.off_rtg_5   AS away_off_rtg_5,
        a.off_rtg_10  AS away_off_rtg_10,
        a.def_rtg_5   AS away_def_rtg_5,
        a.def_rtg_10  AS away_def_rtg_10,
        a.pace_5      AS away_pace_5,
        a.pace_10     AS away_pace_10,
        a.pts_avg_5         AS away_pts_avg_5,
        a.pts_allowed_avg_5 AS away_pts_allowed_avg_5,
        a.days_rest         AS away_days_rest,
        a.is_back_to_back   AS away_is_b2b,
        a.win_streak        AS away_win_streak,

        hg.pts  AS home_pts,
        ag.pts  AS away_pts
    FROM game_lines gl
    JOIN rolling_stats h  ON h.team = gl.home_team AND h.date = gl.date
    JOIN rolling_stats a  ON a.team = gl.away_team AND a.date = gl.date
    JOIN games hg ON hg.game_id = gl.game_id AND hg.team = gl.home_team AND hg.status = 'final'
    JOIN games ag ON ag.game_id = gl.game_id AND ag.team = gl.away_team AND ag.status = 'final'
    WHERE hg.pts IS NOT NULL AND ag.pts IS NOT NULL
    """
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    # Head-to-head: promedio últimos 3 enfrentamientos por cada par home/away
    h2h_rows = []
    with get_connection(db_path) as conn:
        for _, row in df.iterrows():
            h2h = conn.execute("""
                SELECT AVG(pts) as avg_h, AVG(opp_pts) as avg_a
                FROM (
                    SELECT pts, opp_pts FROM games
                    WHERE team = ? AND opponent = ? AND status = 'final' AND date < ?
                    ORDER BY date DESC LIMIT 3
                )
            """, (row['home_team'], row['away_team'], row['date'])).fetchone()
            h2h_rows.append({
                'game_id': row['game_id'],
                'h2h_avg_home_pts': h2h['avg_h'],
                'h2h_avg_away_pts': h2h['avg_a'],
            })

    if h2h_rows:
        h2h_df = pd.DataFrame(h2h_rows)
        df = df.merge(h2h_df, on='game_id', how='left')
    else:
        df['h2h_avg_home_pts'] = np.nan
        df['h2h_avg_away_pts'] = np.nan

    # Injury pct por partido
    with get_connection(db_path) as conn:
        avail = pd.read_sql_query("""
            SELECT game_id, team,
                   SUM(CASE WHEN LOWER(status) NOT IN ('out','doubtful') THEN 1.0 ELSE 0 END)
                   / COUNT(*) AS active_pct
            FROM player_availability
            GROUP BY game_id, team
        """, conn)

    home_avail = avail.rename(columns={'active_pct': 'home_active_pct', 'team': 'home_team'})
    away_avail = avail.rename(columns={'active_pct': 'away_active_pct', 'team': 'away_team'})

    df = df.merge(home_avail[['game_id', 'home_team', 'home_active_pct']], on=['game_id', 'home_team'], how='left')
    df = df.merge(away_avail[['game_id', 'away_team', 'away_active_pct']], on=['game_id', 'away_team'], how='left')

    return df


# ── entrenamiento ─────────────────────────────────────────────────────────────

def train(db_path: str = DB_PATH, n_estimators: int = 200,
          max_depth: int = 8, min_samples_leaf: int = 5) -> str | None:
    """
    Entrena dos modelos RF (home_pts, away_pts) con todos los partidos disponibles.
    Requiere mínimo 20 partidos. Devuelve el version_id o None si no hay datos.
    """
    df = _build_training_df(db_path)

    if len(df) < 20:
        print(f"Datos insuficientes para entrenar ({len(df)} partidos, mínimo 20).")
        return None

    X = df[FEATURE_COLS]
    y_home = df['home_pts']
    y_away = df['away_pts']

    rf_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )

    # Pipeline con imputación (NaN frecuentes en inicio de temporada)
    def make_pipeline():
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('rf', RandomForestRegressor(**rf_params)),
        ])

    pipe_home = make_pipeline()
    pipe_away = make_pipeline()

    pipe_home.fit(X, y_home)
    pipe_away.fit(X, y_away)

    # Métricas in-sample (orientativas; las del paper son OOS prospectivas)
    mae_home = mean_absolute_error(y_home, pipe_home.predict(X))
    mae_away = mean_absolute_error(y_away, pipe_away.predict(X))
    rmse_home = root_mean_squared_error(y_home, pipe_home.predict(X))
    rmse_away = root_mean_squared_error(y_away, pipe_away.predict(X))

    version_id = f"v{datetime.now().strftime('%Y%m%d')}"
    h_path, a_path = _models_path(version_id)
    joblib.dump(pipe_home, h_path)
    joblib.dump(pipe_away, a_path)

    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO model_versions
            (version_id, trained_at, train_from, train_to, n_games,
             mae_home, mae_away, rmse_home, rmse_away, params)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            version_id,
            datetime.utcnow().isoformat(),
            str(df['date'].min()),
            str(df['date'].max()),
            len(df),
            round(mae_home, 3),
            round(mae_away, 3),
            round(rmse_home, 3),
            round(rmse_away, 3),
            json.dumps(rf_params),
        ))

    print(f"Modelo entrenado: {version_id} | {len(df)} partidos")
    print(f"  MAE local={mae_home:.1f} pts  visitante={mae_away:.1f} pts")
    print(f"  RMSE local={rmse_home:.1f}   visitante={rmse_away:.1f}")
    return version_id


# ── predicción ────────────────────────────────────────────────────────────────

def predict(db_path: str = DB_PATH, version_id: str | None = None) -> list[dict]:
    """
    Genera predicciones para los partidos de hoy.
    CRÍTICO: predicted_at se graba ANTES del tip-off (garantía del paper).
    """
    version_id = version_id or _latest_version(db_path)
    if not version_id:
        print("Sin modelo entrenado. Corre train() primero.")
        return []

    pipe_home, pipe_away = _load_models(version_id)
    today = date.today().isoformat()

    with get_connection(db_path) as conn:
        games_today = conn.execute("""
            SELECT game_id, home_team, away_team, date
            FROM game_lines
            WHERE date = ?
        """, (today,)).fetchall()

    if not games_today:
        print(f"Sin partidos programados para hoy ({today}).")
        return []

    results = []
    predicted_at = datetime.utcnow().isoformat()

    for game in games_today:
        features = _build_prediction_features(
            game['home_team'], game['away_team'], today, game['game_id'], db_path
        )
        if features is None:
            print(f"  Sin features para {game['home_team']} vs {game['away_team']} — saltando")
            continue

        X = pd.DataFrame([features])[FEATURE_COLS]
        pred_home = float(pipe_home.predict(X)[0])
        pred_away = float(pipe_away.predict(X)[0])

        with get_connection(db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO predictions
                (game_id, model_version, predicted_home_pts, predicted_away_pts,
                 predicted_total, predicted_at)
                VALUES (?,?,?,?,?,?)
            """, (
                game['game_id'], version_id,
                round(pred_home, 1), round(pred_away, 1),
                round(pred_home + pred_away, 1),
                predicted_at,
            ))

        result = {
            'game_id': game['game_id'],
            'home': game['home_team'],
            'away': game['away_team'],
            'pred_home': round(pred_home, 1),
            'pred_away': round(pred_away, 1),
            'pred_total': round(pred_home + pred_away, 1),
        }
        results.append(result)
        print(f"  {game['home_team']} {pred_home:.0f} – {pred_away:.0f} {game['away_team']}  "
              f"(total={pred_home+pred_away:.0f})")

    print(f"Predicciones logueadas: {len(results)} partidos | timestamp={predicted_at[:16]}")
    return results


def _build_prediction_features(home_team: str, away_team: str, game_date: str,
                                 game_id: str, db_path: str) -> dict | None:
    """Features de un partido próximo usando los stats más recientes de cada equipo."""
    with get_connection(db_path) as conn:
        h = conn.execute("""
            SELECT * FROM rolling_stats WHERE team=? AND date < ?
            ORDER BY date DESC LIMIT 1
        """, (home_team, game_date)).fetchone()

        a = conn.execute("""
            SELECT * FROM rolling_stats WHERE team=? AND date < ?
            ORDER BY date DESC LIMIT 1
        """, (away_team, game_date)).fetchone()

        h2h = conn.execute("""
            SELECT AVG(pts) as avg_h, AVG(opp_pts) as avg_a
            FROM (
                SELECT pts, opp_pts FROM games
                WHERE team=? AND opponent=? AND status='final'
                ORDER BY date DESC LIMIT 3
            )
        """, (home_team, away_team)).fetchone()

        home_avail = conn.execute("""
            SELECT SUM(CASE WHEN LOWER(status) NOT IN ('out','doubtful') THEN 1.0 ELSE 0 END)
                   / COUNT(*) AS active_pct
            FROM player_availability WHERE team=? AND game_id=?
        """, (home_team, game_id)).fetchone()

        away_avail = conn.execute("""
            SELECT SUM(CASE WHEN LOWER(status) NOT IN ('out','doubtful') THEN 1.0 ELSE 0 END)
                   / COUNT(*) AS active_pct
            FROM player_availability WHERE team=? AND game_id=?
        """, (away_team, game_id)).fetchone()

    # Sin ningún historial no podemos predecir
    if not h and not a:
        return None

    def _col(row, col):
        return float(row[col]) if row and row[col] is not None else None

    return {
        'home_off_rtg_5': _col(h, 'off_rtg_5'),
        'home_off_rtg_10': _col(h, 'off_rtg_10'),
        'home_def_rtg_5': _col(h, 'def_rtg_5'),
        'home_def_rtg_10': _col(h, 'def_rtg_10'),
        'home_pace_5': _col(h, 'pace_5'),
        'home_pace_10': _col(h, 'pace_10'),
        'home_pts_avg_5': _col(h, 'pts_avg_5'),
        'home_pts_allowed_avg_5': _col(h, 'pts_allowed_avg_5'),
        'home_days_rest': _col(h, 'days_rest'),
        'home_is_b2b': _col(h, 'is_back_to_back'),
        'home_win_streak': _col(h, 'win_streak'),
        'away_off_rtg_5': _col(a, 'off_rtg_5'),
        'away_off_rtg_10': _col(a, 'off_rtg_10'),
        'away_def_rtg_5': _col(a, 'def_rtg_5'),
        'away_def_rtg_10': _col(a, 'def_rtg_10'),
        'away_pace_5': _col(a, 'pace_5'),
        'away_pace_10': _col(a, 'pace_10'),
        'away_pts_avg_5': _col(a, 'pts_avg_5'),
        'away_pts_allowed_avg_5': _col(a, 'pts_allowed_avg_5'),
        'away_days_rest': _col(a, 'days_rest'),
        'away_is_b2b': _col(a, 'is_back_to_back'),
        'away_win_streak': _col(a, 'win_streak'),
        'h2h_avg_home_pts': h2h['avg_h'] if h2h else None,
        'h2h_avg_away_pts': h2h['avg_a'] if h2h else None,
        'home_active_pct': home_avail['active_pct'] if home_avail else None,
        'away_active_pct': away_avail['active_pct'] if away_avail else None,
    }


# ── evaluación (paper) ────────────────────────────────────────────────────────

def evaluate(db_path: str = DB_PATH) -> dict:
    """
    Calcula métricas prospectivas OOS — sólo predicciones logueadas ANTES del tip-off
    donde el resultado real ya está disponible.
    Este es el número que va al paper.
    """
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("""
            SELECT p.game_id, p.model_version, p.predicted_at,
                   p.predicted_home_pts, p.predicted_away_pts, p.predicted_total,
                   p.actual_home_pts, p.actual_away_pts,
                   gl.over_under AS vegas_total,
                   (p.actual_home_pts + p.actual_away_pts) AS actual_total
            FROM predictions p
            JOIN game_lines gl ON gl.game_id = p.game_id
            WHERE p.actual_home_pts IS NOT NULL
        """, conn)

    if df.empty:
        print("Sin predicciones resueltas aún.")
        return {}

    n = len(df)
    mae_home  = mean_absolute_error(df['actual_home_pts'], df['predicted_home_pts'])
    mae_away  = mean_absolute_error(df['actual_away_pts'], df['predicted_away_pts'])
    mae_total = mean_absolute_error(df['actual_total'], df['predicted_total'])
    rmse_home  = root_mean_squared_error(df['actual_home_pts'], df['predicted_home_pts'])
    rmse_away  = root_mean_squared_error(df['actual_away_pts'], df['predicted_away_pts'])
    rmse_total = root_mean_squared_error(df['actual_total'], df['predicted_total'])

    # Benchmark: ¿qué error tiene la línea de Vegas?
    vegas_rows = df.dropna(subset=['vegas_total'])
    mae_vegas = mean_absolute_error(
        vegas_rows['actual_total'], vegas_rows['vegas_total']
    ) if not vegas_rows.empty else None

    metrics = {
        'n_games': n,
        'mae_home': round(mae_home, 2),
        'mae_away': round(mae_away, 2),
        'mae_total': round(mae_total, 2),
        'rmse_home': round(rmse_home, 2),
        'rmse_away': round(rmse_away, 2),
        'rmse_total': round(rmse_total, 2),
        'mae_vegas_total': round(mae_vegas, 2) if mae_vegas else None,
    }

    print(f"\n=== Evaluación prospectiva OOS ({n} partidos) ===")
    print(f"  MAE  local={mae_home:.1f}  visitante={mae_away:.1f}  total={mae_total:.1f}")
    print(f"  RMSE local={rmse_home:.1f}  visitante={rmse_away:.1f}  total={rmse_total:.1f}")
    if mae_vegas:
        delta = mae_total - mae_vegas
        print(f"  Vegas MAE total={mae_vegas:.1f}  Δ vs RF={delta:+.1f} pts")
    return metrics


if __name__ == '__main__':
    train()
    predict()
