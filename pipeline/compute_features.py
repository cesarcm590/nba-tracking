"""
Computa rolling features por equipo y las guarda en rolling_stats.

Métricas calculadas:
  - OffRtg  = PTS / POSS × 100
  - DefRtg  = OPP_PTS / OPP_POSS × 100
  - Pace    = POSS / (MIN / 48)
  - días de descanso, back-to-back, win streak
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

from .setup_db import get_connection, DB_PATH


def _possessions(fga, oreb, tov, fta) -> float:
    """Estimación estándar de posesiones."""
    return fga - oreb + tov + 0.44 * fta


def _rolling_ratings(df: pd.DataFrame, window: int) -> pd.Series:
    """OffRtg, DefRtg y Pace rolling para un equipo ordenado por fecha."""
    poss = df.apply(lambda r: _possessions(r.fga or 0, r.oreb or 0, r.tov or 0, r.fta or 0), axis=1)
    opp_poss = df.apply(lambda r: _possessions(r.opp_fga or 0, r.opp_oreb or 0, r.opp_tov or 0, r.opp_fta or 0), axis=1)

    off_rtg = (df['pts'] / poss.replace(0, np.nan)) * 100
    def_rtg = (df['opp_pts'] / opp_poss.replace(0, np.nan)) * 100
    pace = (poss + opp_poss) / 2 / (df['min_played'].replace(0, np.nan) / 48)

    return pd.DataFrame({
        'off_rtg': off_rtg,
        'def_rtg': def_rtg,
        'pace': pace,
        'pts': df['pts'],
        'opp_pts': df['opp_pts'],
    })


def _win_streak(won_series: pd.Series) -> int:
    """Racha actual: positivo = victorias, negativo = derrotas."""
    streak = 0
    for w in reversed(won_series.tolist()):
        if w is None:
            break
        if streak == 0:
            streak = 1 if w else -1
        elif (w and streak > 0) or (not w and streak < 0):
            streak += (1 if w else -1)
        else:
            break
    return streak


def compute_rolling_stats(db_path: str = DB_PATH) -> int:
    """
    Lee todos los partidos finalizados y calcula rolling stats para cada
    (equipo, fecha). Sólo procesa fechas nuevas que no estén ya en rolling_stats.
    """
    with get_connection(db_path) as conn:
        games = pd.read_sql_query("""
            SELECT game_id, date, team, opponent, is_home,
                   pts, opp_pts, fga, oreb, tov, fta,
                   opp_fga, opp_oreb, opp_tov, opp_fta,
                   min_played, won
            FROM games
            WHERE status = 'final'
            ORDER BY team, date
        """, conn)

        existing = pd.read_sql_query(
            "SELECT team, date FROM rolling_stats", conn
        )

    if games.empty:
        print("Sin partidos finales para computar features.")
        return 0

    existing_set = set(zip(existing['team'], existing['date']))
    rows_to_insert = []

    for team, group in games.groupby('team'):
        group = group.sort_values('date').reset_index(drop=True)
        stats = _rolling_ratings(group, window=0)  # calcula por fila, ventana se aplica abajo

        dates = group['date'].tolist()
        wons = group['won'].tolist()

        for i, (_, row) in enumerate(group.iterrows()):
            key = (team, row['date'])
            if key in existing_set:
                continue

            def _roll(col, w):
                start = max(0, i - w)
                vals = stats[col].iloc[start:i].dropna()
                return round(float(vals.mean()), 4) if len(vals) >= 1 else None

            # Días de descanso
            if i == 0:
                days_rest = 7  # valor neutro al inicio de temporada
            else:
                prev_date = pd.Timestamp(dates[i - 1])
                curr_date = pd.Timestamp(row['date'])
                days_rest = (curr_date - prev_date).days - 1

            is_b2b = 1 if days_rest == 0 else 0

            rows_to_insert.append({
                'team': team,
                'date': row['date'],
                'off_rtg_5': _roll('off_rtg', 5),
                'off_rtg_10': _roll('off_rtg', 10),
                'off_rtg_15': _roll('off_rtg', 15),
                'def_rtg_5': _roll('def_rtg', 5),
                'def_rtg_10': _roll('def_rtg', 10),
                'def_rtg_15': _roll('def_rtg', 15),
                'pace_5': _roll('pace', 5),
                'pace_10': _roll('pace', 10),
                'pts_avg_5': _roll('pts', 5),
                'pts_allowed_avg_5': _roll('opp_pts', 5),
                'days_rest': days_rest,
                'is_back_to_back': is_b2b,
                'win_streak': _win_streak(pd.Series(wons[:i])),
            })

    if not rows_to_insert:
        print("Rolling stats ya actualizadas.")
        return 0

    with get_connection(db_path) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO rolling_stats
            (team, date, off_rtg_5, off_rtg_10, off_rtg_15,
             def_rtg_5, def_rtg_10, def_rtg_15,
             pace_5, pace_10, pts_avg_5, pts_allowed_avg_5,
             days_rest, is_back_to_back, win_streak)
            VALUES (:team, :date, :off_rtg_5, :off_rtg_10, :off_rtg_15,
                    :def_rtg_5, :def_rtg_10, :def_rtg_15,
                    :pace_5, :pace_10, :pts_avg_5, :pts_allowed_avg_5,
                    :days_rest, :is_back_to_back, :win_streak)
        """, rows_to_insert)

    print(f"Rolling stats calculadas: {len(rows_to_insert)} filas nuevas")
    return len(rows_to_insert)


def build_feature_row(home_team: str, away_team: str, game_date: str,
                      db_path: str = DB_PATH) -> dict | None:
    """
    Construye el vector de features para un partido próximo.
    Combina rolling stats de ambos equipos + head-to-head + injury pct.
    Retorna None si no hay suficiente historia.
    """
    with get_connection(db_path) as conn:
        home_stats = conn.execute("""
            SELECT * FROM rolling_stats
            WHERE team = ? AND date < ?
            ORDER BY date DESC LIMIT 1
        """, (home_team, game_date)).fetchone()

        away_stats = conn.execute("""
            SELECT * FROM rolling_stats
            WHERE team = ? AND date < ?
            ORDER BY date DESC LIMIT 1
        """, (away_team, game_date)).fetchone()

        # Head-to-head: promedio de puntos en los últimos 3 enfrentamientos
        h2h = conn.execute("""
            SELECT AVG(pts) as avg_home_pts, AVG(opp_pts) as avg_away_pts, COUNT(*) as n
            FROM games
            WHERE team = ? AND opponent = ? AND status = 'final'
            ORDER BY date DESC LIMIT 3
        """, (home_team, away_team)).fetchone()

    if not home_stats or not away_stats:
        return None

    def _s(row, col, default=None):
        return row[col] if row[col] is not None else default

    features = {
        'home_off_rtg_5': _s(home_stats, 'off_rtg_5'),
        'home_off_rtg_10': _s(home_stats, 'off_rtg_10'),
        'home_def_rtg_5': _s(home_stats, 'def_rtg_5'),
        'home_def_rtg_10': _s(home_stats, 'def_rtg_10'),
        'home_pace_5': _s(home_stats, 'pace_5'),
        'home_pace_10': _s(home_stats, 'pace_10'),
        'home_pts_avg_5': _s(home_stats, 'pts_avg_5'),
        'home_pts_allowed_avg_5': _s(home_stats, 'pts_allowed_avg_5'),
        'home_days_rest': _s(home_stats, 'days_rest', 3),
        'home_is_b2b': _s(home_stats, 'is_back_to_back', 0),
        'home_win_streak': _s(home_stats, 'win_streak', 0),
        'away_off_rtg_5': _s(away_stats, 'off_rtg_5'),
        'away_off_rtg_10': _s(away_stats, 'off_rtg_10'),
        'away_def_rtg_5': _s(away_stats, 'def_rtg_5'),
        'away_def_rtg_10': _s(away_stats, 'def_rtg_10'),
        'away_pace_5': _s(away_stats, 'pace_5'),
        'away_pace_10': _s(away_stats, 'pace_10'),
        'away_pts_avg_5': _s(away_stats, 'pts_avg_5'),
        'away_pts_allowed_avg_5': _s(away_stats, 'pts_allowed_avg_5'),
        'away_days_rest': _s(away_stats, 'days_rest', 3),
        'away_is_b2b': _s(away_stats, 'is_back_to_back', 0),
        'away_win_streak': _s(away_stats, 'win_streak', 0),
        'h2h_avg_home_pts': h2h['avg_home_pts'] if h2h and h2h['n'] > 0 else None,
        'h2h_avg_away_pts': h2h['avg_away_pts'] if h2h and h2h['n'] > 0 else None,
        'h2h_games': h2h['n'] if h2h else 0,
    }
    return features


if __name__ == '__main__':
    compute_rolling_stats()
