"""
Ingesta de partidos y estadísticas de equipo.

Fuentes:
  - nba_api LeagueGameLog  → box scores históricos + stats de equipo
  - ESPN Scoreboard API    → agenda de hoy + líneas O/U
"""

import time
import sqlite3
import requests
import pandas as pd
from datetime import date, timedelta
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams as nba_teams_static

from .setup_db import get_connection, DB_PATH

SEASON = '2026-27'
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
RATE_LIMIT_SLEEP = 0.8  # segundos entre llamadas a nba_api


def _abbr_map() -> dict:
    """Mapeo de nombre completo → abreviatura NBA."""
    return {t['full_name']: t['abbreviation'] for t in nba_teams_static.get_teams()}


def fetch_season_game_log(season: str = SEASON, db_path: str = DB_PATH) -> int:
    """
    Descarga el game log completo de la temporada y lo guarda en `games`.
    Devuelve el número de filas insertadas/actualizadas.
    """
    print(f"Descargando game log {season}...")

    gl = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star='Regular Season',
        direction='ASC',
    )
    time.sleep(RATE_LIMIT_SLEEP)

    df = gl.get_data_frames()[0]
    if df.empty:
        print("Sin datos aún (temporada no iniciada).")
        return 0

    # Normalizar columnas
    df.columns = [c.upper() for c in df.columns]
    df['DATE'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')

    # Determinar home/away desde MATCHUP ("BOS vs. LAL" = home, "LAL @ BOS" = away)
    df['IS_HOME'] = df['MATCHUP'].apply(lambda m: 1 if 'vs.' in m else 0)
    df['OPPONENT'] = df['MATCHUP'].apply(
        lambda m: m.split('vs. ')[-1] if 'vs.' in m else m.split('@ ')[-1]
    )

    rows = df[[
        'GAME_ID', 'DATE', 'TEAM_ABBREVIATION', 'OPPONENT',
        'IS_HOME', 'PTS', 'IS_HOME',
        'FGA', 'OREB', 'TOV', 'FTA',
        'WL',
    ]].copy()
    rows.rename(columns={
        'TEAM_ABBREVIATION': 'team',
        'PTS': 'pts',
        'FGA': 'fga',
        'OREB': 'oreb',
        'TOV': 'tov',
        'FTA': 'fta',
    }, inplace=True)
    rows['won'] = (rows['WL'] == 'W').astype(int)
    rows['status'] = 'final'

    # Para opp_pts y stats del rival, hacemos join del mismo game_id con el otro equipo
    # Construimos un DataFrame "espejo" para el oponente
    mirror = rows[['GAME_ID', 'team', 'pts', 'fga', 'oreb', 'tov', 'fta']].copy()
    mirror.columns = ['GAME_ID', 'opponent', 'opp_pts', 'opp_fga', 'opp_oreb', 'opp_tov', 'opp_fta']

    rows = rows.merge(mirror, left_on=['GAME_ID', 'OPPONENT'], right_on=['GAME_ID', 'opponent'], how='left')

    with get_connection(db_path) as conn:
        inserted = 0
        for _, r in rows.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO games
                (game_id, date, team, opponent, is_home,
                 pts, opp_pts, fga, oreb, tov, fta,
                 opp_fga, opp_oreb, opp_tov, opp_fta,
                 min_played, won, status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                str(r['GAME_ID']), r['DATE'], r['team'], r['OPPONENT'],
                int(r['IS_HOME']),
                _safe_int(r.get('pts')), _safe_int(r.get('opp_pts')),
                _safe_int(r.get('fga')), _safe_int(r.get('oreb')),
                _safe_int(r.get('tov')), _safe_int(r.get('fta')),
                _safe_int(r.get('opp_fga')), _safe_int(r.get('opp_oreb')),
                _safe_int(r.get('opp_tov')), _safe_int(r.get('opp_fta')),
                240.0,  # 5 periodos × 48 min (aproximado)
                int(r['won']), 'final',
            ))
            inserted += 1

    print(f"Partidos ingresados/actualizados: {inserted}")
    return inserted


def fetch_today_schedule(db_path: str = DB_PATH) -> int:
    """
    Obtiene el calendario de hoy + líneas O/U desde ESPN.
    Inserta en `games` (status='scheduled') y en `game_lines`.
    """
    today_str = date.today().strftime('%Y%m%d')
    print(f"Descargando agenda de hoy ({today_str}) desde ESPN...")

    resp = requests.get(ESPN_SCOREBOARD, params={'dates': today_str}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    inserted = 0
    with get_connection(db_path) as conn:
        for event in data.get('events', []):
            comp = event['competitions'][0]
            game_id = f"ESPN_{event['id']}"
            game_date = event['date'][:10]

            competitors = {c['homeAway']: c for c in comp['competitors']}
            home = competitors.get('home', {})
            away = competitors.get('away', {})

            home_abbr = home.get('team', {}).get('abbreviation', '')
            away_abbr = away.get('team', {}).get('abbreviation', '')

            status_obj = comp.get('status', {}).get('type', {})
            is_final = status_obj.get('completed', False)
            status = 'final' if is_final else 'scheduled'

            home_pts = _safe_int(home.get('score')) if is_final else None
            away_pts = _safe_int(away.get('score')) if is_final else None

            # Línea O/U
            odds = comp.get('odds', [{}])[0] if comp.get('odds') else {}
            over_under = odds.get('overUnder')
            spread = odds.get('spread')

            # Insertar línea de apuestas
            conn.execute("""
                INSERT OR REPLACE INTO game_lines
                (game_id, date, home_team, away_team, over_under, home_spread)
                VALUES (?,?,?,?,?,?)
            """, (game_id, game_date, home_abbr, away_abbr, over_under, spread))

            # Insertar filas del partido (home y away)
            for abbr, opp_abbr, is_home, pts, opp_pts in [
                (home_abbr, away_abbr, 1, home_pts, away_pts),
                (away_abbr, home_abbr, 0, away_pts, home_pts),
            ]:
                conn.execute("""
                    INSERT OR IGNORE INTO games
                    (game_id, date, team, opponent, is_home,
                     pts, opp_pts, won, status)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (game_id, game_date, abbr, opp_abbr, is_home,
                      pts, opp_pts, None, status))

            inserted += 1

    print(f"Partidos de hoy: {inserted}")
    return inserted


def update_final_scores(db_path: str = DB_PATH) -> int:
    """
    Actualiza scores de partidos que ya terminaron (status scheduled → final).
    """
    yesterday = (date.today() - timedelta(days=1)).strftime('%Y%m%d')
    resp = requests.get(ESPN_SCOREBOARD, params={'dates': yesterday}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    updated = 0
    with get_connection(db_path) as conn:
        for event in data.get('events', []):
            comp = event['competitions'][0]
            if not comp.get('status', {}).get('type', {}).get('completed', False):
                continue

            game_id = f"ESPN_{event['id']}"
            competitors = {c['homeAway']: c for c in comp['competitors']}
            home_pts = _safe_int(competitors.get('home', {}).get('score'))
            away_pts = _safe_int(competitors.get('away', {}).get('score'))

            home_abbr = competitors.get('home', {}).get('team', {}).get('abbreviation', '')
            away_abbr = competitors.get('away', {}).get('team', {}).get('abbreviation', '')

            conn.execute("""
                UPDATE games SET pts=?, opp_pts=?, won=?, status='final'
                WHERE game_id=? AND team=?
            """, (home_pts, away_pts, 1 if home_pts > away_pts else 0, game_id, home_abbr))

            conn.execute("""
                UPDATE games SET pts=?, opp_pts=?, won=?, status='final'
                WHERE game_id=? AND team=?
            """, (away_pts, home_pts, 1 if away_pts > home_pts else 0, game_id, away_abbr))

            # Rellenar actual_pts en predictions (para calcular error del modelo)
            conn.execute("""
                UPDATE predictions
                SET actual_home_pts=?, actual_away_pts=?
                WHERE game_id=?
            """, (home_pts, away_pts, game_id))

            updated += 1

    print(f"Partidos actualizados con resultado final: {updated}")
    return updated


def _safe_int(val) -> int | None:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


if __name__ == '__main__':
    fetch_today_schedule()
    update_final_scores()
