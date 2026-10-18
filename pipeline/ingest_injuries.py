"""
Ingesta de injury report desde ESPN.
Registra disponibilidad de jugadores antes de cada partido.
"""

import requests
import sqlite3
from datetime import datetime

from .setup_db import get_connection, DB_PATH

ESPN_INJURIES = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries'
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'


# Mapeo ESPN → NBA abreviatura estándar
# ESPN usa algunas abreviaturas distintas a las de nba_api
ESPN_TO_NBA = {
    'GS': 'GSW', 'NY': 'NYK', 'SA': 'SAS', 'NO': 'NOP',
    'OKC': 'OKC', 'UTAH': 'UTA', 'WSH': 'WAS', 'CHA': 'CHA',
}


def _normalize_abbr(abbr: str) -> str:
    return ESPN_TO_NBA.get(abbr.upper(), abbr.upper())


def fetch_injury_report(db_path: str = DB_PATH) -> int:
    """
    Descarga el injury report actual de ESPN y lo guarda en player_availability.
    Intenta asociar cada jugador al próximo game_id de su equipo.
    """
    print("Descargando injury report desde ESPN...")

    resp = requests.get(ESPN_INJURIES, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Obtener próximos partidos de hoy para vincular game_id
    game_ids_by_team = _get_todays_game_ids()
    recorded_at = datetime.utcnow().isoformat()

    rows = []
    for team_entry in data.get('injuries', []):
        team_abbr = _normalize_abbr(team_entry.get('team', {}).get('abbreviation', ''))
        game_id = game_ids_by_team.get(team_abbr)

        for injury in team_entry.get('injuries', []):
            athlete = injury.get('athlete', {})
            rows.append({
                'game_id': game_id,
                'team': team_abbr,
                'player_id': str(athlete.get('id', '')),
                'player_name': athlete.get('displayName', ''),
                'status': injury.get('status', ''),
                'season_avg_minutes': None,
                'recorded_at': recorded_at,
            })

    with get_connection(db_path) as conn:
        for r in rows:
            conn.execute("""
                INSERT OR REPLACE INTO player_availability
                (game_id, team, player_id, player_name, status, season_avg_minutes, recorded_at)
                VALUES (:game_id, :team, :player_id, :player_name, :status, :season_avg_minutes, :recorded_at)
            """, r)

    print(f"Jugadores con estatus registrado: {len(rows)}")
    return len(rows)


def get_active_minutes_pct(team: str, game_id: str, db_path: str = DB_PATH) -> float | None:
    """
    Calcula el % de minutos disponibles vs roster completo.
    Retorna None si no hay datos suficientes.

    Lógica: players con status 'Out' o 'Doubtful' se cuentan como no disponibles.
    Se usa season_avg_minutes si está disponible, sino peso igual por jugador.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT status, season_avg_minutes
            FROM player_availability
            WHERE team = ? AND game_id = ?
        """, (team, game_id)).fetchall()

    if not rows:
        return None

    unavailable = {'out', 'doubtful'}
    total = len(rows)
    out_count = sum(1 for r in rows if r['status'].lower() in unavailable)

    if total == 0:
        return None

    return round((total - out_count) / total, 4)


def _get_todays_game_ids() -> dict:
    """Devuelve {team_abbr: game_id} para los partidos de hoy."""
    from datetime import date
    today_str = date.today().strftime('%Y%m%d')
    try:
        resp = requests.get(ESPN_SCOREBOARD, params={'dates': today_str}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    result = {}
    for event in data.get('events', []):
        game_id = f"ESPN_{event['id']}"
        for comp in event.get('competitions', []):
            for competitor in comp.get('competitors', []):
                abbr = _normalize_abbr(competitor.get('team', {}).get('abbreviation', ''))
                result[abbr] = game_id

    return result


if __name__ == '__main__':
    fetch_injury_report()
