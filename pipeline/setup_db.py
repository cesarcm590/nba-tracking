import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'nba.db')


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_tables(db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript("""
        -- Una fila por equipo por partido (2 filas por game_id)
        CREATE TABLE IF NOT EXISTS games (
            game_id     TEXT,
            date        TEXT,
            team        TEXT,
            opponent    TEXT,
            is_home     INTEGER,
            pts         INTEGER,
            opp_pts     INTEGER,
            fga         INTEGER,
            oreb        INTEGER,
            tov         INTEGER,
            fta         INTEGER,
            opp_fga     INTEGER,
            opp_oreb    INTEGER,
            opp_tov     INTEGER,
            opp_fta     INTEGER,
            min_played  REAL,
            won         INTEGER,
            status      TEXT DEFAULT 'scheduled',
            PRIMARY KEY (game_id, team)
        );

        -- Línea de apuestas por partido (para benchmark del paper)
        CREATE TABLE IF NOT EXISTS game_lines (
            game_id      TEXT PRIMARY KEY,
            date         TEXT,
            home_team    TEXT,
            away_team    TEXT,
            over_under   REAL,
            home_spread  REAL
        );

        -- Features rolling computadas (una fila por equipo por fecha)
        CREATE TABLE IF NOT EXISTS rolling_stats (
            team              TEXT,
            date              TEXT,
            off_rtg_5         REAL,
            off_rtg_10        REAL,
            off_rtg_15        REAL,
            def_rtg_5         REAL,
            def_rtg_10        REAL,
            def_rtg_15        REAL,
            pace_5            REAL,
            pace_10           REAL,
            pts_avg_5         REAL,
            pts_allowed_avg_5 REAL,
            days_rest         INTEGER,
            is_back_to_back   INTEGER,
            win_streak        INTEGER,
            PRIMARY KEY (team, date)
        );

        -- Disponibilidad de jugadores (injury report)
        CREATE TABLE IF NOT EXISTS player_availability (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id             TEXT,
            team                TEXT,
            player_id           TEXT,
            player_name         TEXT,
            status              TEXT,
            season_avg_minutes  REAL,
            recorded_at         TEXT,
            UNIQUE (game_id, player_id)
        );

        -- Predicciones del modelo (corazón del paper)
        -- predicted_at DEBE ser anterior al tip-off para validez prospectiva
        CREATE TABLE IF NOT EXISTS predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id             TEXT,
            model_version       TEXT,
            predicted_home_pts  REAL,
            predicted_away_pts  REAL,
            predicted_total     REAL,
            predicted_at        TEXT,
            actual_home_pts     INTEGER,
            actual_away_pts     INTEGER,
            UNIQUE (game_id, model_version)
        );

        -- Log de versiones del modelo (reentrenamiento semanal)
        CREATE TABLE IF NOT EXISTS model_versions (
            version_id   TEXT PRIMARY KEY,
            trained_at   TEXT,
            train_from   TEXT,
            train_to     TEXT,
            n_games      INTEGER,
            mae_home     REAL,
            mae_away     REAL,
            rmse_home    REAL,
            rmse_away    REAL,
            params       TEXT
        );
        """)
    print(f"Base de datos inicializada en: {os.path.abspath(db_path)}")


if __name__ == '__main__':
    create_tables()
