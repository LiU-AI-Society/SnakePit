import logging
import os
import typing
import json
from flask import Flask, request
from datetime import datetime

GAME_ID_MAP = {}
NEXT_GAME_NUMBER = 1


def log_request():
    # Log each incoming request as a one-line JSON entry.
    data = request.get_json(silent=True)
    if not data:
        return  # nothing to log
    game_id = data.get("game", {}).get("id")
    game_key = game_id if game_id is not None else "default"
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    global GAME_ID_MAP
    if game_key not in GAME_ID_MAP:
        # Generate filename using current day, month, hour, and minute, e.g., "26april18_05"
        game_filename = datetime.now().strftime("%d%B%H_%M").lower()
        GAME_ID_MAP[game_key] = game_filename
    game_filename = GAME_ID_MAP[game_key]
    log_file = os.path.join(log_dir, f"{game_filename}.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(data) + "\n")


def run_server(handlers: typing.Dict, use_ngrok=False):
    app = Flask("Battlesnake")

    # Register log_request to log every incoming request.
    app.before_request(log_request)

    @app.get("/")
    def on_info():
        return handlers["info"]()

    @app.post("/start")
    def on_start():
        game_state = request.get_json()
        handlers["start"](game_state)
        return "ok"

    @app.post("/move")
    def on_move():
        game_state = request.get_json()
        return handlers["move"](game_state)

    @app.post("/end")
    def on_end():
        game_state = request.get_json()
        handlers["end"](game_state)
        # Print the log filename after game ends.
        game_id = game_state.get("game", {}).get("id")
        game_key = game_id if game_id is not None else "default"
        # Use the generated filename
        game_filename = GAME_ID_MAP.get(game_key, "unknown")
        print(f"Game saved to {game_filename}.jsonl")
        return "ok"

    @app.after_request
    def identify_server(response):
        response.headers.set(
            "server", "battlesnake/github/starter-snake-python"
        )
        return response

    host = "127.0.0.1"  # Default to localhost
    port = 8080  # Default to port 80

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

   
    print(f"\nRunning Battlesnake at http://{host}:{port}")

    app.run(host=host, port=port)