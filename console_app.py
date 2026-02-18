"""Console partner app for platform management and simulation workflows."""

from __future__ import annotations

import argparse
import cmd
import json
import shlex

from internet_hub import extract_keyphrases, fetch_world_events
from platform_db import (
    DEFAULT_DB_PATH,
    authenticate_account,
    create_account,
    create_project,
    get_snapshot,
    init_db,
    list_events,
    list_models,
    list_phrases,
    list_projects,
    register_model,
    seed_common_phrases,
    add_events,
    bulk_upsert_phrases,
)
from simulation_lab import train_agent_with_self_play


class NeuroConsole(cmd.Cmd):
    intro = "NeuroDSL Console ready. Type help or ? to list commands."
    prompt = "neuro> "

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        super().__init__()
        self.db_path = db_path
        init_db(self.db_path)

    def _json(self, payload: dict | list) -> None:
        print(json.dumps(payload, indent=2))

    def do_initdb(self, _arg: str) -> None:
        init_db(self.db_path)
        self._json({"ok": True, "db_path": self.db_path})

    def do_status(self, _arg: str) -> None:
        self._json(get_snapshot(self.db_path))

    def do_signup(self, arg: str) -> None:
        parts = shlex.split(arg)
        if len(parts) < 2:
            print("usage: signup <username> <password> [lang]")
            return
        lang = parts[2] if len(parts) > 2 else "en"
        account = create_account(parts[0], parts[1], db_path=self.db_path, preferred_lang=lang)
        self._json(account)

    def do_login(self, arg: str) -> None:
        parts = shlex.split(arg)
        if len(parts) < 2:
            print("usage: login <username> <password>")
            return
        auth = authenticate_account(parts[0], parts[1], db_path=self.db_path)
        self._json(auth)

    def do_new_project(self, arg: str) -> None:
        parts = shlex.split(arg)
        if len(parts) < 2:
            print("usage: new_project <owner_username> <project_name> [description]")
            return
        description = parts[2] if len(parts) > 2 else ""
        proj = create_project(parts[0], parts[1], description=description, db_path=self.db_path)
        self._json(proj)

    def do_projects(self, arg: str) -> None:
        owner = arg.strip()
        self._json(list_projects(db_path=self.db_path, owner_username=owner))

    def do_models(self, arg: str) -> None:
        project = arg.strip()
        self._json(list_models(db_path=self.db_path, project_name=project))

    def do_register_model(self, arg: str) -> None:
        parts = shlex.split(arg)
        if len(parts) < 3:
            print("usage: register_model <owner_username> <project_name> <model_name> [checkpoint_path]")
            return
        checkpoint = parts[3] if len(parts) > 3 else ""
        model = register_model(
            owner_username=parts[0],
            project_name=parts[1],
            model_name=parts[2],
            checkpoint_path=checkpoint,
            db_path=self.db_path,
        )
        self._json(model)

    def do_seed_phrases(self, _arg: str) -> None:
        count = seed_common_phrases(db_path=self.db_path)
        self._json({"seeded": count})

    def do_sync_events(self, arg: str) -> None:
        limit = 25
        offline = False
        parts = shlex.split(arg)
        if parts:
            try:
                limit = int(parts[0])
            except ValueError:
                if parts[0].lower() == "offline":
                    offline = True
            if len(parts) > 1 and parts[1].lower() == "offline":
                offline = True
        events = fetch_world_events(max_items=limit, include_network=(not offline))
        inserted = add_events(events, db_path=self.db_path)
        phrase_entries = extract_keyphrases([f"{e.get('title', '')} {e.get('summary', '')}" for e in events], top_k=80)
        bulk_upsert_phrases(phrase_entries, db_path=self.db_path)
        self._json({"inserted_events": inserted, "events_seen": len(events), "phrase_updates": len(phrase_entries)})

    def do_events(self, arg: str) -> None:
        limit = int(arg.strip() or "10")
        self._json(list_events(db_path=self.db_path, limit=limit))

    def do_phrases(self, arg: str) -> None:
        parts = shlex.split(arg)
        if not parts:
            self._json(list_phrases(db_path=self.db_path, limit=30))
            return
        language = parts[0]
        limit = int(parts[1]) if len(parts) > 1 else 30
        self._json(list_phrases(db_path=self.db_path, language=language, limit=limit))

    def do_sim_train(self, arg: str) -> None:
        parts = shlex.split(arg)
        cycles = int(parts[0]) if len(parts) > 0 else 2
        episodes = int(parts[1]) if len(parts) > 1 else 12
        out_dir = parts[2] if len(parts) > 2 else "outputs/console_sim"
        report = train_agent_with_self_play(cycles=cycles, episodes_per_cycle=episodes, out_dir=out_dir)
        self._json(report)

    def do_exit(self, _arg: str) -> bool:
        return True

    def do_quit(self, _arg: str) -> bool:
        return True

    def emptyline(self) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroDSL platform console")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--cmd", default="", help="Run one console command and exit.")
    args = parser.parse_args()

    shell = NeuroConsole(db_path=args.db_path)
    if args.cmd:
        shell.onecmd(args.cmd)
        return 0
    shell.cmdloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
