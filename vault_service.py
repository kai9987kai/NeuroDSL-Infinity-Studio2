import sqlite3
import hashlib
import os
import requests
from typing import List, Tuple, Optional

class VaultService:
    """Manages the user account system and the world knowledge database."""
    
    def __init__(self, db_path="infinity_vault.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Accounts Table
        cursor.execute('''CREATE TABLE IF NOT EXISTS accounts 
                          (username TEXT PRIMARY KEY, password_hash TEXT, projects TEXT)''')
        # Knowledge Table
        cursor.execute('''CREATE TABLE IF NOT EXISTS global_knowledge 
                          (phrase TEXT, language TEXT, context TEXT)''')
        # Photo Vault
        cursor.execute('''CREATE TABLE IF NOT EXISTS photo_vault 
                          (name TEXT PRIMARY KEY, image_blob BLOB, metadata TEXT)''')
        # Seed Knowledge (Global Dict)
        self._seed_dictionary(cursor)
        conn.commit()
        conn.close()

    def _seed_dictionary(self, cursor):
        seeds = [
            ("Neural Network", "en", "Core Architecture"),
            ("Réseau de neurones", "fr", "Architecture de base"),
            ("Neuronales Netz", "de", "Kernarchitektur"),
            ("Red neuronal", "es", "Arquitectura central"),
            ("Rete neurale", "it", "Architettura centrale"),
            ("Intelligence", "en", "Capability"),
            ("Intelligenz", "de", "Fähigkeit")
        ]
        for p, l, c in seeds:
            cursor.execute("INSERT OR IGNORE INTO global_knowledge (phrase, language, context) VALUES (?, ?, ?)", (p, l, c))

    def create_account(self, username, password) -> bool:
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO accounts (username, password_hash, projects) VALUES (?, ?, ?)", 
                           (username, pwd_hash, "[]"))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False

    def login(self, username, password) -> bool:
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM accounts WHERE username=?", (username,))
        row = cursor.fetchone()
        conn.close()
        return row and row[0] == pwd_hash

    def add_knowledge_entry(self, phrase, lang, context):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO global_knowledge (phrase, language, context) VALUES (?, ?, ?)", 
                       (phrase, lang, context))
        conn.commit()
        conn.close()

    def get_knowledge(self, lang="en") -> List[Tuple[str, str]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT phrase, context FROM global_knowledge WHERE language=?", (lang,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def add_photo(self, name, image_bytes, metadata=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO photo_vault (name, image_blob, metadata) VALUES (?, ?, ?)", 
                       (name, sqlite3.Binary(image_bytes), metadata))
        conn.commit()
        conn.close()

    def get_photo(self, name) -> Optional[bytes]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_blob FROM photo_vault WHERE name=?", (name,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

class EventCrawler:
    """Simulates a crawler to fetch world current events."""
    
    @staticmethod
    def fetch_headlines() -> List[str]:
        # Simulated live news fetch
        headlines = [
            "Neural networks achieve 99% accuracy on zero-shot logical reasoning tasks.",
            "Global compute surplus leads to significant drop in decentralized training costs.",
            "New ASI-Safety standards proposed by international research coalition.",
            "Simulation theory gains traction as virtual environment fidelity hits 10^9 real-world parity."
        ]
        return headlines
