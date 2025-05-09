import sqlite3

def init_db():
    conn = sqlite3.connect('predicciones.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emocion TEXT NOT NULL,
            fecha TEXT NOT NULL,
            tipo_animal TEXT
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()