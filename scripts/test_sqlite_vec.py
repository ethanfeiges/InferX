"""Check that the ARM64 sqlite-vec build loads and returns correct nearest neighbors."""
import platform
import sqlite3
import struct
from pathlib import Path

ext = Path(__file__).resolve().parent.parent / "third_party" / "sqlite-vec" / "vec0.dll"
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
db.load_extension(str(ext))
db.enable_load_extension(False)
print("machine:", platform.machine())
print("vec_version:", db.execute("select vec_version()").fetchone()[0])
print(db.execute("select vec_debug()").fetchone()[0].replace("\n", " | "))


def pack(values):
    return struct.pack(f"{len(values)}f", *values)


db.execute("create virtual table v using vec0(id integer primary key, e float[4])")
vectors = {1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0.9, 0.1, 0, 0], 4: [0, 0, 0, 1]}
for key, vector in vectors.items():
    db.execute("insert into v(id, e) values (?, ?)", (key, pack(vector)))
rows = db.execute(
    "select id, distance from v where e match ? and k = 3 order by distance",
    (pack([1, 0, 0, 0]),),
).fetchall()
print("rows:", rows)
assert [row[0] for row in rows[:2]] == [1, 3], "wrong nearest neighbors"
assert abs(rows[0][1]) < 1e-6, "wrong distance"
print("PASS")
