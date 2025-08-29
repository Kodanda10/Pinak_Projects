from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn, time
from typing import Optional

app = FastAPI(title="Pinak Mock Memory Service", version="1.0")

DB = {
    "episodic": [],
    "procedural": [],
    "rag": [],
    "events": [],
    "session": {},
    "working": {},
}

def _ts(): return int(time.time())

@app.get("/api/v1/memory/health")
def health():
    return {"ok": True, "service": "mock", "port": 8000, "time": _ts()}

@app.post("/api/v1/memory/episodic")
async def add_episodic(item: dict):
    item = dict(item); item["id"] = f"e{len(DB['episodic'])+1}"; item["ts"] = _ts()
    DB["episodic"].append(item)
    DB["events"].append({"type": "add_episodic", "ref": item["id"], "ts": item["ts"]})
    return {"ok": True, "data": item}

@app.get("/api/v1/memory/episodic")
def list_episodic(limit: int = 50):
    return {"ok": True, "data": DB["episodic"][-limit:]}

@app.post("/api/v1/memory/procedural")
async def add_procedural(item: dict):
    item = dict(item); item["id"] = f"p{len(DB['procedural'])+1}"; item["ts"] = _ts()
    DB["procedural"].append(item)
    DB["events"].append({"type": "add_procedural", "ref": item["id"], "ts": item["ts"]})
    return {"ok": True, "data": item}

@app.get("/api/v1/memory/procedural")
def list_procedural(limit: int = 50):
    return {"ok": True, "data": DB["procedural"][-limit:]}

@app.post("/api/v1/memory/rag")
async def add_rag(item: dict):
    item = dict(item); item["id"] = f"r{len(DB['rag'])+1}"; item["ts"] = _ts()
    DB["rag"].append(item)
    DB["events"].append({"type": "add_rag", "ref": item["id"], "ts": item["ts"]})
    return {"ok": True, "data": item}

@app.get("/api/v1/memory/rag")
def list_rag(limit: int = 50):
    return {"ok": True, "data": DB["rag"][-limit:]}

@app.get("/api/v1/memory/events")
def list_events(limit: int = 50, type: Optional[str] = None):
    ev = DB["events"]
    if type: ev = [e for e in ev if e["type"] == type]
    return {"ok": True, "data": ev[-limit:]}

@app.post("/api/v1/memory/session")
async def add_session(item: dict):
    key = item.get("key"); ttl = int(item.get("ttl", 600)); value = item.get("value")
    if not key: raise HTTPException(400, "key required")
    DB["session"][key] = {"value": value, "exp": _ts()+ttl}
    DB["events"].append({"type": "add_session", "ref": key, "ts": _ts()})
    return {"ok": True, "data": {"key": key, "ttl": ttl}}

@app.get("/api/v1/memory/session")
def list_session():
    now = _ts()
    live = {k:v for k,v in DB["session"].items() if v["exp"]>now}
    return {"ok": True, "data": live}

@app.post("/api/v1/memory/working")
async def add_working(item: dict):
    key = item.get("key"); exp_in = int(item.get("expires_in", 600)); value = item.get("value")
    if not key: raise HTTPException(400, "key required")
    DB["working"][key] = {"value": value, "exp": _ts()+exp_in}
    DB["events"].append({"type": "add_working", "ref": key, "ts": _ts()})
    return {"ok": True, "data": {"key": key, "expires_in": exp_in}}

@app.get("/api/v1/memory/working")
def list_working():
    now = _ts()
    live = {k:v for k,v in DB["working"].items() if v["exp"]>now}
    return {"ok": True, "data": live}

@app.get("/api/v1/memory/search")
def search_all_layers(q: str):
    ql = q.lower()
    matches = {"episodic":[], "procedural":[], "rag":[]}
    for it in DB["episodic"]:
        if ql in str(it).lower(): matches["episodic"].append(it)
    for it in DB["procedural"]:
        if ql in str(it).lower(): matches["procedural"].append(it)
    for it in DB["rag"]:
        if ql in str(it).lower(): matches["rag"].append(it)
    return {"ok": True, "data": matches}

@app.middleware("http")
async def auth_mw(request: Request, call_next):
    # Allow all for mock, but record header if present
    request.state.auth = request.headers.get("authorization")
    return await call_next(request)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)