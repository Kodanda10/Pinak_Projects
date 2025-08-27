#!/usr/bin/env python3
import os, json, time
import requests

BASE = os.getenv('PINAK_MEMORY_URL', 'http://localhost:8001')
PID = os.getenv('PINAK_PROJECT_ID', 'Pnk-Demo')
TOKEN = os.getenv('PINAK_TOKEN')

headers = {'X-Pinak-Project': PID}
if TOKEN:
    headers['Authorization'] = f'Bearer {TOKEN}'

def post(path, payload, extra_headers=None):
    h = dict(headers)
    if extra_headers:
        h.update(extra_headers)
    r = requests.post(f"{BASE}/api/v1/memory/{path}", json=payload, headers=h, timeout=10)
    print(path, r.status_code, r.text[:120])
    return r

def get(path, params=None):
    r = requests.get(f"{BASE}/api/v1/memory/{path}", params=params, headers=headers, timeout=10)
    print(path, r.status_code)
    return r

def main():
    # semantic
    post('add', {"content":"demo: semantic memory","tags":["demo"]}, None)
    # episodic/procedural/rag
    post('episodic/add', {"content":"demo: episodic","salience":2})
    post('procedural/add', {"skill_id":"build","steps":["a","b"]})
    post('rag/add', {"query":"faq","external_source":"kb"})
    # session/working
    sid='sess-demo'
    post('session/add', {"session_id":sid, "content":"user prompt"})
    post('session/add', {"session_id":sid, "content":"agent response"})
    post('working/add', {"content":"scratch","ttl_seconds":60})
    # event + governance mirror
    post('event', {"type":"gov_audit","path":"/demo","method":"POST","status":200,"role":"editor"})

    # reads
    print('search:', get('search', {"query":"semantic"}).text)
    print('search_v2:', get('search_v2', {"query":"demo","layers":"semantic,episodic,procedural,rag"}).text[:200])
    print('events:', get('events').text[:200])
    print('changelog:', get('changelog').text[:200])
    print('session list:', get('session/list', {"session_id": sid}).text[:200])
    print('working list:', get('working/list').text[:200])
    # metrics
    try:
        m = requests.get(f"{BASE}/metrics", timeout=5)
        print('metrics head:', m.text.splitlines()[:5])
    except Exception as e:
        print('metrics unavailable:', e)

if __name__ == '__main__':
    main()

