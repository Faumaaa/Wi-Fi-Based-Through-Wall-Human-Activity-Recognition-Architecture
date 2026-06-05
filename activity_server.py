import asyncio
import websockets
import json
import sys
import re
from datetime import datetime

connected_clients = set()

async def handler(websocket):
    connected_clients.add(websocket)
    print(f"[+] Dashboard connected. Total: {len(connected_clients)}", flush=True)
    try:
        async for _ in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[-] Dashboard disconnected.", flush=True)

async def broadcast(data):
    if connected_clients:
        msg = json.dumps(data)
        await asyncio.gather(*[c.send(msg) for c in connected_clients])

async def run_ml_and_read(ml_script_path, broadcast_fn):
    print(f"🚀 Starting ML model: {ml_script_path}", flush=True)

    process = await asyncio.create_subprocess_exec(
        sys.executable, ml_script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    print("📡 Reading ML model output...", flush=True)

    known = ['walking', 'running', 'sitting', 'standing', 'falling']

    async for line in process.stdout:
        line = line.decode().strip()
        if not line:
            continue

        print(f"[ML] {line}", flush=True)

        # ✅ Parse the MAIN activity line from your Demo_code.py output:
        # Format: "  ✅ ACTIVITY : WALKING ◀ NEW  (83%)"
        # or:     "  ✅ ACTIVITY : WALKING  (83%)"
        if 'ACTIVITY' in line and ':' in line:
            try:
                # Extract activity name (between ':' and next space/special char)
                after_colon = line.split(':', 1)[1].strip()
                # Remove special characters like ◀ NEW
                clean = re.sub(r'[◀✅]', '', after_colon).strip()
                parts = clean.split()

                activity = parts[0].lower().strip()

                # Extract confidence percentage like (83%)
                pct_match = re.search(r'\((\d+)%\)', line)
                if pct_match:
                    confidence = int(pct_match.group(1)) / 100
                else:
                    confidence = 1.0

                if activity in known:
                    payload = {
                        "activity": activity,
                        "confidence": round(confidence, 2),
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    print(f"[→ SENDING] {payload}", flush=True)
                    await broadcast_fn(payload)

            except Exception as e:
                print(f"[PARSE ERROR] {e} — line: {line}", flush=True)

async def main():
    ML_SCRIPT = "Demo_code.py"

    print("🚀 HAR WebSocket Server Starting...", flush=True)

    server = await websockets.serve(handler, "0.0.0.0", 8765)

    print("✅ WebSocket server is LIVE on port 8765", flush=True)

    await asyncio.gather(
        run_ml_and_read(ML_SCRIPT, broadcast),
        server.wait_closed()
    )

asyncio.run(main())