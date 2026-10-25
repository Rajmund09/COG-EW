import urllib.request
import json

def post(url, data=None):
    if data is None:
        data = {}
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("1. Resetting simulation...")
    s = post("http://127.0.0.1:8000/api/simulation/reset", {"scenario_id": 1})
    print(f"Initial Status: {s['status']}, Altitude: {s['vehicle']['altitude_m']} m")
    print(f"Standoff weapon: {s.get('standoff_weapon')}")

    print("2. Starting simulation...")
    s = post("http://127.0.0.1:8000/api/simulation/start")
    print(f"Started: {s['status']}")

    min_alt = 999999.0
    max_alt = 0.0
    hit_strike_egress = False
    completed = False

    for i in range(120):
        s = post("http://127.0.0.1:8000/api/simulation/tick", {"dt": 0.1})
        alt = float(s["vehicle"]["altitude_m"])
        min_alt = min(min_alt, alt)
        max_alt = max(max_alt, alt)
        status = s["status"]
        if status == "strike_egress":
            hit_strike_egress = True
            sw = s.get("standoff_weapon", {})
            print(f"Tick {i:02d} [STRIKE_EGRESS]: Waverider Alt: {alt:.0f}m, Weapon Alt: {sw.get('altitude_m', 0):.0f}m")
        if status == "completed":
            completed = True
            print(f"Tick {i:02d} [COMPLETED]: Waverider Alt: {alt:.0f}m, Target Destroyed: {s['target']['destroyed']}")
            break

    print(f"\n--- MISSION FLIGHT ENVELOPE SUMMARY ---")
    print(f"Min Altitude: {min_alt:.0f} m")
    print(f"Max Altitude: {max_alt:.0f} m")
    print(f"Hit Standoff Strike & Egress: {hit_strike_egress}")
    print(f"Completed Successfully: {completed}")

    assert min_alt >= 28000.0, f"Error: Waverider dropped too low: {min_alt}m"
    assert hit_strike_egress, "Error: Did not trigger standoff strike mode"
    assert completed, "Error: Did not complete mission"
    assert s["target"]["destroyed"], "Error: Target bunker was not destroyed"
    print("\n>>> ALL TESTS PASSED: 100% AEROSPACE STANDOFF STRIKE & SAFE EGRESS VERIFIED <<<")

if __name__ == "__main__":
    main()
