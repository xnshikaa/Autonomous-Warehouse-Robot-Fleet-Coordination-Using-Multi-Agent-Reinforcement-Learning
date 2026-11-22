"""Verify the running warehouse service with real WebSocket commands."""
from __future__ import annotations

import argparse
import json
from urllib.request import urlopen

from websockets.sync.client import connect


def verify(base_url: str) -> None:
    def get(path):
        with urlopen(f"{base_url}/{path}", timeout=5) as response:
            return json.load(response)

    assert get("health")["ok"]
    assert get("readiness")["ok"]
    with connect(base_url.replace("http:", "ws:", 1) + "/ws", open_timeout=5) as socket:
        def wait(predicate):
            for _ in range(50):
                message = json.loads(socket.recv(timeout=5))
                if predicate(message):
                    return message
            raise AssertionError("The warehouse did not report the expected state")

        def command(kind, predicate, **arguments):
            socket.send(json.dumps({"type": kind, **arguments}))
            return wait(predicate)

        wait(lambda state: state.get("type") == "state_update")
        command("pause", lambda state: state.get("simulationPaused") is True)
        for algorithm in ("QMIX", "IQL", "Rule-Based"):
            command("set_algorithm", lambda state: state.get("algorithm") == algorithm, algorithm=algorithm)
            for size in (5, 10, 20):
                configured = command(
                    "set_fleet_size",
                    lambda state: state.get("policy", {}).get("activeFleetSize") == size,
                    size=size,
                )
                assert configured["policy"]["source"] == algorithm
                assert configured["policy"]["trained"] == (algorithm != "Rule-Based")
                assert len(configured["tasks"]) == len(configured["robots"]) == size
                command("start", lambda state: state.get("simulationPaused") is False)
                running = wait(lambda state: state.get("timestep", 0) > 0 and state.get("algorithm") == algorithm)
                assert all(robot["pathSource"] == "POLICY_TRAJECTORY" for robot in running["robots"])
                assert all(robot["path"][-1] == robot["position"] for robot in running["robots"])
                command("pause", lambda state: state.get("simulationPaused") is True)
                print(f"{algorithm} · {size} robots · step {running['timestep']} verified")

        before = get("state")
        rejection = command(
            "set_algorithm", lambda state: state.get("type") == "command_error", algorithm="unsupported"
        )
        assert rejection["error"]["code"] == "invalid_algorithm"
        after = get("state")
        assert after["policy"] == before["policy"] and after["episode"] == before["episode"]
        command("set_fleet_size", lambda state: len(state.get("robots", ())) == 5, size=5)
        command("set_algorithm", lambda state: state.get("algorithm") == "QMIX", algorithm="QMIX")
        command("reset", lambda state: state.get("simulationPaused") and state.get("timestep") == 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    verify(parser.parse_args().url.rstrip("/"))
