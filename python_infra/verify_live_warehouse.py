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
            for _ in range(150):
                message = json.loads(socket.recv(timeout=5))
                if predicate(message):
                    return message
            raise AssertionError("The warehouse did not report the expected state")

        def command(kind, predicate, **arguments):
            socket.send(json.dumps({"type": kind, **arguments}))
            return wait(predicate)

        wait(lambda state: state.get("type") == "state_update")
        command("pause", lambda state: state.get("simulationPaused") is True)
        command("set_speed", lambda state: state.get("config", {}).get("speedMultiplier") == 4, speed=4)
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
                layout = configured["layout"]
                legal = {tuple(cell) for cell in layout["walkable_cells"]} - {
                    tuple(cell) for cell in layout["human_corridors"]["cells"]
                }
                previous = configured
                while previous["timestep"] < 20 and not previous["simulationFinished"]:
                    running = wait(lambda state: state.get("timestep", 0) > previous["timestep"])
                    positions = {tuple(robot["position"]) for robot in running["robots"]}
                    assert len(positions) == size and positions <= legal
                    assert running["metrics"]["physicalCollisions"] == 0
                    assert all(robot["pathSource"] == "POLICY_TRAJECTORY" for robot in running["robots"])
                    assert all(robot["path"][-1] == robot["position"] for robot in running["robots"])
                    if running["timestep"] == previous["timestep"] + 1:
                        old_occupied = {tuple(robot["position"]) for robot in previous["robots"]}
                        for old, new in zip(previous["robots"], running["robots"], strict=True):
                            start, end = tuple(old["position"]), tuple(new["position"])
                            if start != end:
                                assert end not in old_occupied
                                assert sum(abs(a - b) for a, b in zip(start, end)) == 1
                    previous = running
                command("pause", lambda state: state.get("simulationPaused") is True)
                print(f"{algorithm} · {size} robots · step {previous['timestep']} verified")

        before = get("state")
        rejection = command(
            "set_algorithm", lambda state: state.get("type") == "command_error", algorithm="unsupported"
        )
        assert rejection["error"]["code"] == "invalid_algorithm"
        after = get("state")
        assert after["policy"] == before["policy"] and after["episode"] == before["episode"]
        mission = command("set_fleet_size", lambda state: len(state.get("robots", ())) == 1, size=1)
        assert mission["policy"]["source"] == "Rule-Based"
        previous_episode = mission["episode"]
        command("start", lambda state: state.get("simulationPaused") is False)
        final = wait(lambda state: state.get("simulationFinished") is True)
        assert final["metrics"]["physicalCollisions"] == final["metrics"]["blockedMoves"] == 0
        assert final["metrics"]["pickups"] == final["metrics"]["completedTasks"] == 1
        assert final["robots"][0]["status"] == "IDLE"
        task = final["tasks"][0]
        assert task["pickupTimestep"] < task["completedTimestep"]
        assert final["robots"][0]["position"] == task["deliveryPos"]
        command("reset", lambda state: state.get("episode", 0) > previous_episode and state["timestep"] == 0)
        print(f"Mission seed {mission['missionSeed']} · pickup {task['pickupTimestep']} · delivery {task['completedTimestep']}")
        command("set_fleet_size", lambda state: len(state.get("robots", ())) == 5, size=5)
        for algorithm, seed in (("QMIX", 310011), ("IQL", 310001), ("Rule-Based", 310011)):
            command("set_algorithm", lambda state: state.get("algorithm") == algorithm, algorithm=algorithm)
            command("load_reference", lambda state: state.get("referenceSeed") == seed and state["timestep"] == 0)
            command("start", lambda state: state.get("simulationPaused") is False)
            final = wait(lambda state: state.get("simulationFinished") is True)
            assert final["metrics"]["physicalCollisions"] == 0
            assert final["metrics"]["pickups"] == final["metrics"]["completedTasks"] == 5
            assert all(task["pickupTimestep"] < task["completedTimestep"] for task in final["tasks"])
            print(f"{algorithm} reference {seed} · 5 pickups and deliveries · step {final['timestep']}")
        command("set_fleet_size", lambda state: len(state.get("robots", ())) == 5, size=5)
        command("set_algorithm", lambda state: state.get("algorithm") == "QMIX", algorithm="QMIX")
        command("reset", lambda state: state.get("simulationPaused") and state.get("timestep") == 0)
        command("set_speed", lambda state: state.get("config", {}).get("speedMultiplier") == 1, speed=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    verify(parser.parse_args().url.rstrip("/"))
