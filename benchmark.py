import time
from src.ai.environment import WarehouseEnvironment

print("Simulation performance benchmark")
print("=" * 45)

for n in [5, 10, 20]:
    env = WarehouseEnvironment(num_robots=n, max_steps=2000)
    env.reset()

    for i in range(n):
        env.add_task(
            task_id=i,
            pickup_position=(10, 10),
            delivery_position=(15, 15),
            assigned_robot=i,
        )

    actions = {i: 0 for i in range(n)}

    start = time.perf_counter()

    for _ in range(1000):
        env.step(actions)

    elapsed = time.perf_counter() - start

    print(
        f"{n} robots: {elapsed:.4f}s "
        f"({1000 / elapsed:.1f} steps/sec)"
    )
