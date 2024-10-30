from src.algo.boulderdash import BoulderDash, BoulderDashGenerator


def run_boulderdash_console():
    generator_kwargs = dict(
        shape=(16, 16),
        ratio_wall=.05,
        ratio_wall_horizontal=0.2,
        ratio_wall_horizontal_gap=0.3,
        ratio_rock=0.05,
        ratio_diamond=.01,
        ratio_sand=0.2,
        with_border=False,
    )
    generator = BoulderDashGenerator()
    bd = generator.create_random(**generator_kwargs)
    while True:
        print()
        bd.dump(ansi_colors=True)
        cmd = input("\nw/a/s/d> ").lower()
        action = bd.ACTIONS.Nop
        if cmd == "q":
            break
        elif cmd == "r":
            bd = generator.create_random(**generator_kwargs)
            continue
        elif cmd == "w":
            action = bd.ACTIONS.Up
        elif cmd == "a":
            action = bd.ACTIONS.Left
        elif cmd == "s":
            action = bd.ACTIONS.Down
        elif cmd == "d":
            action = bd.ACTIONS.Right

        result1 = bd.apply_action(action)
        result2 = bd.step()

        for key, value in bd.RESULTS.__dict__.items():
            if value == result1:
                r1 = key
            if value == result2:
                r2 = key
        print(f"result: {r1}, {r2}")
        if result2 == bd.RESULTS.PlayerDied:
            print("HIT AND DIED!!")



if __name__ == "__main__":

    run_boulderdash_console()
