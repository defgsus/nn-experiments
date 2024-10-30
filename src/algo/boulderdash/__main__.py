from src.algo.boulderdash.boulderdash import BoulderDash


def run_boulderdash_console():
    bd = BoulderDash.from_random((16, 16))
    while True:
        print()
        bd.dump(ansi_colors=True)
        cmd = input("\nw/a/s/d> ").lower()
        action = bd.ACTIONS.Nop
        if cmd == "q":
            break
        elif cmd == "r":
            bd = bd.from_random(bd.shape)
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
