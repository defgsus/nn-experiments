

if __name__ == "__main__":
    from src.clipig.app import main
    main()

"""
from src.clipig.clipig_task import ClipigTask
from src.clipig.clipig_worker import ClipigWorker


def main():
    task = ClipigTask({
        "num_iterations": 100,
    })
    # task.run()

    worker = ClipigWorker()
    with worker:
        worker.run_task({"num_iterations": 100})

        for event in worker.events(blocking=True):
            print("EVENT:", event)



if __name__ == "__main__":
    main()
"""
