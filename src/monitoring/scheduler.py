import time
import schedule
import drifts


def detect_drifts():

    limit = 1000
    input_path = 'artifacts'
    drifts.run(input_path, limit)


schedule.every(1).hour.do(detect_drifts)

if __name__ == "__main__":
    
    while True:
        schedule.run_pending()
        time.sleep(1)
