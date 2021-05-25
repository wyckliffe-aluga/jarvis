
from Worker import Worker
import itertools
import concurrent.futures
import multiprocessing
import os

total_steps = 12
n_workers = 4  # multiprocessing.cpu_count() # choose number of workers here

global_counter = itertools.count()
solve_time_list = []


workers = [Worker(global_counter, solve_time_list, total_steps,
                  COCO_doc_path=os.path.join(os.getcwd(), "LuybenExamplePart.fsd")) for _ in range(n_workers)]
run_worker = lambda worker_: worker_.run()

with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
    executor.map(run_worker, workers, timeout=60)

print(f"Average solve time for {n_workers} workers is {sum(solve_time_list)/len(solve_time_list)} seconds")