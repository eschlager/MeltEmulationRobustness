
# -*- coding: utf-8 -*-
"""
@author: eschlager
utils for parallel data loading with pytorch DataLoader
"""
import os
import logging


class AffinityInitializer:
    def __init__(self, base_offset=0, cores_per_worker=1, name=''):
        self.base_offset = base_offset
        self.cores_per_worker = cores_per_worker
        self.name = name

    def __call__(self, worker_id):
        affinity_worker_init_fn(worker_id, self.base_offset, self.cores_per_worker, self.name)

def affinity_worker_init_fn(worker_id, base_offset=0, cores_per_worker=1, name=''):
    """
    Function to set CPU affinity for DataLoader workers.
    This is meant to be called by a wrapper function.
    """
    start_core = base_offset + worker_id * cores_per_worker
    end_core = start_core + cores_per_worker
    core_ids = list(range(start_core, end_core))
    try:
        os.sched_setaffinity(0, core_ids)
        print(f"[{name} worker {worker_id}] pinned to cores: {core_ids}")
    except AttributeError:
        print(f"Affinity not supported on this platform.")


def shutdown_dataloader(dl):
    try:
        it = getattr(dl, '_iterator', None)
        if it is not None:
            if hasattr(it, '_shutdown_workers'):
                try:
                    it._shutdown_workers()
                except Exception:
                    logging.exception("iterator._shutdown_workers failed")
                    pass
            try:
                dl._iterator = None
            except Exception:
                pass
        
        if hasattr(dl, "_shutdown_workers"):
            try:
                dl._shutdown_workers()
            except Exception:
                pass

    except Exception:
        logging.exception("shutdown dataloader failed")
    try:
        del dl
    except Exception:
        pass