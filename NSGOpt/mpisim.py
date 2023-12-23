if __name__ == '__main__':
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
        #future = executor.submit(abs, -42)
        #assert future.result() == 42
        #answer = set(executor.map(abs, [-42, 42]))
        #assert answer == {42}
            import D1PatchSample2opt_vary_chans