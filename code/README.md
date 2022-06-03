# Code snippets to learn the parallel  API in OpenFOAM

First, take a look at `testParallel/testParallel.C` and change the functions which get called in `main` to your liking.

To run some of the benchmarks, you need to make sure MPI has enough space in its buffers to fit the messages. Use the following
command to run the code:

```bash
export MPI_BUFFER_SIZE=90000000
wmake && cd pitzDaily/ && foamCleanTutorials . && ./Allrun &&  cat log.testParallel && cd -
```
