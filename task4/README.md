Task 4 is implemented by enabling profiling inside the Task 2(a), Task 2(b), and Task 3 training code:

- `task2a/run_glue.py --profile_task4`
- `task2b/run_glue.py --profile_task4`
- `task3/run_glue.py --profile_task4`

Use `task4/run_4nodes.sh` to launch the profiled run for one of the three communication methods.
