# Citations

## Spatial algebra

mantis's traced types (`Motion`, `Force`, `Inertia`, `SE3`) and its algorithms (`rnea`, `crba`,
`aba`) follow Featherstone's 6D spatial vector algebra.

```bibtex
@book{featherstone2008rbda,
  author    = {Roy Featherstone},
  title     = {Rigid Body Dynamics Algorithms},
  publisher = {Springer},
  address   = {New York},
  year      = {2008},
  doi       = {10.1007/978-1-4899-7560-7}
}
```

## Parallel performance analysis

`parallel_analysis.py` measures traced programs with Grama et al.'s metrics: work and span,
speedup, efficiency, cost optimality, overhead and isoefficiency.

```bibtex
@book{grama2003parallel,
  author    = {Ananth Grama and Anshul Gupta and George Karypis and Vipin Kumar},
  title     = {Introduction to Parallel Computing},
  edition   = {2},
  publisher = {Addison-Wesley},
  address   = {Harlow, England},
  year      = {2003},
  isbn      = {978-0-201-64865-2}
}
```

## Register and shared-memory pressure

The pressure work (thread-tier scratch reuse, a shared-memory budget in `schedule.build`, and
recomputation in exchange for storage) draws on these. Where a DOI couldn't be verified, it's
omitted and a URL is given.

- **Ordering a computation to minimise live values.** Sethi-Ullman numbering gives the evaluation
  order of an expression tree that needs the fewest registers. It is the base heuristic for
  ordering statements inside a chunk, and for ranking tasks inside a stage, by peak live values.
- **Scheduling and register allocation solved together.** Goodman and Hsu showed that scheduling
  first and allocating afterwards wastes registers in large basic blocks, which a straight-line
  trace is. Their integrated scheduler switches to minimising pressure when live values near the
  limit. Berson, Gupta and Soffa's register reuse DAGs (URSA) detect excess demand and decide
  where to split. This is the model for making `schedule.build` respect a shared-memory budget
  instead of colouring whatever lifetimes the speed-first placement produced.
- **Assigning slots.** Chaitin's graph colouring decides when a value must spill. Poletto and
  Sarkar's linear scan does interval allocation over a linear order, which is what mantis's slot
  colouring (step 5) and the thread tier's chunk scratch are.
- **Recomputing instead of storing.** Briggs, Cooper and Torczon's rematerialization recomputes a
  value whose inputs are still available instead of spilling it: the "scalarise more" lever.
  Griewank and Walther's revolve gives provably optimal checkpointing schedules for trading
  memory against recomputation in reverse-mode AD (`mantis.ad`). Halide frames the same
  storage-vs-recompute tradeoff as part of a schedule.
- **Lower bounds on memory traffic.** Hong and Kung's red-blue pebble game bounds the traffic
  between fast and slow memory for a computation DAG with limited fast memory. It is the model
  for "registers vs shared vs local memory" on a traced DAG.
- **The GPU side.** Gebhart et al. measured that on-chip registers, shared memory and cache trade
  against each other per kernel (up to 71% faster with a unified split). Volkov's thesis shows
  occupancy and latency hiding are not monotone in either, so fitting a budget has to be timed
  (`tune.py`), not assumed to help.

```bibtex
@article{sethi1970optimal,
  author  = {Ravi Sethi and Jeffrey D. Ullman},
  title   = {The Generation of Optimal Code for Arithmetic Expressions},
  journal = {Journal of the ACM},
  volume  = {17}, number = {4}, pages = {715--728}, year = {1970},
  doi     = {10.1145/321607.321620}
}
@inproceedings{goodman1988scheduling,
  author    = {James R. Goodman and Wei-Chung Hsu},
  title     = {Code Scheduling and Register Allocation in Large Basic Blocks},
  booktitle = {Proceedings of the 2nd International Conference on Supercomputing (ICS)},
  pages     = {442--452}, year = {1988},
  doi       = {10.1145/55364.55407}
}
@inproceedings{berson1998integrated,
  author    = {David A. Berson and Rajiv Gupta and Mary Lou Soffa},
  title     = {Integrated Instruction Scheduling and Register Allocation Techniques},
  booktitle = {Languages and Compilers for Parallel Computing (LCPC)},
  year      = {1998},
  note      = {https://cs.virginia.edu/~soffa/Soffa_Pubs_all/Conferences/Integrated.Berson.1998.pdf}
}
@inproceedings{chaitin1982coloring,
  author    = {Gregory J. Chaitin},
  title     = {Register Allocation \& Spilling via Graph Coloring},
  booktitle = {Proceedings of the SIGPLAN Symposium on Compiler Construction},
  pages     = {98--105}, year = {1982},
  note      = {SIGPLAN Notices 17(6); DOI not verified}
}
@article{poletto1999linear,
  author  = {Massimiliano Poletto and Vivek Sarkar},
  title   = {Linear Scan Register Allocation},
  journal = {ACM Transactions on Programming Languages and Systems},
  volume  = {21}, number = {5}, pages = {895--913}, year = {1999},
  note    = {DOI not verified}
}
@inproceedings{briggs1992remat,
  author    = {Preston Briggs and Keith D. Cooper and Linda Torczon},
  title     = {Rematerialization},
  booktitle = {Proceedings of the ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)},
  pages     = {311--321}, year = {1992},
  doi       = {10.1145/143095.143143}
}
@article{griewank2000revolve,
  author  = {Andreas Griewank and Andrea Walther},
  title   = {Algorithm 799: Revolve: An Implementation of Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {26}, number = {1}, pages = {19--45}, year = {2000},
  doi     = {10.1145/347837.347846}
}
@inproceedings{ragankelley2013halide,
  author    = {Jonathan Ragan-Kelley and Connelly Barnes and Andrew Adams and Sylvain Paris and Fr{\'e}do Durand and Saman Amarasinghe},
  title     = {Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines},
  booktitle = {Proceedings of the ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)},
  year      = {2013},
  note      = {https://people.csail.mit.edu/jrk/halide-pldi13.pdf; DOI not verified}
}
@inproceedings{hong1981pebble,
  author    = {Jia-Wei Hong and H. T. Kung},
  title     = {I/O Complexity: The Red-Blue Pebble Game},
  booktitle = {Proceedings of the 13th ACM Symposium on Theory of Computing (STOC)},
  pages     = {326--333}, year = {1981},
  note      = {DOI not verified}
}
@inproceedings{gebhart2012unifying,
  author    = {Mark Gebhart and Stephen W. Keckler and Brucek Khailany and Ronny Krashinsky and William J. Dally},
  title     = {Unifying Primary Cache, Scratch, and Register File Memories in a Throughput Processor},
  booktitle = {45th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  pages     = {96--106}, year = {2012},
  doi       = {10.1109/MICRO.2012.18}
}
@techreport{volkov2016latency,
  author      = {Vasily Volkov},
  title       = {Understanding Latency Hiding on GPUs},
  institution = {EECS Department, University of California, Berkeley},
  number      = {UCB/EECS-2016-143}, year = {2016},
  note        = {PhD thesis; https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html}
}
```
