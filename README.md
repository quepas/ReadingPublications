# Reading publications

## A Survey on Compiler Autotuning using Machine Learning (2018)

* __Ref__: Ashouri2018
* __Authors__: Amir H. Ashouri, William Killian, John Cavazos, Gianluca Palermo, Cristina Silvano
* __DOI__: https://arxiv.org/abs/1801.04405

An amazing survey of over 200 articles on the use of machine-learning to selecting which compiler optimizations to apply (optimization selection) and at which order (phase-ordering).
Machine-learning allows for smart exploration of huge search spaces that both of these problems have.

The survey starts with a description of three groups of program characteristics that could be used for machine-learning algorithms as data features:

* Static — program properties obtained only from reading the source code, these include scalar features (e.g. number of memory accesses), as well as graph-based (i.e. encoding control-flow, data-flow)
* Dynamic — program properties obtained from the execution of the program, e.g. hardware performance counters, execution time; architecture dependent and independent.
* Hybrid — the mix of above properties.

Next, the paper describes two methods for dimensionality reduction: Principal Component Analysis (PCA) and Exploratory Factor Analysis (EFA).
Both of these methods reduce the number of required and used program features.
Thus, decreasing the time to learn machine-learning models.

This is followed by an extensive description of learning types used in the considered literature: supervised, unsupervised, and reinforcement, with corresponding machine-learning algorithms.
Moreover, the paper groups literature by the performed prediction types and tasks into five groups

* Clustering and downsampling
* Speedup prediction
* Compiler sequence prediction
* Tournament and intermediate prediction
* Feature prediction.

Next, the survey list three techniques for optimization space exploration, namely: adaptive, iterative, and non-interative compilation.
Moreover, a classification of literature based on the target platform (embedded, desktop, HPC) and compiler (GCC, LLVM, Intel-ICC, JIT Compiler, Java Compiler, Polyhedral Model) is shown.
The paper concludes with the list of the most influential papers divided by the following topics, such as: introducing learning methods, genetic algorithms, iterative compilation, dynamic and hybrid features and others.

## A Survey of Machine Learning for Big Code and Naturalness (2018)

* __Ref__: Allamanis2018
* __Authors__: Miltiadis Allamanis, Earl T. Barr, Premkumar Devanbu, Charles Sutton
* __DOI__: https://arxiv.org/abs/1709.06182

The naturalness hypothesis states that programming languages are a form of a bimodal communication with humans and machines, and as such, they display similarities to natural languages.
Therefore, the huge amount of available source codes (the "big code") can be treated with machine-learning techniques known from natural language processing field (NLP) to improve software engineering tools (e.g. better auto-completion, automatic variable naming, automatic generation of a documentation).

The paper starts with a comparison of the natural text and the code.
For example, the code is executable, formally defined, whereas the natural code is more flexible, with less restrictive semantics (e.g. a natural text with missing words might still be understandable).

Next, the paper introduces probabilistic models of code (similar to models from NLP) with applications.

__Code-generation Models__ are described with a probability distribution `P(ℂ|Ctx(ℂ))`, where `ℂ` is a code representation and `Ctx(ℂ)` is a context used for computing probability of `ℂ`.
Depending on the type of context `Ctx(ℂ)`, the model performs a different function:

* When `Ctx(ℂ)` is empty then `P` is a _language model_ (e.g. generating a code).
* When `Ctx(ℂ)` is a non-code modality, then `P` is a _code-generative multimodal model_ (e.g. generating a code from the natural language description)
* When `Ctx(ℂ)` is a code, then `P` is a _transducer model_ of code (e.g. code migration).

_Code-generating models_ divide into _token-level models_ (sequences), _syntactic models_ (trees), _semantic models_ (graphs) with applications such as code migration, code completion, enforcing coding conventions, program synthesis and analysis etc.

__Representational models__ of a source code define a probabilistic model as `P(π|f(ℂ))` where `π` is some code property, and function `f` transforms the code `ℂ` into a target representation (distributed representation or structured prediction).
The possible applications for this model are code search, code summarization, generation of commit messages etc.

__Pattern mining models__ aim to discover patterns in the source code (e.g. API mining, defect prediction, clone detection).
The probabilistic model is defined as follows:

`P(f(ℂ)) = Σ over L: P(g(ℂ)|L)*P(L)`

Where `g` is a deterministic function returning a (partial) view of the code, and `L` is a set of latent variables that the model aims to infer.

The paper surveys almost 200 papers and finishes with a description of open challenges and possible research directions.

## Analysis of Benchmark Characteristics and Benchmark Performance Prediction (1996)

* __Ref__: Saavedra1996
* __Authors__: Rafael H. Saavedra, Alan J. Smith
* __DOI__: https://doi.org/10.1145/235543.235545

The first part of the paper introduces a model for predicting program execution time.
The second part shows how to characterize benchmarks and compute the similarity between them.

### The model

The presented model predicts an execution time `T` of a program `A` (containing abstract operations `AbOp` e.g. addition, division) on a machine `M` using linear equation (dot product):

`T(A,M) = C(A)•P(M)`

Where `C(A)` is a vector of operations frequencies and `P(M)` is a vector of execution times of each operation.
A special benchmark called _machine characterizer_ creates `P(M)` by measuring execution times of abstract operations `AbOp`.
A _program analyzer_ first counts the occurrence of each abstract operation in the code (static analysis) and then measures how many times each code line was executed (dynamic analysis) per basic block:

`C(i) = Σ over j=[1,m]: d(i,j)`

Where `C(i)` is a frequency of an operation `i`, and `d(i,j)` is a number of operations `i` in a block `j`.

Next, authors extend the model to account for cache effects:

`T(A,M) = C(A)•P(M) + M(A,M)•S(M)`

Where `M(A,M)` is a vector of cache misses on each cache level for program `A` and `S(M)` contains stall time penalties for the machine `M`.
In order to account for other optimisations, the model uses modified timings for abstract operations `P(M,O)` where `O` is an optimized program.
This approach is not able to encode all possible execution contexts and optimisations.
For example, vectorisation depends on three additional parameters: the startup delay (an initial overhead), the maximum vector length, and the asymptotic latency.

Nevertheless, the model prediction error is lower than 30% for 229 out of 244 Fortran benchmarks (SPEC, Perfect Club and other small programs).
Moreover, model limitations when dealing with optimisations are tackled in authors' other papers.

### Benchmarks characterisation and similarity

The paper presents the characteristics of two benchmark suites (SPEC, Perfect Club) and a few other programs.
For example, the most used language constructs in considered benchmarks are:

* Statements: assignments (~61%), do loops (~26%)
* Types: single fp (~35%), double fp (~36%), integers (~27%)
* Operations: add/subtract (~52%), multiply (~33%), comparisons (~13%)
* Arithmetic: scalar (~50%), 1D array (~30%), 2D array (~15%)

Even complicated benchmarks spend the majority of time in an execution path with a small number of basic blocks (even 1-5 basic block can represent on average 60-70% of the program).
This observation leads to the definition of the _benchmark instability using skewness_, which measures how much program execution time is concentrated around a small amount of basic blocks or abstract operations.
The experimental results show that some benchmarks are highly skewed with up to 99% of execution time spent in just one basic block (e.g. benchmark MATRIX300).
However, the further analysis shows no correlation between the skewness of the benchmark and the prediction error of the model.

Finally, the paper introduces two metrics of similarity between benchmark programs.
The first metric compares the amount and kind of executed operations (similar operations should yield a similar performance), and the second metric compares the program execution times.
Using similarity between benchmarks it is possible to create clusters of the most similar benchmarks.
These clusters can be sampled in order to obtain _minimal benchmarking set_, which is a minimal collection of the most representative benchmarks (that test different performance aspects).
Moreover, the end of the paper introduces the concept of _benchmark equivalence_, when the ratio of execution time between two benchmarks is similar on two different machines.

## Compile-Time Based Performance Prediction (2000)

* __Ref__: Cascaval2000
* __Authors__: Calin Cascaval, Luiz DeRose, David A. Padua, Daniel A. Reed
* __DOI__: https://doi.org/10.1007/3-540-44905-1_23

The paper presents a prediction model of the execution time for scientific programs (programmed in Fortran).
The model consists of symbolic expressions describing the performance of program constructs with a size of data as free variables.
In case of insufficient information (e.g. size of data, loop bounds, branch frequencies) an additional profiling can be performed.
The introduced _Performance Prediction Model_ looks as follows:

`T(total) = T(CPU) + T(MEM) + T(COMM) + T(I/O)`

In the paper, only CPU and memory parts are considered.
CPU sub-model estimates the time used by the processor by counting the number of operations multiplied by the operation cost.
Operations with similar performance characteristics (e.g. addition and multiplication) are grouped together:

`T(CPU) = CycleTime * ForEachGroup_i(k_i * C_i)`

Where `k_i` is a number of operations from `i`-group, and `C_i` represents single operation cost from this group.

In order to better predict the performance of an optimising compiler, the model:

* Eliminates loop invariants — very common optimisation
* Considers only floating-point operations — the most common type of operations in the scientific computing
* Ignores scalar references — they easily fit into cache memories/registers.

The _Memory Hierarchy Model_ which approximate the performance of CPU cache memories (on different levels) is an extension to the CPU sub-model:

`T(MEM) = CycleTime * ForEachLevel_i(M_i * C_i)`

Where `M_i` is the number of cache misses on `i`-level and `C_i` is the penalty for a miss on that level.
Depending on the amount of information two cache miss models are used:

* Stack Distance Model (more accurate)
* Indirect Direct Model

The experimental results show that a minority of benchmarks requires additional code profiling.
The results for SpLib benchmark show an average of 25% error for execution time and cache miss prediction.
For other benchmarks and CPU models, the average error is around 20%.

## MaJIC: A Matlab Just-In-Time Compiler (2001)

* __Ref__: Almasi2001
* __Authors__: George Almasi, David A. Padua
* __DOI__: https://doi.org/10.1007/3-540-45574-4_5
* __Website__: http://polaris.cs.uiuc.edu/majic/majic.html

MaJIC is an interpreter of MATLAB language with just-in-time (JIT) compilation.
The JIT compilation is performed during the program execution, hence the compilation process has more information about the program.
The lack of information about some code parts is particularly visible in MATLAB, where variables are dynamic and their type and shape can easily change during the program execution.

The MaJIC interpreter contains three modules.
An **analyzer** performs analysis in a just-in-time manner, before the execution of a code portion under analysis.
The type inference is performed over a finite lattice of three components `𝕃(𝐢) ⨯ 𝕃(𝐬) ⨯ 𝕃(𝐫)`: intrinsic type, shape and value range.
The analysis also inspects if a symbol points to a variable, a built-in or an user-defined function.
Resolving the kind of symbol is expressed as a data-flow problem known as _reaching definition_ with modifications.
Hence, each reference to a variable has assigned a set of possible definitions called use-definition chain (definitions come from all available execution paths).
If each use of the symbol has a matching definition then the symbol points to a variable.

A **code generator** selects a specialized implementation of a function or data structure to insert in the target code.
The information used for making this decision is stored as annotations of nodes in the abstract-syntax tree (AST represents the program code).
Moreover, the generator performs four non-trivial tasks:

1) Pre-allocates arrays for intermediate computations e.g. `a*b*c` requires to store the result of `a*b` before multiplying it by `c`
2) Transforms expressions to high-level constructs e.g. `a*x+b'*c` to `dgemv`
3) Allocates 10% more of memory for arrays that resize dynamically
4) Unrolls short loops and vector operations on small matrices.

Finally, a **code repository** manages many versions of the same code with different e.g. annotations, types (from many program executions).
More importantly, the repository tracks and refines codes used on subsequent analyses similar to _gradual typing_.

The interpreter test covers 12 benchmarks with fixed-size problems (in order to execute for around 1 minute each).
Apart from a classical comparison of execution time of MaJIC and FALCON compiler, the paper tests how much dynamic analysis (JIT) improves the accuracy and increases the information about a program over static analysis on a task of finding scalar variables in the program.
Finally, the paper shows how much of the total execution time of the program is spent in type inference, kind analysis, and code generation phases.

### Comments

The work was done before an introduction of JIT compiler to MATLAB interpreter.
MaJIC was planned to be integrated with MATLAB vectorizer developed by Vijay Menon et al.

## Menhir: An Environment for High Performance Matlab (1998/1999)

* __Ref__: Chauveau1998/Chauveau1999 (journal _"extension"_)
* __Authors__: Stéphane Chauveau, François Bodin
* __DOI__: https://doi.org/10.1007/3-540-49530-4_3,
https://doi.org/10.1155/1999/525690
* __Website__: http://www.irisa.fr/caps/PROJECTS/Menhir/menhir/lcr98/

Menhir is a retargetable compiler translating MATLAB code into C or Fortran.
The compiler contains Menhir's target system description (MTSD) that characterizes properties of the target system (generated language, sequential or parallel code, implementation of data structures, memory management, implementation of MATLAB operators and functions).
The description language allows defining e.g. a new type of an upper triangular matrix.
Using the type analysis, Menhir propagates the new type through the program.
When a function is called in MATLAB with an argument of the upper triangular matrix type, Menhir generates in the target language the call to a specialized library that works explicitly on upper triangular matrices.

The compiler shows performance gain on various benchmarks, tested on a single processor and a parallel machines against various old MATLAB to C compilers.
However, Menhir is not an automatic tool, because benchmarks had to be annotated with ~20 directives missing information (e.g. variable shape).

## Rapidly Selecting Good Compiler Optimizations using Performance Counters (2007)

* __Ref__: Cavazos2007
* __Authors__: John Cavazos, Grigori Fursin, Felix Agakov, Edwin Bonilla, Michael F.P. O'Boyle, Olivier Temam
* __DOI__: https://doi.org/10.1109/CGO.2007.32

The paper describes how to build a predictive model for optimisation selection while representing a source code with its performance characteristics i.e. hardware performance counters.
Dynamic features describe the behaviour of a program on the specific architecture (by counting e.g. cache misses, floating-point operations) without the knowledge of the programming language syntax, as opposed to static features.
This is even more important for programs with a control-flow which are hard to describe using static features.

In the research, for each benchmark program (SPEC INT, SPEC FP, MiBench, Polyhedron), all available performance counters are collected using PAPI/PAPIex.
Hardware limitations of performance counters acqusision are bypassed using multiplexing.
The model is learned from empirical data in a form of performance characteristics.
Thus, the model has no prior human bias in a form of e.g. priority for a specific performance counters.
Benchmark programs differ in length, complexity and execution time, hence, recorded performance counters are normalized by event _TOT_INS_, the number of executed instructions.
The predicted class of the model is a binary vector with a status on/off for `N` optimisations (without phase-ordering).

The results show an improvement over using static features and different search space traversal techniques, namely  _random search (RAND)_, and _combined elimination (CE)_ which elimnate in each iteration optimisations that degrade performance.
Whereas CE requires on average 609 executions, the proposed method requires no more than 25 executions to obtain similar or better results.
Moreover, the paper shows which performance counters are crucial for predicting good optimisations by performing importance analysis using _mutual infromation_.

## Reasoning About Time in Higher-Level Language Software (1989)

* __Ref__: Shaw1989
* __Author__: Alan C. Shaw
* __DOI__: https://doi.org/10.1109/32.29487

The paper introduced a novel methodology for predicting and proving time bounds in an idealized high-level language using assertions from Hoare logic.
The methodology distinguishes real-time from computer time and includes timing formulas and reasoning steps for expressions, assignments, function calls, sequence of statements, conditional if-else and case constructs, infinite and while-do loops.
Moreover, the paper considers time constructs such as delays, timeouts and time stamps.
Furthermore, the paper describes synchronization and communication constructs.

The method doesn't account for:
* Compiler optimisations (e.g. common subexpression elimination or register allocation)
* Hardware features (e.g. caching, instruction look-ahead)
* Creation of dynamic processes

As stated in the paper, an analysis of the methodology to obtain good deterministic time bounds on a real programming language, run-time system, and target architecture is necessary.

## The Structure and Performance of Efficient Interpreters (2003)

* __Ref__: Ertl2003
* __Authors__: M. Anton Ertl, David Gregg
* __WWW__: https://www.jilp.org/vol5/v5paper12.pdf

The paper analyses how the mispredict penalty of indirect branches affects performance of interpreters.
The penalty occurs when a brench was incorrectly predicated as a next branch and was partially executed up to some point in the command pipeline (the longer pipeline, the longer penalty takes).
These instructions are closely connected to the fetch, dispatch, and execute cycle common in interpreters and virtual machines.

The experiments contain tests of different branch predictors over various interpreters and benchmarks.

It is worth noticing that since year 2003 the branch prediction methods have improved massiveley.

## The Structure and Performance of Interpters (1996)

* __Ref__: Romer1996
* __Authors__: Theodore H. Romer, Dennis Lee, Geoffrey M. Voelker, Alec Wolman, Wayne A. Wong, Jean-Loup Baer, Brian N. Bershad, and Henry M. Levy
* __DOI__: https://doi.org/10.1145/237090.237175

The paper analyses the performance of four common language interpreters: MIPSI, Java, Perl, Tcl.
Each interpreter has:
* An interface consisting of virtual commands (e.g. instructions in Java bytecode)
* A memory model for storing and accessing program data
* Connection to native, external libraries (e.g. Java uses external library for graphics rendering)

An interpreter executes virtual commands on a given machine and an architecture using native commands.
An execution of a virtual command has an overhead of decoding and issuing the corresponding native instructions.
The more instruction to execute (simple interface), the bigger the overall overhead is.
On the other hand, complex interface with fewer virtual instruction consists of commands which are hard to decode and execute.
Therefore, one indicator of the interpreter performance is the ratio of native to virtual commands which describe the overhead required to execute the virtual commands.

An efficient memory model for the interpreter is crucial for its high performance due to the high use of instructions altering the memory.
When comparing the performance of interpreters it is important to asses how much of the program execution is delegated to an external library (e.g. MATLAB uses Intel MKL library for linear algebra operations).
Moreover, the distribution of virtual commands in benchmark programs is skewed and some commands account for the majority of the performance.

The empirical evalution in the paper shows interpreted programs are slower than the native ones (due to the overhead of an execution of virtual commands).
Moreover, different interpreters tend to have similar performance characterstics despite the kind of program they execute (performance of the interpreter overwhelms the performance of the program).
However, storing the program as data does not increase the number of cache misses significantly.
Finally, the paper conludes with a statement saying there is no need for a creation of specific hardware for interpreters.
