# Tutorial on Using Genetic Programming to Design Complex Adaptive Systems

**A hands-on SAB 2026 tutorial with JAX, Kozax, and ABMax**

> SAB 2026 — From Animals to Animats 18  
> Berlin, Germany · October 2026 · Four hours, including breaks

---

## Overview

How can we design complex systems that behave adaptively while being interpretable at the same time?

In this hands-on tutorial, participants will learn how **genetic programming** can be used to discover compact, readable behavioural rules for agent-based models. We combine [Kozax](https://github.com/sdevries0/Kozax), a scalable genetic programming library in JAX, with [ABMax](https://github.com/i-m-iron-man/abmax.git), a JAX-based agent-based modelling framework, to construct and evaluate interpretable models of complex adaptive systems.

The tutorial introduces core ideas through **inverse Schelling's model**, an example of **Inverse Generative Social Science (IGSS)**. Participants will use genetic programming to discover local relocation rules that generate coexisting segregated and integrated neighbourhoods. The session combines primers with practical exercises: building the simulator, connecting it to evolutionary search, and interpreting the resulting rules and spatial patterns.

---

## Frameworks

- Kozax introduces a general framework for evolving computer programs with genetic programming in JAX. With JAX, the computer programs can be vectorized and evaluated on parallel on CPU and GPU. Furthermore, just-in-time compilation provides massive speedups for evolving offspring.
- ABMax is a general-purpose agent-based modeling(ABM) framework in Jax
It provides:
  - Two algorithms to apply unique updates to a dynamic number of agents selected during run-time. Both are JIT-friendly and can be vectorized across different number of models
    * Rank-Match (RM)
    * Sort-Count-Iterate (SCI)
  - JAX-transformation-friendly data structures and functions that can be used to define sets of agents and their manipulations.
    * Selecting agents based on a run-time determined key.
    * Searching and sorting agents based on their attributes.
    * Updating an arbitrary number of agents to a specific state.
    * Stepping agents in a vectorized way.
    * Running multiple such simulations in parallel.
  - Implementation of common algorithms used in ABM implemented in vmap and JIT-friendly way.

## Who Should Attend?

This tutorial is aimed at researchers and students interested in any of the following:

- Agent-based modelling and complex adaptive systems
- Artificial life and adaptive behaviour
- Evolutionary computation and genetic programming
- Interpretable machine learning
- Collective behaviour and multi-agent systems
- JAX-based scientific computing

**Prerequisites:** Basic familiarity with Python is recommended. Familiarity with JAX, genetic programming and agent-based modelling is helpful; primers on these topics, complex adaptive systems, ABMax and Kozax will be provided.

---

## Learning Outcomes

By the end of the tutorial, participants will be able to:

1. Explain how individual agent rules and interactions generate collective patterns
2. Formulate an IGSS problem using a target spatial pattern and a simulation fitness measure
3. Use essential JAX concepts: arrays, vectorisation, random number handling, scans and just-in-time compilation
4. Construct a Schelling simulation with ABMax, including relocation events, destination allocation and synchronous updates
5. Define symbolic search spaces and evaluate candidate utility expressions with Kozax
6. Connect Kozax to ABMax to evolve local relocation rules
7. Compare evolved rules with baseline rules across multiple initial conditions
8. Interpret discovered expressions and assess how well they reproduce the target patterns

---

## Tentative Schedule

**Duration: four hours, including three 10-minute breaks.** Times below are measured from the start of the tutorial.

| Elapsed time | Session | Focus |
|--------------|---------|-------|
| 00:00–00:20 | Complex Adaptive Systems Primer | Interactions, emergence, feedback and adaptation |
| 00:20–00:35 | Agent-Based Modelling Primer | Explicit events, state changes and scheduling; connections to differential-equation models |
| 00:35–00:55 | Inverse Generative Social Science | Inverse Schelling's model, target patterns and fitness |
| 00:55–01:10 | JAX Primer | Arrays, vectorisation, random keys, scans and just-in-time compilation |
| **01:10–01:20** | **Break** | **10 minutes** |
| 01:20–01:35 | ABMax Primer | Agents, states, parameters and simulation functions |
| 01:35–02:10 | ABMax Implementation | Build and run the Schelling model; inspect and modify a baseline utility rule |
| **02:10–02:20** | **Break** | **10 minutes** |
| 02:20–02:40 | Genetic Programming Primer | Expression trees, selection, crossover and mutation |
| 02:40–02:50 | Kozax Primer | Features, operators and symbolic program evaluation |
| 02:50–03:05 | Kozax–ABMax Interface | Evaluate a candidate utility rule through the simulator and return its fitness |
| **03:05–03:15** | **Break** | **10 minutes** |
| 03:15–03:45 | Evolutionary Discovery | Run a small GP search, inspect trajectories and compare rules |
| 03:45–04:00 | Interpretation and Discussion | Explain evolved rules, test fresh initial conditions and discuss extensions |

---

## Hands-on Task

**Inverse Schelling's model: discovering relocation rules that generate mixed spatial patterns.**

Blue and green agents occupy a grid containing vacant cells. Each agent evaluates its current neighbourhood and possible destinations using factors such as neighbourhood similarity, vacancy, recent movement, neighbourhood residence age and relocation distance. The aim is to discover compact utility expressions whose interactions generate coexisting homogeneous blue, homogeneous green and mixed neighbourhoods.

The exercise builds on **[Generating Mixed Patterns of Residential Segregation: An Evolutionary Approach](https://www.jasss.org/26/2/7.html)** by Chathika Gunaratne, Erez Hatna, Joshua M. Epstein and Ivan Garibay (2023), published in *Journal of Artificial Societies and Social Simulation*, 26(2), 7.

The tutorial adapts this approach to ABMax and Kozax, with **Hungarian assignment** to allocate distinct destinations and **synchronous batch updates** to commit accepted moves from a shared state snapshot. GP evolves the utility expressions while these allocation and update mechanisms remain fixed.

Participants will:

1. Run and visualise a baseline Schelling simulation.
2. Define a search space of neighbourhood features and mathematical operators.
3. Connect Kozax's candidate expressions to the ABMax simulation.
4. Use a C-index-based fitness measure to evaluate the coexistence of the three neighbourhood types over the final part of each trajectory and across multiple scenarios.
5. Run evolutionary search, compare discovered rules with baselines, and inspect the resulting patterns and movement histories.
6. Test fresh initial conditions and discuss what the discovered expressions suggest about the mechanisms generating the patterns.


---

## Preparation

Please bring a **laptop with a web browser** and access to a Google account. The hands-on exercises will run entirely in Google Colab, so no local installation is required.

Open the tutorial notebook using the link below. In Colab, select **File → Save a copy in Drive** if you want to retain your changes. The notebook will automatically download the supporting files and create the required folders.

| Resource | Access |
|----------|--------|
| Tutorial repository | [View on GitHub](https://github.com/sab-gp-cas-tutorial/website) |
| Hands-on notebook | [Open in Google Colab](https://colab.research.google.com/github/sab-gp-cas-tutorial/website/blob/main/tutorial/sab_tutorial.ipynb) |
| Setup instructions | Included in the notebook |
| Slides | [View here](https://docs.google.com/presentation/d/1heJN129jgIHIrQ1HQeg32m0ujRRsACsh/edit?usp=sharing&ouid=102615649183888622882&rtpof=true&sd=true) |

## Instructors

**Siddharth Chaturvedi**  
Department of Machine Learning and Neural Computing, Donders Institute for Brain, Cognition and Behaviour, Radboud University.  
Siddharth is a developer of ABMax, a JAX-based framework for scalable agent-based modelling. His research focuses on adaptive behaviour, agent-based artificial intelligence, and complex systems.

**Sigur de Vries**  
Department of Machine Learning and Neural Computing, Donders Institute for Brain, Cognition and Behaviour, Radboud University.  
Sigur is a developer of Kozax, a flexible and scalable genetic programming library implemented in JAX.

**Artificial Cognitive Systems lab**
led by [prof. Marcel van Gerven](https://scholar.google.com/citations?user=sX0ZypwAAAAJ&hl=en), studies the computational mechanisms of learning, inference and control in natural and artificial systems. To this end, we bring together ideas from a wide range of disciplines such as machine learning, computational neuroscience, control theory, dynamical systems theory, statistical physics and theoretical biology. Ultimately, our goal is to bridge the gap between natural and artificial intelligence and contribute more capable and efficient AI solutions to address a wide variety of scientific and societal challenges.

---

## Contact

For questions about the tutorial, please contact:

**Siddharth Chaturvedi** — siddharth.chaturvedi@donders.ru.nl

For general conference information, visit the [SAB 2026 website](https://sab-conference.org).

---

*This tutorial is part of [SAB 2026 — From Animals to Animats 18](https://sab-conference.org), Berlin, 19–22 October 2026.*

