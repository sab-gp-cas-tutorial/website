# Tutorial on Using Genetic Programming to Design Complex Adaptive Systems

**A hands-on SAB 2026 tutorial with JAX, Kozax, and ABMax**

> SAB 2026 — From Animals to Animats 18  
> Berlin, Germany · October 2026 · Half-day tutorial

---

## Overview

How can we design complex systems that behave adaptively while being interpretable at the same time?

In this hands-on tutorial, participants will learn how **genetic programming** and **symbolic regression** can be used to discover compact, readable update rules for agent-based models. We combine [Kozax](https://github.com/sdevries0/Kozax), a scalable genetic programming library in JAX, with [ABMax](https://github.com/i-m-iron-man/abmax.git), a JAX-based agent-based modelling framework, to construct and evaluate interpretable models of complex adaptive systems.

The tutorial introduces core ideas through a practical **consensus task**: agents must combine noisy individual information with local social information. Participants will use genetic programming to discover an agent update rule and compare the resulting equation with a known hand-designed mechanism.

---

## Frameworks
- Kozax: Kozax introduces a general framework for evolving computer programs with genetic programming in JAX. With JAX, the computer programs can be vectorized and evaluated on parallel on CPU and GPU. Furthermore, just-in-time compilation provides massive speedups for evolving offspring.
- ABMax: Abmax is a general-purpose agent-based modeling(ABM) framework in Jax
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

**Prerequisites:** Basic familiarity with Python, JAX, genetic programming and agent-based modelling is helpful. A primer on JAX, genetic programming, and agent-based modelling will be provided.

---

## Learning Outcomes

By the end of the tutorial, participants will be able to:

1. Explain how symbolic regression can provide interpretable alternatives to black-box agent controllers
2. Understand the basic principles of genetic programming
3. Use essential JAX concepts: vectorisation, random number handling, and just-in-time compilation
4. Define symbolic search spaces and evaluate candidate expressions with Kozax
5. Construct agent-based simulations with ABMax
6. Connect discovered symbolic expressions to agent update rules
7. Evaluate and interpret a genetically programmed solution to a multi-agent consensus task

---

## Tentative Schedule

| Session | Topic |
|---------|-------|
| Introduction | Complex adaptive systems, agent-based models, and interpretable agent rules |
| JAX Primer | Arrays, vectorisation, random numbers, and just-in-time compilation |
| Genetic Programming with Kozax | Search spaces, candidate expressions, objectives, and selection |
| Agent-Based Modelling with ABMax | Agents, states, parameters, and update functions |
| Hands-on Exercise | Discovering an interpretable rule for a local consensus task |
| Discussion | Interpreting discovered mechanisms and extending the pipeline |

---

## Hands-on Task

Agents move in a shared environment while maintaining an internal estimate of a hidden target. Some agents receive noisy information about the target; others observe only nearby neighbours.

Participants will use genetic programming to search for a compact local update rule that combines social information with self-pinning. The discovered expression is then evaluated inside a scalable agent-based simulation built with ABMax.

---

## Preparation

Please bring a **laptop capable of running Python notebooks**.

Installation instructions, tutorial notebooks, and supporting materials will be added to this repository before the conference.

| Resource | Status |
|----------|--------|
| Tutorial repository | *Coming soon* |
| Installation instructions | *Coming soon* |
| Slides | *Coming soon* |
| Jupyter notebooks | *Coming soon* |

---

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

