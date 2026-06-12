# Tutorial on Using Genetic Programming to Design Complex Adaptive Systems

**A hands-on SAB 2026 tutorial with JAX, Kozax, and ABMax**

> SAB 2026 — From Animals to Animats 18  
> Berlin, Germany · October 2026 · Half-day tutorial

---

## Overview

How can we design agents that behave adaptively while keeping their internal mechanisms interpretable?

In this hands-on tutorial, participants will learn how **genetic programming** and **symbolic regression** can be used to discover compact, readable update rules for agent-based models. We combine [Kozax](https://github.com/sdevries0/Kozax), a scalable genetic programming library in JAX, with [ABMax](https://github.com/i-m-iron-man/abmax.git), a JAX-based agent-based modelling framework, to construct and evaluate interpretable models of complex adaptive systems.

The tutorial introduces core ideas through a practical **consensus task**: agents must combine noisy individual information with local social information. Participants will use genetic programming to discover an agent update rule and compare the resulting equation with a known hand-designed mechanism.

---

## Who Should Attend?

This tutorial is aimed at researchers and students interested in any of the following:

- Agent-based modelling and complex adaptive systems
- Artificial life and adaptive behaviour
- Evolutionary computation and genetic programming
- Interpretable machine learning
- Collective behaviour and multi-agent systems
- JAX-based scientific computing

**Prerequisites:** Basic familiarity with Python is helpful. Prior experience with JAX, genetic programming, or agent-based modelling is **not required**.

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
| ABMax documentation | *Coming soon* |
| Kozax documentation | *Coming soon* |

---

## Instructors

**Siddharth Chaturvedi**  
Department of Machine Learning and Neural Computing, Donders Institute for Brain, Cognition and Behaviour, Radboud University.  
Siddharth is a developer of ABMax, a JAX-based framework for scalable agent-based modelling. His research focuses on adaptive behaviour, agent-based artificial intelligence, and complex systems.

**Sigur de Vries**  
Department of Machine Learning and Neural Computing, Donders Institute for Brain, Cognition and Behaviour, Radboud University.  
Sigur is a developer of Kozax, a flexible and scalable genetic programming library implemented in JAX.

---

## Contact

For questions about the tutorial, please contact:

**Siddharth Chaturvedi** — siddharth.chaturvedi@donders.ru.nl

For general conference information, visit the [SAB 2026 website](https://sab-conference.org).

---

*This tutorial is part of [SAB 2026 — From Animals to Animats 18](https://sab-conference.org), Berlin, 19–22 October 2026.*

