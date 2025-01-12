# Genetic Algorithm for RPQ Problem Optimization with Minimization of Total Completion Time

This repository implements a genetic algorithm (GA) to solve the RPQ scheduling problem (Release date, Processing time, Delivery time) with the objective of minimizing the makespan (total completion time). The algorithm is designed for tasks with specific constraints, such as release dates and delivery times, and incorporates daily time windows (shifts) into the scheduling process.

---

## Table of Contents
- [Problem Overview](#problem-overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example Results](#example-results)
  
---

## Problem Overview

The RPQ scheduling problem focuses on assigning tasks to machines while considering:
- Release Date (R): The earliest time a task can start.
- Processing Time (P): The time required to complete the task.
- Delivery Time (Q): The time to finalize and deliver the task after processing.

The primary goal is to determine a task schedule that minimizes the total completion time, ensuring efficient utilization of resources while adhering to scheduling constraints.

---

## Features

- Tournament Selection: A selection method that randomly chooses a subset of individuals from the population and selects the best individual for reproduction.
- Crossover and Mutation: Implementation of crossover and mutation operations to introduce genetic diversity and explore the solution space. There are 3 types of crossover implemented:
    - PMX (Partially Mapped Crossover),
    - OX1 (Order Crossover 1),
    - Permutation-preserving k-point crossover
- Fitness Evaluation: Evaluation of individuals based on their total completion time, guiding the algorithm toward better schedules.
- Flexible Configuration: Adjustable parameters for population size, mutation rate, crossover probability, and tournament size.
- Progress bar informing about generation execution
  
---

## How It Works

- Initialization: Generate an initial population of schedules.
- Selection: Use tournament selection to choose parents for reproduction.
- Crossover: Combine parents to produce offspring with potentially better schedules.
- Mutation: Introduce random changes to offspring for diversity.
- Evaluation: Assess the fitness of each individual in the population.
- Iteration: Repeat the selection, crossover, mutation, and evaluation process for a defined number of generations or until a termination condition is met.

---

## Requirements

The project requires the following:
- Python 3.8 or newer
- Libraries: copy, random, matplotlib (for visualization, tqdm (progress bar)

---

## Installation
git clone https://github.com/pzimnota/rpq-genetic.git

---

## Usage
1. Decide if you want to get randomize data or insert it yourself (no imports atm):
   - leave data = generate_random_problem() in def genetic_algorithm()
   - or uncomment data = load_problem_data()
     
2. Run the script:
   - python algorithm.py 
   
3. Results:
   - Total day time
   - Fitness plot
   - Process sequence diagram for each day
   - Arranging processes in the appropriate order

---

## Example Results
<p align="center">
  <img src="https://github.com/user-attachments/assets/a073abf7-3888-4f2c-af3c-1f805cfca96d" alt="fitness_plot" />
</p>

<p align="center" style="text-align: center;">
  Fitness plot with 450 processes.
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/25502761-2f95-4cee-b391-aba5dceacb00" alt="day_graph" />
</p>

<p align="center" style="text-align: center;">    
  One of the days graph.
</p>

---
