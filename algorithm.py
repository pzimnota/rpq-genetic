import copy
import random

import matplotlib.pyplot as plt
from tqdm import tqdm

import constants



def load_problem_data():
    return [
        (1, 10, 5, 7),
        (2, 13, 6, 26),
        (3, 11, 7, 24),
        (4, 20, 4, 21),
        (5, 30, 3, 8),
        (6, 0, 6, 17),
        (7, 30, 2, 0),
    ]


def generate_random_problem():
    size = constants.SIZE

    maxium_r = 500
    maxium_p = 600
    maxium_q = 500

    problem = []

    for i in range(0, size):
        elem = (i, random.randrange(0, maxium_r), random.randrange(30, maxium_p), random.randrange(0, maxium_q))
        problem.append(elem)

    return problem


def init_population(problem_data):
    population = []
    for _ in range(constants.POPULATION_SIZE):
        shuffled = random.sample(problem_data, len(problem_data))
        population.append(shuffled)
    return population


def fitness(process_queue):
    current_time = 0
    current_day_time = 0
    last_finish_time = 0

    end_of_day_waste_time_added = 0  # time added to the end of the day meaning that nothing else could be done
    # WE ASSUME THAT THE DELIVERY CAN ALSO TAKE PLACE ONLY DURING WORK AND MUST BE COMPLETED

    for j, r, p, q in process_queue:
        # this if only deals with resetting the days and adding the missing time to the delivery,
        # it does not add production and delivery time
        missing_time = r - current_day_time
        if (
                missing_time >= 0 and current_day_time + missing_time + p + q > constants.MAXIMUM_DAY_WORK) or current_day_time + p + q > constants.MAXIMUM_DAY_WORK:
            # process/ product won't be delivered and executed during that day, so we skip it to another, new day
            end_of_day_waste_time_added += constants.MAXIMUM_DAY_WORK - current_day_time  # wasted time
            current_day_time = 0  # day reset
            missing_time = r

        if missing_time >= 0:  # normal case in which process can be done during this day
            current_time += missing_time
            current_day_time += missing_time

        current_time += p
        current_day_time += p
        finish_time = current_time + q
        if finish_time > last_finish_time:
            last_finish_time = finish_time

    return last_finish_time + end_of_day_waste_time_added


def tournament_selection(previous_population):
    chosen_individuals = random.sample(previous_population, constants.TOURNAMENT_BATCH_SIZE)
    best_fitness = float('inf')
    best_process = None
    for individual in chosen_individuals:
        # Possible optimization by storing fitness
        individual_fitness = fitness(individual)
        if individual_fitness < best_fitness:
            best_fitness = individual_fitness
            best_process = individual

    return best_process


def crossover(individual1, individual2):
    new_individual1 = copy.deepcopy(individual1)
    new_individual2 = copy.deepcopy(individual2)

    margin_length = int(len(individual1) * constants.CROSSOVER_POINT_MARGIN)
    crosspoint = random.randrange(margin_length, len(individual1) - margin_length)

    for i in range(0, crosspoint):
        element = individual1[i]

        index2 = new_individual2.index(element)
        new_individual2[i], new_individual2[index2] = new_individual2[index2], new_individual2[i]

    for i in range(0, crosspoint):
        element = individual2[i]

        index2 = new_individual1.index(element)
        new_individual1[i], new_individual1[index2] = new_individual1[index2], new_individual1[i]

    return new_individual1, new_individual2


def crossover2(individual1, individual2):
    new_individual2 = copy.deepcopy(individual2)

    mutation_start = random.randrange(0, len(individual1) - 1)
    mutation_end = random.randrange(mutation_start, len(individual1) - 1)

    values_to_copy = individual1[mutation_start:mutation_end + 1]
    individual2_old_values = individual2[mutation_start:mutation_end + 1]

    new_individual2[mutation_start:mutation_end + 1] = values_to_copy
    for new_element, old_element in zip(values_to_copy, individual2_old_values):
        index_of_added_element = new_individual2.index(new_element)
        new_individual2[index_of_added_element] = old_element

    assert (len(new_individual2) == len(individual1))
    assert (set(new_individual2) - set(individual1) == set())

    return new_individual2, None


def crossover3(individual1, individual2):
    number_of_segments = random.randrange(1, constants.CROSSOVER_MAXIMUM_SEGMENTS)

    sample_values = sorted(random.sample(range(0, len(individual1)), number_of_segments * 2))

    new_individual = [None] * len(individual1)

    for i in range(0, len(sample_values), 2):
        start_index = sample_values[i]
        end_index = sample_values[i + 1]
        new_individual[start_index:end_index + 1] = individual1[start_index:end_index + 1]

    missing_values = list(filter(lambda x: x not in new_individual, individual1))

    current_missing_value = 0
    for i in range(0, len(new_individual)):
        if new_individual[i] is None:
            new_individual[i] = missing_values[current_missing_value]
            current_missing_value += 1

    assert (len(new_individual) == len(individual1))
    assert (set(new_individual) - set(individual1) == set())

    return new_individual, None


def mutate(individual):
    for _ in range(0, int(len(individual) * constants.MUTATION_SIZE)):
        index1 = random.randint(0, len(individual) - 1)
        index2 = random.randint(0, len(individual) - 1)
        individual[index1], individual[index2] = individual[index2], individual[index1]


def genetic_algorithm():
    random.seed(1) # seed for reseach purpose
    # data = load_problem_data()
    data = generate_random_problem()
    population = init_population(copy.deepcopy(data))

    population_fitnesses = {}
    best_fittnesses = {}
    best_processes = {}

    for generation_index in tqdm(range(constants.GENERATIONS)):

        generation_total_fitness = 0
        generation_best_fit = float('inf')
        generation_best_process = None

        new_population = []

        # Generate population with crossover
        while len(new_population) < constants.POPULATION_SIZE:
            individual = tournament_selection(population)

            if random.random() < constants.CROSSOVER_PROB and len(new_population):
                individual2 = tournament_selection(population)
                new_individual1, _ = crossover3(individual, individual2)
                new_population.append(new_individual1)
            else:
                new_population.append(individual)

        # Mutate
        for individual in new_population:
            if random.random() < constants.MUTATION_PROB:
                mutate(individual)

        # Evaluate the fitnesses of the generation
        for individual in new_population:
            individual_fitness = fitness(individual)
            generation_total_fitness += individual_fitness
            if individual_fitness < generation_best_fit:
                generation_best_fit = individual_fitness
                generation_best_process = copy.deepcopy(individual)

        # Update generations data
        population_fitnesses[generation_index] = generation_total_fitness / constants.POPULATION_SIZE
        best_fittnesses[generation_index] = generation_best_fit
        best_processes[generation_index] = generation_best_process

        population = new_population

    best_fit_index = 0
    for element, i in zip(best_fittnesses.values(), range(0, len(best_fittnesses))):
        if element < best_fit_index:
            best_fit_index = element

    print(to_days_representation(best_processes[best_fit_index]))
    print(f"Genetic best: {best_fittnesses[best_fit_index]}")
    plt.plot(population_fitnesses.keys(), population_fitnesses.values(), label="Average fitness")
    plt.plot(best_fittnesses.keys(), best_fittnesses.values(), label="Best fitness")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Cmax")
    plt.grid(True)
    plt.annotate(f"Number of processes = {constants.SIZE}", xy=(0.1, 0.95), xycoords='axes fraction', fontsize=17)
    plt.show()

    for day in to_days_representation(best_processes[best_fit_index]):
        print(day)
        chartify(day)


def to_days_representation(process_queue):
    days = []
    day = []
    current_time = 0
    current_day_time = 0

    for j, r, p, q in process_queue:
        # this if only deals with resetting the days and adding the missing time to the delivery,
        # it does not add production and delivery time
        missing_time = r - current_day_time
        if (
                missing_time >= 0 and current_day_time + missing_time + p + q > constants.MAXIMUM_DAY_WORK) or current_day_time + p + q > constants.MAXIMUM_DAY_WORK:
            # process/ product won't be delivered and executed during that day, so we skip it to another, new day
            current_day_time = 0  # day reset
            missing_time = r
            days.append(day)
            day = []

        if missing_time >= 0:  # normal case in which process can be done during this day
            current_time += missing_time
            current_day_time += missing_time

        day.append(((current_day_time, p), [r, q], j))
        current_time += p
        current_day_time += p

    days.append(day)
    print(f"Total day time: {current_time / constants.MAXIMUM_DAY_WORK}")
    return days


def chartify(data):
    bar_data = list(map(lambda e: e[0], data))
    process_order = list(map(lambda e: e[2], data))
    error_data_points_start = list(map(lambda e: e[0][0], data))
    error_data_points_end = list(map(lambda e: e[0][0] + e[0][1], data))
    error_data_range_lower = list(map(lambda e: e[1][0], data))
    error_data_range_upper = list(map(lambda e: e[1][1], data))

    sizes_up = [32]
    sizes_down = [18]
    for i in range(1, len(error_data_points_start)):
        if (sizes_up[-1] == 48):
            sizes_up.append(32)
            sizes_down.append(18)
        else:
            sizes_up.append(sizes_up[-1] + 1)
            sizes_down.append(sizes_down[-1] - 1)

    fig, ax = plt.subplots()
    ax.set_ylim(0, 50)
    ax.broken_barh(bar_data, (20, 10), color="#D1FFCF", edgecolor="black")
    _, cp1, _ = ax.errorbar(error_data_points_start, sizes_up,
                            xerr=[error_data_range_lower, [0] * len(error_data_range_lower)], ecolor="black", ls='none',
                            xuplims=True, capsize=0, capthick=2)
    cp1[0].set_marker('|')
    cp1[0].set_markersize(8)

    _, cp2, _ = ax.errorbar(error_data_points_end, sizes_down,
                            xerr=[[0] * len(error_data_range_lower), error_data_range_upper], ecolor="black", ls='none',
                            xlolims=True, capsize=0, capthick=2)
    cp2[0].set_marker('|')
    cp2[0].set_markersize(8)
    ax.vlines(error_data_points_start, ymin=25, ymax=sizes_up, colors="black")
    ax.vlines(error_data_points_end, ymin=20, ymax=sizes_down, colors="black")
    ax.grid(True)
    plt.xlabel("Time")
    for s, e, j in zip(error_data_points_start, error_data_points_end, process_order):
        ax.text(x=(s + e) / 2, y=22, s=j, ha='center', va='center', color='black')
    plt.xlim(0)
    plt.show()


def main():
    genetic_algorithm()


if __name__ == "__main__":
    main()
