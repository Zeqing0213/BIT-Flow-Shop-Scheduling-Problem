import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import time

def calc_makespan(solution, processing_time): 
    number_of_jobs = len(solution)
    number_of_machines = len(processing_time[0])
    cost = [0] * number_of_jobs   

    for machine_no in range(number_of_machines):   
        for slot in range(number_of_jobs):  
            cost_so_far = cost[slot]  
            if slot > 0:
                cost_so_far = max(cost[slot - 1], cost[slot])   
            cost[slot] = cost_so_far + processing_time[solution[slot]][machine_no]   

    return cost[number_of_jobs - 1]   


def generate_initial_solution(number_of_jobs): 
    return list(range(number_of_jobs))


def generate_neighbors(solution): 
    neighbors = []
    length = len(solution)

    for i in range(length):
        for j in range(i + 1, length):
            neighbor = list(solution)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)

    return neighbors

# Original hill-climbing Algorithm
def hill_climbing(processing_time):
    number_of_jobs = len(processing_time)
    current_solution = generate_initial_solution(number_of_jobs)   
    best_solution = list(current_solution)   
    best_makespan = calc_makespan(best_solution, processing_time)   

    while True:
        neighbors = generate_neighbors(current_solution)   
        found_better_neighbor = False   

        for neighbor in neighbors:
            neighbor_makespan = calc_makespan(neighbor, processing_time)   

            
            if neighbor_makespan < best_makespan:
                best_solution = list(neighbor)
                best_makespan = neighbor_makespan
                found_better_neighbor = True

       
        if found_better_neighbor:
            current_solution = list(best_solution)
        else:
            break

    return best_solution, best_makespan
# Enhanced hill-climbing Algorithm
def enhance_hill_climbing(processing_time, num_initializations=10, num_iterations=100, num_local_iterations=50):
    best_solution = None   
    best_makespan = float('inf')   

    
    for _ in range(num_initializations):
        current_solution = generate_initial_solution(len(processing_time))
        current_makespan = calc_makespan(current_solution, processing_time)

        
        for _ in range(num_iterations):
            found_better_neighbor = False   

            
            for _ in range(num_local_iterations):
                local_best_solution = list(current_solution)   
                local_best_makespan = current_makespan   
                neighbors = generate_neighbors(current_solution)   

                
                for neighbor in neighbors:
                    neighbor_makespan = calc_makespan(neighbor, processing_time)   

                    
                    if neighbor_makespan < local_best_makespan:
                        local_best_solution = list(neighbor)
                        local_best_makespan = neighbor_makespan
                        found_better_neighbor = True

                   
                    elif neighbor_makespan == local_best_makespan:
                        if neighbor < local_best_solution:
                            local_best_solution = list(neighbor)
                            local_best_makespan = neighbor_makespan
                            found_better_neighbor = True

                
                if found_better_neighbor:
                    current_solution = list(local_best_solution)
                    current_makespan = local_best_makespan
                else:
                    break

           
            if current_makespan < best_makespan:
                best_solution = list(current_solution)
                best_makespan = current_makespan

    return best_solution, best_makespan


def generate_random_neighbor(solution):   
    neighbor = list(solution)
    length = len(neighbor)
    i, j = random.sample(range(length), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

# Simulated annealing Algorithm
def simulated_annealing(processing_time, initial_temperature=10000, cooling_rate=0.99, iterations_per_temperature=1000):
    number_of_jobs = len(processing_time)
    current_solution = generate_initial_solution(number_of_jobs)  
    best_solution = list(current_solution)  
    best_makespan = calc_makespan(best_solution, processing_time)
    temperature = initial_temperature

    while temperature > 0.01:
        improved = False  
        for _ in range(iterations_per_temperature):
            new_solution = generate_random_neighbor(current_solution)   
            new_makespan = calc_makespan(new_solution, processing_time)   
            cost_difference = new_makespan - best_makespan  # 计算新解与最优解的完成时间差

            
            if cost_difference < 0 or random.random() < math.exp(-cost_difference / temperature):
                current_solution = list(new_solution)
                best_makespan = new_makespan
                improved = True

           
            if improved:
                best_solution = list(current_solution)

        temperature *= cooling_rate   

    return best_solution, best_makespan


def generate_random_color():
    r = random.random()
    g = random.random()
    b = random.random()
    return (r, g, b)

def plot_gantt_chart(solution, processing_time, best_makespan, algorithm_name, instance_number):
    number_of_jobs = len(solution)
    number_of_machines = len(processing_time[0])

    
    start_time = [[0] * number_of_machines for _ in range(number_of_jobs)]
    end_time = [[0] * number_of_machines for _ in range(number_of_jobs)]

    for job in range(number_of_jobs):
        for machine in range(number_of_machines):
            if machine == 0:
                if job == 0:
                    start_time[job][machine] = 0   
                else:
                    start_time[job][machine] = end_time[job - 1][machine]
            else:
                start_time[job][machine] = max(end_time[job][machine - 1], end_time[job - 1][machine])

            end_time[job][machine] = start_time[job][machine] + processing_time[solution[job]][machine]   

    fig, ax = plt.subplots()

    for machine in range(number_of_machines):
        for job in range(number_of_jobs):
            start = start_time[job][machine]
            duration = end_time[job][machine] - start   
            center = start + duration / 2.0
            rect = Rectangle((start, machine - 0.4), duration, 0.8, facecolor='C' + str(job % 10), edgecolor='black')
            ax.add_patch(rect)
            ax.text(center, machine, str(job + 1), ha='center', va='center', color='white', fontweight='bold')   

  
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(number_of_machines))
    ax.set_yticklabels(range(1, number_of_machines + 1))
    ax.invert_yaxis()
    ax.set_title(algorithm_name + ' - Instance ' + str(instance_number))

    completion_time = max(end_time[-1])   
    ax.axvline(x=completion_time, color='k', linestyle='--', linewidth=1)   
    ax.text(completion_time, -0.5, str(best_makespan), ha='right', va='top')   
    ax.set_ylim(-0.5, number_of_machines)

    output_folder = "output_figure"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = f"{output_folder}/{algorithm_name}_Instance_{instance_number}.png"
    plt.savefig(filename)

    plt.show()

for j in range(11):
    n, m = map(int, input().split())
    processing_times = []
    for _ in range(n):
        processing_time = list(map(int, input().split()))  
        i = 0
        while i < len(processing_time):  
            del processing_time[i]
            i += 1

        processing_times.append(processing_time) 

    print('Instance', j)

    # 调用三种算法函数计算并打印最优调度时间，画出对应甘特图
    start_time = time.time()
    best_solution_hill_climbing, best_makespan_hill_climbing = hill_climbing(processing_times)
    end_time = time.time()
    runtime_hill_climbing = round(end_time - start_time, 3)
    print("原始登山算法:")
    print("最优调度方案:", best_solution_hill_climbing)
    print("最优调度时间:", best_makespan_hill_climbing)
    print("运行时间:", runtime_hill_climbing)
    plot_gantt_chart(best_solution_hill_climbing, processing_times, best_makespan_hill_climbing, 'Original Hill Climbing', j)

    start_time = time.time()
    best_solution_enhance_hill_climbing, best_makespan_enhance_hill_climbing = enhance_hill_climbing(processing_times)
    end_time = time.time()
    runtime_enhance_hill_climbing = round(end_time - start_time, 3)
    print("enhanced hill climbing:")
    print("最优调度方案:", best_solution_enhance_hill_climbing)
    print("最优调度时间:", best_makespan_enhance_hill_climbing)
    print("运行时间:", runtime_enhance_hill_climbing)
    plot_gantt_chart(best_solution_enhance_hill_climbing, processing_times, best_makespan_enhance_hill_climbing, 'Enhanced Hill Climbing', j)

    start_time = time.time()
    best_solution_simulated_annealing, best_makespan_simulated_annealing = simulated_annealing(processing_times)
    end_time = time.time()
    runtime_simulated_annealing = round(end_time - start_time, 3)
    print("Simulated Annealing:")
    print("最优调度方案:", best_solution_simulated_annealing)
    print("最优调度时间:", best_makespan_simulated_annealing)
    print("运行时间:", runtime_simulated_annealing)
    plot_gantt_chart(best_solution_simulated_annealing, processing_times, best_makespan_simulated_annealing, 'Simulated Annealing', j)
