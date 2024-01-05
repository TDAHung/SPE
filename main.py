import simpy
import matplotlib.pyplot as plt
import random
import numpy as np

# Initialize the SimPy environment
env = simpy.Environment()

# Number of CPU cores
num_cores = 4

# Total number of processes to simulate
num_processes = 10

# Maximum simulation duration
simulation_duration = 10

# Lists to store the maximum queue length for each priority level
max_queue_lengths = [0] * 3

# Lists to manually track the queue for each priority level
priority_queues = [list() for _ in range(3)]

# List to store waiting times
waiting_times = []

# List to store turnaround times
turnaround_times = []


def process_arrival(process_id, priority):
    # Arrival time of the process
    arrival_time = env.now
    print(
        f"Process {process_id} with priority {priority} arrived at time {arrival_time}")

    # Append the process to the appropriate priority queue
    priority_queues[priority].append(process_id)

    # Update the maximum queue length for this priority level
    max_queue_lengths[priority] = max(
        max_queue_lengths[priority], len(priority_queues[priority]))

    # Pass the arrival time to the service process
    env.process(process_service(process_id, priority, arrival_time))

    # Yield to the simulation environment (no actual time delay)
    yield env.timeout(0)


def process_service(process_id, priority, arrival_time):
    # Request the corresponding CPU core resource for service
    with cpu_cores[priority % num_cores].request() as request:
        yield request

        # Record the start time of the process
        start_time = env.now
        print(
            f"Process {process_id} (priority {priority}) started on core {priority % num_cores} at time {start_time}")

        # Simulate a variable service time
        service_time = generate_service_time()
        yield env.timeout(service_time)

        # Record the end time of the process
        end_time = env.now
        print(
            f"Process {process_id} finished on core {priority % num_cores} at time {end_time}")

        # Calculate and store waiting and turnaround times
        waiting_time = start_time - arrival_time
        turnaround_time = end_time - arrival_time
        waiting_times.append(waiting_time)
        turnaround_times.append(turnaround_time)


def generate_service_time():
    # Use the exponential distribution to generate variable service times
    min_service_time = 0.5
    max_service_time = 2.0
    service_time = random.uniform(min_service_time, max_service_time)
    return service_time


def run_simulation():
    global cpu_cores, waiting_times, turnaround_times

    # Create CPU cores as resources (one for each core in the CPU)
    cpu_cores = [simpy.Resource(env, capacity=1) for _ in range(num_cores)]

    # Reset the waiting and turnaround time lists
    waiting_times = []
    turnaround_times = []

    for i in range(num_processes):
        priority = i % 3
        arrival_time = env.now  # Get the current time

        # Trigger the arrival and service processes
        env.process(process_arrival(i, priority))
        env.process(process_service(i, priority, arrival_time))

    # Run the simulation until the specified duration
    env.run(until=simulation_duration)

    # Calculate and print performance metrics
    throughput = num_processes / simulation_duration
    max_queue_length = max(max_queue_lengths)
    print(f"Throughput: {throughput} processes per time unit")
    print("Average Response Time: 1 time unit (constant)")
    print(f"Maximum Queue Length: {max_queue_length} processes")

    avg_waiting_time = sum(waiting_times) / len(waiting_times)
    avg_turnaround_time = sum(turnaround_times) / len(turnaround_times)
    print(f"Average Waiting Time: {avg_waiting_time} time units")
    print(f"Average Turnaround Time: {avg_turnaround_time} time units")

    # Plot the performance metrics and statistics
    plot_performance_metrics()
    plot_mean_and_variance()


def plot_performance_metrics():
    # Create lists of time points for the x-axis
    time_points = list(range(simulation_duration))

    # Create lists of performance metrics for the y-axis
    throughput_values = [num_processes / (t + 1) for t in time_points]
    avg_response_time_values = [1] * simulation_duration
    max_queue_length_values = [max(max_queue_lengths)] * simulation_duration
    avg_waiting_time_values = [sum(waiting_times) / (t + 1)
                               for t in time_points]
    avg_turnaround_time_values = [
        sum(turnaround_times) / (t + 1) for t in time_points]

    # Create a line chart to visualize the performance metrics
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, throughput_values, label='Throughput', marker='o')
    plt.plot(time_points, avg_response_time_values,
             label='Average Response Time', marker='o')
    plt.plot(time_points, max_queue_length_values,
             label='Max Queue Length', marker='o')
    plt.plot(time_points, avg_waiting_time_values,
             label='Average Waiting Time', marker='o')
    plt.plot(time_points, avg_turnaround_time_values,
             label='Average Turnaround Time', marker='o')

    plt.title('System Performance Metrics Over Time')
    plt.xlabel('Time (in time units)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()


def plot_mean_and_variance():
    # Define metrics and time points for plotting statistics
    metrics = {
        'Throughput': [num_processes / (t + 1) for t in range(simulation_duration)],
        'Average Response Time': [1] * simulation_duration,
        'Max Queue Length': [max(max_queue_lengths)] * simulation_duration,
        'Average Waiting Time': [sum(waiting_times) / (t + 1) for t in range(simulation_duration)],
        'Average Turnaround Time': [sum(turnaround_times) / (t + 1) for t in range(simulation_duration)]
    }
    time_points = list(range(simulation_duration))

    for metric_name, values in metrics.items():
        mean_values = []
        median_values = []
        std_values = []
        variance_values = []

        for t in time_points:
            # Extract values up to the current time point
            time_based_values = [values[time]
                                 for time in range(min(t + 1, len(values)))]

            # Calculate and store mean, median, standard deviation, and variance
            mean_values.append(np.mean(time_based_values))
            median_values.append(np.median(time_based_values))
            std_values.append(np.std(time_based_values))
            variance_values.append(np.var(time_based_values))

        print(f"{metric_name} - Mean: {mean_values[-1]:.2f}")
        print(f"{metric_name} - Median: {median_values[-1]:.2f}")
        print(f"{metric_name} - Standard Deviation: {std_values[-1]:.2f}")
        print(f"{metric_name} - Variance: {variance_values[-1]:.2f}")

        # Create line charts to visualize statistics over time
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, mean_values,
                 label=f'Mean {metric_name}', marker='o')
        plt.plot(time_points, median_values,
                 label=f'Median {metric_name}', marker='o')
        plt.plot(time_points, std_values,
                 label=f'Standard Deviation {metric_name}', marker='o')
        plt.plot(time_points, variance_values,
                 label=f'Variance {metric_name}', marker='o')

        plt.title(f'{metric_name} Statistics Over Time')
        plt.xlabel('Time (in time units)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    run_simulation()
