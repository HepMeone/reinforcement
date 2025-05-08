
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_gantt(task_log):
    df = pd.DataFrame(task_log)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, row in df.iterrows():
        ax.barh(row['resource'], row['end'] - row['start'], left=row['start'],
                color='red' if row['priority'] > 1 else 'blue')
        ax.text(row['start'], i, f"Task {row['task_id']}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Resource")
    plt.title("Scheduling Gantt Chart")
    plt.tight_layout()
    plt.savefig("logs/gantt_chart.png")
    plt.close()

def plot_resource_heatmap(util_matrix, departments):
    plt.figure(figsize=(8, 6))
    sns.heatmap(util_matrix, annot=True, xticklabels=departments, yticklabels=["Machine" + str(i) for i in range(len(util_matrix))])
    plt.title("Resource Utilization Heatmap")
    plt.xlabel("Departments")
    plt.ylabel("Machines")
    plt.tight_layout()
    plt.savefig("logs/utilization_heatmap.png")
    plt.close()
