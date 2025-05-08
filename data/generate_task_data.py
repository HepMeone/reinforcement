
import pandas as pd
import random

def generate_task_data(num_tasks=50, output_file='./task_data.csv'):
    tasks = []
    for task_id in range(num_tasks):
        task = {
            'task_id': task_id,
            'department': random.randint(0, 2),
            'process_time': random.randint(1, 10),
            'due_time': random.randint(20, 50),
            'priority': random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        }
        tasks.append(task)
    df = pd.DataFrame(tasks)
    df.to_csv(output_file, index=False)
    print(f"Generated {num_tasks} tasks to {output_file}")

if __name__ == '__main__':
    generate_task_data()
