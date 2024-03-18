import os
from result.plotAllocation import plot_allocation

numRounds = 120
folder_name = "5task_iiiii_exp1C1c20-cpu-seed10a4"
path_plot = os.path.join('./result', folder_name)

allocation_files = [f for f in os.listdir(path_plot) if f.startswith('Algorithm')]
positions = {}
# plot allocation map
#targets = ['bayesian', 'proposed', 'random', 'round_robin']
targets = ['proposed']
algo = ['Alpha-fair', 'Random', 'Round robin']
for i, f in enumerate(allocation_files):
    for target in targets:
        if target in f:
            positions[target] = i
            break

for i in range(len(targets)):
    allocated_tasks_lists = []
    file_name = os.path.join(path_plot, allocation_files[positions[targets[i]]])
    with open(file_name, 'r') as file:
        tasks_data = ''
        recording = False
        for line in file:
            if "Allocated Tasks:" in line:
                recording = True
                tasks_data += line.split(":", 1)[1].strip()
            elif recording:
                # Check if the line still belongs to 'Allocated Tasks'
                if line.startswith('Task[') or 'Round [' in line:

                    recording = False
                else:
                    tasks_data += ' ' + line.strip()

        if tasks_data:
            # Replace spaces with commas and remove any newlines

            tasks_data = tasks_data.replace(' ', ',').replace('\n', '')
            tasks_data = tasks_data.replace(',,', ',')

            tasks_data = tasks_data.replace('][', '],[')
            tasks_data = '[' + tasks_data + ']'

            # Ensure the string is a valid Python list format
            tasks_list = eval(tasks_data)
            tasks_list = [[int(item) for item in sublist] for sublist in tasks_list]

    plot_allocation(tasks_list, path_plot, numRounds, algo[i])