import re
import sys
import csv

def process_log_file(filename):
    losses_by_percentage = {}
    total_steps = None

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if 'training loss:' in line:
                # Extract current_step and total_steps
                step_match = re.search(r'\s+(\d+)/(\d+)', line)
                loss_match = re.search(r'training loss: ([\d\.]+)', line)
                if step_match and loss_match:
                    current_step = int(step_match.group(1))
                    if not total_steps:
                        total_steps = int(step_match.group(2))
                    loss = float(loss_match.group(1))
                    percentage = int(current_step * 100 / total_steps)
                    losses_by_percentage.setdefault(percentage, []).append(loss)
                else:
                    print(f"Warning: Could not parse line: {line.strip()}")

    # Calculate average loss per percentage
    average_losses = []
    for percent in sorted(losses_by_percentage.keys()):
        avg_loss = sum(losses_by_percentage[percent]) / len(losses_by_percentage[percent])
        average_losses.append(avg_loss)

    # Write to CSV
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Average Loss'])
        for avg_loss in average_losses:
            csv_writer.writerow([avg_loss])

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <logfile>")
    else:
        process_log_file(sys.argv[1])
