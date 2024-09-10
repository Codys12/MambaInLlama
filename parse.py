import re
import csv
from collections import defaultdict

def parse_file(filename):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
    
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as file:
                content = file.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Unable to decode the file with any of the attempted encodings: {encodings}")

    # Split the content into blocks based on percentage lines
    blocks = re.split(r'(\d+%\|.*?\n)', content)
    
    data = defaultdict(lambda: {'attn': [], 'mamba': [], 'combined': [], 'distance': [], 'training_loss': []})
    current_percentage = None

    for block in blocks:
        percentage_match = re.match(r'(\d+)%\|', block)
        if percentage_match:
            current_percentage = f"{percentage_match.group(1)}%"
        elif current_percentage:
            attn = re.findall(r'attn_output magnitude: (\d+\.\d+)', block)
            mamba = re.findall(r'mamba_output magnitude: (\d+\.\d+)', block)
            combined = re.findall(r'combined magnitude: (\d+\.\d+)', block)
            distance = re.findall(r'distance: (\d+\.\d+)', block)
            training_loss = re.findall(r'training loss: (\d+\.\d+)', block)

            data[current_percentage]['attn'].extend(map(float, attn))
            data[current_percentage]['mamba'].extend(map(float, mamba))
            data[current_percentage]['combined'].extend(map(float, combined))
            data[current_percentage]['distance'].extend(map(float, distance))
            data[current_percentage]['training_loss'].extend(map(float, training_loss))

    return data

def calculate_averages(data):
    averages = {}
    for percentage, values in data.items():
        averages[percentage] = {
            'attn': sum(values['attn']) / len(values['attn']) if values['attn'] else 0,
            'mamba': sum(values['mamba']) / len(values['mamba']) if values['mamba'] else 0,
            'combined': sum(values['combined']) / len(values['combined']) if values['combined'] else 0,
            'distance': sum(values['distance']) / len(values['distance']) if values['distance'] else 0,
            'training_loss': sum(values['training_loss']) / len(values['training_loss']) if values['training_loss'] else 0
        }
    return averages

def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Percentage', 'Avg Attn Magnitude', 'Avg Mamba Magnitude', 'Avg Combined Magnitude', 'Avg Distance', 'Avg Training Loss'])
        # Sort the percentages numerically
        sorted_percentages = sorted(data.keys(), key=lambda x: int(x[:-1]))
        for percentage in sorted_percentages:
            values = data[percentage]
            writer.writerow([
                percentage,
                f"{values['attn']:.2f}",
                f"{values['mamba']:.2f}",
                f"{values['combined']:.2f}",
                f"{values['distance']:.2f}",
                f"{values['training_loss']:.2f}"
            ])

def main():
    input_file = 'data.txt'
    output_file = 'magnitude_distance_and_loss_averages.csv'

    try:
        data = parse_file(input_file)
        averages = calculate_averages(data)
        save_to_csv(averages, output_file)
        print(f"Averages have been calculated and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()