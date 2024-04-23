import csv

def calculate_average_duration(data):
  average_durations = []

  for i in range(0, len(data), 4):
    neural_network_data = data[i:i+4]

    sum_duration = 0
    for row in neural_network_data:
      duration_str = row["Duration"]
      duration = float(duration_str.split()[0])
      sum_duration += duration

    average_duration = sum_duration / len(neural_network_data)
    average_durations.append(average_duration)

  total_average_duration = sum(average_durations) / len(average_durations)
  return total_average_duration

data = []
with open('./stream_4_batch_4(Sheet1).csv', 'r') as file:
  reader = csv.DictReader(file)
  for row in reader:
    data.append(row)

average_duration = calculate_average_duration(data)
print(f"Average duration for all neural networks: {average_duration:.6f} μs")
