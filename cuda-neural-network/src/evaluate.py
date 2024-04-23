def calculate_average_duration(data):
  average_durations = []

  for i in range(0, len(data), 5):
    neural_network_data = data[i:i+5]

    sum_duration = 0
    for row in neural_network_data:
      duration = float(row[2])
      sum_duration += duration
    average_duration = sum_duration / len(neural_network_data)

    average_durations.append(average_duration)

  return average_durations

data = [
    ["linearLayerForward(float, float, float *....", 0.2948185, 2.304, "GPU 0  Stream 16"],
    ["reluActivation Forward(float, float, int....", 0.2948235, 2.176, "GPU 0  Stream 16"],
    ["linearLayerForward(float, float, float *....", 0.2948275, 3.329, "GPU 0  Stream 16"],
    ["reluActivation Forward(float, float, int,...", 0.2948315, 1.824, "GPU 0  Stream 16"],
    ["sigmoidActivation Forward(float, float....", 0.2948355, 1.920, "GPU 0  Stream 16"],
    ["linearLayerForward(float, float, float *....", 0.2948945, 2.208, "GPU 0  Stream 16"],
]

average_durations = calculate_average_duration(data)

for i, avg_duration in enumerate(average_durations):
  print(f"Average duration for neural network {i+1}: {avg_duration:.6f} μs")
