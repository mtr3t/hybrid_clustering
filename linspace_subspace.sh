#!/bin/bash

nsearches=2
sigma_start=1
sigma_stop=10

# Calculate the step size
sigma_step=$(bc <<< "scale=10; ($sigma_stop - $sigma_start) / ($nsearches - 1)")

# Generate the array using a loop
for ((j=0; j<nsearches; j++)); do
    sigma_value=$(bc <<< "scale=10; $sigma_start + $j * $sigma_step")
    sbatch run_subspace.sh $sigma_value
done