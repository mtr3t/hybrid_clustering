#!/bin/bash

ntopco=1
nsearches=2
sigma_start=1
sigma_stop=10
gamma_start=1
gamma_stop=10

# Calculate the step size
sigma_step=$(bc <<< "scale=10; ($sigma_stop - $sigma_start) / ($nsearches - 1)")
gamma_step=$(bc <<< "scale=10; ($gamma_stop - $gamma_start) / ($nsearches - 1)")

# Generate the array using a loop
for ((i=0; i<ntopco; i++)); do
    for ((j=0; j<nsearches; j++)); do
        sigma_value=$(bc <<< "scale=10; $sigma_start + $j * $sigma_step")
        gamma_value=$(bc <<< "scale=10; $gamma_start + $j * $gamma_step")
        sbatch run_hybrid.sh $sigma_value $gamma_value $i
    done
done