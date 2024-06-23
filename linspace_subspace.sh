#!/bin/bash

ntopco=2
nsearches=2
gamma_start=1
gamma_stop=10

# Calculate the step size
gamma_step=$(bc <<< "scale=10; ($gamma_stop - $gamma_start) / ($nsearches - 1)")

# Generate the array using a loop
for ((i=0; i<ntopco; i++)); do
    for ((j=0; j<nsearches; j++)); do
        gamma_value=$(bc <<< "scale=10; $gamma_start + $j * $gamma_step")
        sbatch run_subspace.sh $gamma_value $i
    done
done
