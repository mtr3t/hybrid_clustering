#!/bin/bash

start=0.001
stop=0.1
num=20

# Calculate the step size
step=$(bc <<< "scale=10; ($stop - $start) / ($num - 1)")

# Generate the array using a loop
for ((i=0; i<num; i++)); do
    value=$(bc <<< "scale=10; $start + $i * $step")
    echo "$value"
done

