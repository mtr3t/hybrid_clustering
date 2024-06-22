#!/bin/sh -l
# FILENAME:  run_spectral

#SBATCH -A cis230279		# Allocation name
#SBATCH --nodes=1         	# Total # of nodes (must be 1 for serial job)
#SBATCH --ntasks=1        	# Total # of MPI tasks (should be 1 for serial job)
#SBATCH --time=00:02:00    	# Total run time limit (hh:mm:ss)
#SBATCH -J spectral		# Job name
#SBATCH -o spectral/spectral.o%j      	# Name of stdout output file
#SBATCH -e spectral/spectral.e%j      	# Name of stderr error file
#SBATCH -p shared  		# Queue (partition) name
#SBATCH --mail-user=mtr3t@mtmail.mtsu.edu
#SBATCH --mail-type=all   	# Send email to above address at begin and end of job

# Manage processing environment, load compilers and applications.
# module purge
module --force purge 		# Unload all loaded modules and reset everything to original state.
module load anaconda
module use $HOME/privatemodules
module load conda-env/mypackages-py3.11.7

module list 			# List currently loaded modules.
hostname 			# Print the hostname of the compute node on which this job is running.

# Launch serial code
python spectral.py $1