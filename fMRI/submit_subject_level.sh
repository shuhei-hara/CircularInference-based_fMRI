# ===== SLURM Job Options =====
#SBATCH --job-name=your_job_name         
#SBATCH --partition=your_partition_name  
#SBATCH --array=1-3                      # the number of subjects                     
#SBATCH -n 1                             
#SBATCH --cpus-per-task=4                
#SBATCH --mem-per-cpu=32G                
#SBATCH --time=estimated_time                   
#SBATCH --mail-type=ALL                  
#SBATCH -o log/output-%A-%a.txt   

DIR_BIDS= path_to_BIDS
DIR_PREP= path_to_fMRIprep
DIR_OUTPUT= path_to_output
DIR_SCRIPT= path_to_current_directory

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" path_to_BIDS/participants.tsv) # Take the subject ID from participants.tsv based on the SLURM array
cmd='python3 ${DIR_SCRIPT}/model_based_regression.py ${DIR_PREP} ${DIR_OUTPUT} run --bids-dir ${DIR_BIDS} --space template --participant-label ${subject}'

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode