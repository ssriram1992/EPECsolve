dataFolder="dat"
instanceFolder=${dataFolder}/Instances_345
resultsFolder=${dataFolder}/results_bounding
logFile=${resultsFolder}/results.csv
numThreads=8
mkdir ${resultsFolder}
while IFS= read -r instance; do
  ./EPEC -i ${instanceFolder}/${instance} -t ${numThreads} -w 2 -l ${logFile} -s ${resultsFolder}/Solution_Inner_${instance}_3_methodsequential_bounded --timelimit 1800 -a 1 --aggr 3 --add 0 --bound 1 --boundBigM 1e8
  ./EPEC -i ${instanceFolder}/${instance} -t ${numThreads} -w 2 -l ${logFile} -s ${resultsFolder}/Solution_Inner_${instance}_3_methodsequential_nobounded --timelimit 1800 -a 1 --aggr 3 --add 0 --bound 0 --boundBigM 1e8
done <${dataFolder}/selective.txt
