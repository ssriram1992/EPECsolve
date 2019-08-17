dataFolder="dat"
resultsFolder=${dataFolder}/results
logFile=${resultsFolder}/results.csv
numThreads=12
mkdir ${resultsFolder}
for ex in $(ls ${dataFolder}/*.json); do
      instanceName=${ex/.json/}
      instanceNumber=${ex//[!0-9]/}
      echo "----------------------Running instance $ex----------------------"
      ./EPEC -i ${ex//.json/} -t ${numThreads} -w 2 -l ${logFile} -s ${resultsFolder}/Solution_${instanceNumber} --timelimit 360
      printf "\n\n"
done
