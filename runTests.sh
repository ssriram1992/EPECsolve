dataFolder="dat"
instanceFolder=${dataFolder}/Instance_345_Harder
resultsFolder=${dataFolder}/results_Instance_345_Harder
logFile=${resultsFolder}/results.csv
numThreads=8
mkdir ${resultsFolder}
for ex in $(ls ${instanceFolder}/*.json); do
      instanceName=${ex/.json/}
      instanceNumber=${ex//[!0-9]/}
      echo "----------------------Running instance $ex----------------------"
      #Skip full enumeration with the first argument
      if [ ! $1 ]; then
        echo "--------Full Enumeration"
        ./EPEC -i ${ex//.json/} -t ${numThreads} -w 2 -l ${logFile} -s ${resultsFolder}/Solution_Full_${instanceNumber} --timelimit 1800 -a 0
      fi
      for aggressiveness in {1,3,5}; do
        for addPolyMethod in {0,1,2}; do
          echo "--------Inner Approximation"
          ./EPEC -i ${ex//.json/} -t ${numThreads} -w 2 -l ${logFile} -s ${resultsFolder}/Solution_Inner_${instanceNumber}_${aggressiveness}_${addPolyMethod} --timelimit 1800 -a 1 --aggr $aggressiveness --add $addPolyMethod
        done
      done
      printf "\n\n"
done
