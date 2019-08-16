dataFolder="dat"
resultsFolder=${dataFolder}/results
resultsFile=${resultsFolder}/results.csv
numThreads=4
mkdir ${resultsFolder}
for ex in $(ls ${dataFolder}/*.json); do
      instanceName=${ex/.json/}
      instanceNumber=${ex//[!0-9]/}
      echo "----------------------\nRunning instance $ex\n----------------------"
      ./EPEC ${ex//.json/} -t ${numThreads} -s 2 -rf ${resultsFile} -r ${resultsFolder}/Solution_${instanceNumber}
      echo "\n\n"
done
