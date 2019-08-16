dataFolder="dat"
resultsFolder=${dataFolder}/results
resultsFile=${resultsFolder}/results.csv
numThreads=4
mkdir ${resultsFolder}
for ex in $(ls ${dataFolder}/*.json); do
      instanceName=${ex/.json/}
      echo "----------------------\nRunning instance $ex\n----------------------"
      ./EPEC ${ex//.json/} -t ${numThreads} -rf ${resultsFile} -r ${resultsFolder}${instanceName//$dataFolder/}
      echo "\n\n"
done
