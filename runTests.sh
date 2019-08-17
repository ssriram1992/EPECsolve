dataFolder="dat"
resultsFolder=${dataFolder}/results
resultsFile=${resultsFolder}/results.csv
numThreads=12
mkdir ${resultsFolder}
for ex in $(ls ${dataFolder}/*.json); do
      instanceName=${ex/.json/}
      instanceNumber=${ex//[!0-9]/}
      echo "----------------------Running instance $ex----------------------"
      ./EPEC ${ex//.json/} -t ${numThreads} -s 2 -rf ${resultsFile} -r ${resultsFolder}/Solution_${instanceNumber} -tl 360
      printf "\n\n"
done