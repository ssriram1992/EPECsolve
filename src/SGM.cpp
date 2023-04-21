/* #############################################
 *             This file is part of
 *                    ZERO
 *
 *             Copyright (c) 2020
 *     Released under the Creative Commons
 *         CC BY-NC-SA 4.0 License
 *
 *              Find out more at
 *        https://github.com/ds4dm/ZERO
 * #############################################*/

#include "models.h"
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
namespace fs = std::__fs::filesystem;

int main() {

  GRBEnv GurobiEnv;
  std::string path = "dat/Instances_345/";

  for (const auto &entry : fs::directory_iterator(path)) {
    std::cout << "Loading: " << entry.path() << std::endl;

    if (entry.path().string().find(".json") != std::string::npos) {
      std::string realName =
          std::regex_replace(entry.path().string(), std::regex(".json"), "");
      std::string realInstance =
          std::regex_replace(realName, std::regex("dat/Instances_345/"), "");

      Models::EPECInstance EPECInstance(realName);
      if (EPECInstance.Countries.empty()) {
        std::cerr << "Error: instance is empty\n";
        return 1;
      }
      Models::EPEC EPEC(&GurobiEnv);
      EPEC.setNumThreads(8);

      for (auto &Country : EPECInstance.Countries)
        EPEC.addCountry(Country);
      EPEC.addTranspCosts(EPECInstance.TransportationCosts);
      EPEC.finalize();
      EPEC.setAlgorithm(Game::EPECalgorithm::fullEnumeration);
      std::string basePath = "dat/Instances_345/sgm/" + realInstance;
      EPEC.writePrograms(basePath);
    }
  }

  return 0;
}