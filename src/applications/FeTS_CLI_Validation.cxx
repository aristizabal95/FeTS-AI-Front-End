#include <map>

#include "cbicaCmdParser.h"
#include "cbicaLogging.h"
#include "cbicaITKSafeImageIO.h"
#include "cbicaUtilities.h"
#include "cbicaITKUtilities.h"
#include "CaPTkGUIUtils.h"

int main(int argc, char** argv)
{
  cbica::CmdParser parser(argc, argv, "FeTS_CLI");

  auto hardcodedNativeModelWeightPath = getCaPTkDataDir() + "/fets";
  auto allArchs = cbica::subdirectoriesInDirectory(hardcodedNativeModelWeightPath);
  std::string allArchsString;
  for (size_t i = 0; i < allArchs.size(); i++)
  {
    allArchsString += allArchs[i] + ",";
  }
  allArchsString.pop_back();

  std::string dataDir, outputDir, loggingDir, fusionMethod = "STAPLE", hardcodedPlanName = "fets_phase2_2";

  parser.addRequiredParameter("d", "dataDir", cbica::Parameter::DIRECTORY, "Dir with Read/Write access", "Input data directory");
  parser.addOptionalParameter("o", "outputDir", cbica::Parameter::DIRECTORY, "Dir with write access", "Location of logging directory");
  parser.addOptionalParameter("g", "gpu", cbica::Parameter::BOOLEAN, "0-1", "Whether to run the process on GPU or not", "Defaults to '0'");

  parser.addApplicationDescription("This is the CLI interface for FeTS");
  parser.addExampleUsage("-d /path/DataForFeTS -o /path/outputDir -g 1", "This command performs inference using the specific models and generates the output to send");
  
  bool gpuRequested = false, trainingRequested = false, patchValidation = true;

  parser.getParameterValue("d", dataDir);
  parser.getParameterValue("o", outputDir);

  return EXIT_SUCCESS;
}