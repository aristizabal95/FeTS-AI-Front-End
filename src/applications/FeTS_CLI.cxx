#include "cbicaCmdParser.h"
#include "cbicaLogging.h"
#include "cbicaITKSafeImageIO.h"
#include "cbicaUtilities.h"
#include "cbicaITKUtilities.h"
#include "CaPTkGUIUtils.h"

#include "itkMaskImageFilter.h"

#include <map>

int main(int argc, char** argv)
{
  cbica::CmdParser parser(argc, argv, "FeTS_CLI");

  auto allArchs = cbica::subdirectoriesInDirectory(getCaPTkDataDir() + "/fets");
  std::string allArchsString;
  for (size_t i = 0; i < allArchs.size(); i++)
  {
    allArchsString += allArchs[i] + ",";
  }
  allArchsString.pop_back();

  std::string dataDir, modelName, loggingDir, colName, archs, fusionMethod = "STAPLE";

  parser.addRequiredParameter("d", "dataDir", cbica::Parameter::DIRECTORY, "Dir with Read/Write access", "Input data directory");
  parser.addRequiredParameter("m", "modelName", cbica::Parameter::FILE, "Model file", "Input model weights file");
  parser.addRequiredParameter("t", "training", cbica::Parameter::BOOLEAN, "0 or 1", "Whether performing training or inference", "1==Train and 0==Inference");
  parser.addRequiredParameter("L", "LoggingDir", cbica::Parameter::DIRECTORY, "Dir with write access", "Location of logging directory");
  parser.addRequiredParameter("a", "archs", cbica::Parameter::STRING, allArchsString, "The architecture(s) to infer/train on", "Only a single architecture is supported for training", "Comma-separated values for multiple options");
  parser.addOptionalParameter("lF", "labelFuse", cbica::Parameter::STRING, "STAPLE,ITKVoting,SIMPLE,MajorityVoting", "The label fusion strategy to follow for multi-arch inference", "Defaults to: " + fusionMethod);
  parser.addOptionalParameter("g", "gpu", cbica::Parameter::BOOLEAN, "0-1", "Whether to run the process on GPU or not", "Defaults to '0'");
  parser.addOptionalParameter("c", "colName", cbica::Parameter::STRING, "", "Common name of collaborator", "Required for training");
  
  bool gpuRequested = false;
  bool trainingRequested = false;

  parser.getParameterValue("d", dataDir);
  parser.getParameterValue("m", modelName);
  parser.getParameterValue("L", loggingDir);
  parser.getParameterValue("a", archs);
  parser.getParameterValue("t", trainingRequested);


  if (trainingRequested)
  {
    if (parser.isPresent("c"))
    {
      parser.getParameterValue("c", colName);
    }
    else
    {
      std::cerr << "Collaborator name is required to beging training; please specify this using '-c'.\n";
      return EXIT_FAILURE;
    }
  }
  if (parser.isPresent("g"))
  {
    parser.getParameterValue("g", gpuRequested);
  }
  if (parser.isPresent("lF"))
  {
    parser.getParameterValue("lF", fusionMethod);
  }

  // convert everything to lower-case for easier comparison
  std::transform(archs.begin(), archs.end(), archs.begin(), ::tolower);
  std::transform(fusionMethod.begin(), fusionMethod.end(), fusionMethod.begin(), ::tolower);

  auto fetsApplicationPath = cbica::getExecutablePath();
  auto deepMedicExe = getApplicationPath("DeepMedic");

  auto archs_split = cbica::stringSplit(archs, ",");
  auto fusion_split = cbica::stringSplit(fusionMethod, ",");

  auto subjectDirs = cbica::subdirectoriesInDirectory(dataDir);

  if (trainingRequested && (archs_split.size() > 1))
  {
    std::cerr << "Training cannot be currently be performed on more than 1 architecture.\n";
    return EXIT_FAILURE;
  }

  std::string hardcodedPlanName,
    hardcodedOpenFLPath = fetsApplicationPath + "/OpenFederatedLearning/",
    hardcodedLabelFusionPath = fetsApplicationPath + "/LabelFusion/label_fusion",
    hardcodedModelWeightPath = hardcodedOpenFLPath + "/bin/federations/weights/", // start with the common location
    hardcodedPythonPath = hardcodedOpenFLPath + "/venv/bin/python"; // this needs to change for Windows (wonder what happens for macOS?)

  auto pythonEnvironmentFound = false;
  if (cbica::isFile(hardcodedPythonPath))
  {
    pythonEnvironmentFound = true;
  }

  if (!trainingRequested)
  {
    std::string subjectsWithMissingModalities, subjectsWithErrors; // string to store error cases
    for (size_t s = 0; s < subjectDirs.size(); s++) // iterate through all subjects
    {
      auto currentSubjectIsProblematic = false;
      std::string file_t1gd, file_t1, file_t2, file_flair;
      auto fileToCheck = dataDir + "/" + subjectDirs[s] + "/brain_t1gd.nii.gz";
      if (cbica::fileExists(fileToCheck))
      {
        file_t1gd = fileToCheck;
      }
      else
      {
        subjectsWithMissingModalities += subjectDirs[s] + ",";
        currentSubjectIsProblematic = true;
      }

      fileToCheck = dataDir + "/" + subjectDirs[s] + "/brain_t1.nii.gz";
      if (cbica::fileExists(fileToCheck))
      {
        file_t1 = fileToCheck;
      }
      else
      {
        subjectsWithMissingModalities += subjectDirs[s] + ",";
        currentSubjectIsProblematic = true;
      }
      fileToCheck = dataDir + "/" + subjectDirs[s] + "/brain_t2.nii.gz";
      if (cbica::fileExists(fileToCheck))
      {
        file_t2 = fileToCheck;
      }
      else
      {
        subjectsWithMissingModalities += subjectDirs[s] + ",";
        currentSubjectIsProblematic = true;
      }
      fileToCheck = dataDir + "/" + subjectDirs[s] + "/brain_flair.nii.gz";
      if (cbica::fileExists(fileToCheck))
      {
        file_flair = fileToCheck;
      }
      else
      {
        subjectsWithMissingModalities += subjectDirs[s] + ",";
        currentSubjectIsProblematic = true;
      }

      if (!currentSubjectIsProblematic) // proceed only if all modalities for the current subject are present
      {
        for (size_t i = 0; i < archs_split.size(); i++) // iterate through all requested architectures
        {
          if (archs_split[i] == "deepmedic") // special case 
          {
            auto brainMaskFile = dataDir + "/" + subjectDirs[s] + "/deepmedic_seg.nii.gz";

            auto fullCommand = deepMedicExe + " -md " + getCaPTkDataDir() + "/fets/deepMedic/saved_models/brainTumorSegmentation/ " +
              "-i " + file_t1 + "," +
              file_t1gd + "," +
              file_t2 + "," +
              file_flair + " -o " +
              brainMaskFile;

            if (std::system(fullCommand.c_str()) != 0)
            {
              subjectsWithErrors += subjectDirs[s] + ",";
            }
          } // deepmedic check
          else
          {
            if (pythonEnvironmentFound)
            {
              // check for all other models written in pytorch here
              auto fullCommandToRun = hardcodedPythonPath + " " + hardcodedOpenFLPath + "/bin/run_inference_from_flplan.py";
              std::string args = "";

              // check between different architectures
              if (archs_split[i] == "3dunet")
              {
                // this is currently not defined
              }
              else if (archs_split[i] == "3dresunet")
              {
                hardcodedPlanName = "pt_3dresunet_brainmagebrats";
                auto hardcodedModelName = hardcodedPlanName + "_best.pbuf";
                auto allGood = true;
                if (!cbica::isFile((hardcodedModelWeightPath + "/" + hardcodedModelName))) // in case the "best" model is not present, use the "init" model that is distributed with FeTS installation
                {
                  auto hardcodedModelName = hardcodedPlanName + "_init.pbuf";
                  if (!cbica::isFile((hardcodedModelWeightPath + "/" + hardcodedModelName)))
                  {
                    std::cerr << "A compatible model weight file for the architecture '" << archs_split[i] << "' was not found. Please contact admin@fets.ai for help.\n";
                    allGood = false;
                  }
                }

                args += "-mwf " + hardcodedModelName
                  + " -p " + hardcodedPlanName + ".yaml"
                  //<< "-mwf" << hardcodedModelWeightPath // todo: doing customized solution above - change after model weights are using full paths for all
                  + " -d " + dataDir
                  + " -inference_patient " + subjectDirs[s]
                  + " -ld " + loggingDir;

                args += " -md ";
                if (gpuRequested)
                {
                  args += "cuda";
                }
                else
                {
                  args += "cpu";
                }

                if (std::system((fullCommandToRun + " " + args).c_str()) != 0)
                {
                  std::cerr << "Couldn't complete the requested task.\n";
                  return EXIT_FAILURE;
                }

              }
              else if (archs_split[i] == "nnunet")
              {
                // structure according to what is needed - might need to create a function that can call run_inference_from_flplan for different hardcodedModelName
              }
            } // end of python check
          } // end of non-DM archs check
        } // end of archs_split

        /// fusion 
        if (pythonEnvironmentFound)
        {
          if (cbica::isFile(hardcodedLabelFusionPath))
          {
            auto filesInSubjectDir = cbica::filesInDirectory(dataDir + "/" + subjectDirs[s]);
            auto labelFusion_command = hardcodedPythonPath + " " + hardcodedLabelFusionPath + " ";
            std::string filesForFusion;
            for (size_t f = 0; f < filesInSubjectDir.size(); f++)
            {
              if (filesInSubjectDir[f].find("_seg.nii.gz") != std::string::npos) // find all files that have "_seg.nii.gz" in file name
              {
                if (filesInSubjectDir[f].find("final") == std::string::npos) // only do fusion for the files where "final" is not present
                {
                  filesForFusion += filesInSubjectDir[f] + ",";
                }
              }
            } // files loop in subject directory
            filesForFusion.pop_back(); // remove last ","

            for (size_t f = 0; f < fusion_split.size(); f++)
            {
              auto final_fused_file = dataDir + "/" + subjectDirs[s] + "/fused_" + fusion_split[f] + "_seg.nii.gz";
              auto full_fusion_command = labelFusion_command + "-inputs " + filesForFusion + " -classes 0,1,2,4 " // this needs to change after different segmentation algorithms are put in place
                + " -method " + fusion_split[f] + " -output " + final_fused_file;
              if (std::system(full_fusion_command.c_str()) != 0)
              {
                std::cerr << "Something went wrong with fusion for subject '" << subjectDirs[s] << "' using fusion method '" << fusion_split[f] << "'\n";
              }
            }
          } // end of label fusion script check
        } // end of python check
        /*
        
python fusion_run \
-inputs /path/to/seg_algo_1.nii.gz,/path/to/seg_algo_2.nii.gz,/path/to/seg_algo_3.nii.gz \
-classes 0,1,2,4 \
-method STAPLE \
-output /path/to/seg_fusion.nii.gz
        */


      } // end of currentSubjectIsProblematic 
    } // end of subjectDirs
  } // end of trainingRequested check

  std::string specialArgs, args, hardcodedModelName;
  if (trainingRequested)
  {
    specialArgs = "-col " + colName;
  }
  if (modelName.find("_3dresunet_ss") != std::string::npos) // let's not worry about skull-stripping right now
  {
    hardcodedPlanName = "pt_3dresunet_ss_brainmagebrats";
    hardcodedModelName = hardcodedModelWeightPath + hardcodedPlanName + "_best.pt"; // taken from https://github.com/FETS-AI/Models/blob/master/skullstripping/3dresunet/pt_3dresunet_ss_brainmagebrats_best.pt
    if (!trainingRequested)
    {
      specialArgs += "-nmwf " + hardcodedModelName;
    }
  }
  else
  {
    hardcodedPlanName = "pt_3dresunet_brainmagebrats";
    auto hardcodedModelName = hardcodedPlanName + "_best.pbuf";
    if (!cbica::isFile((hardcodedModelWeightPath + "/" + hardcodedModelName)))
    {
      auto hardcodedModelName = hardcodedPlanName + "_init.pbuf";
      if (!cbica::isFile((hardcodedModelWeightPath + "/" + hardcodedModelName)))
      {
        std::cerr << "A compatible model weight file was not found. Please contact admin@fets.ai for help.\n";
        return EXIT_FAILURE;
      }
    }
    if (!trainingRequested)
    {
      specialArgs += "-mwf " + hardcodedModelName;
    }
  }

  // sanity checks
  //if (!cbica::isFile(hardcodedModelWeightPath.toStdString())) // todo: renable after model weights are using full paths for all
  //{
  //  ShowErrorMessage("The requested inference model was not found (it needs to be in ${FeTS_installDir}/bin/OpenFederatedLearning/bin/federations/weights/${planName}_best.pbuf");
  //  return;
  //}
  if (!cbica::isFile(hardcodedPythonPath))
  {
    std::cerr << "The python virtual environment was not found, please refer to documentation to initialize it.\n";
    return EXIT_FAILURE;
  }

  std::string fullCommandToRun = hardcodedPythonPath + " " + fetsApplicationPath;
  if (trainingRequested)
  {
    fullCommandToRun += "/OpenFederatedLearning/bin/run_inference_from_flplan.py";
  }
  else
  {
    fullCommandToRun += "/OpenFederatedLearning/bin/run_collaborator_from_flplan.py";
  }

  args += " -p " + hardcodedPlanName + ".yaml"
    //<< "-mwf" << hardcodedModelWeightPath // todo: doing customized solution above - change after model weights are using full paths for all
    + " -d " + dataDir
    + " -ld " + loggingDir;

  args += " -md ";
  if (gpuRequested)
  {
    args += "cuda";
  }
  else
  {
    args += "cpu";
  }

  if (std::system((fullCommandToRun + " " + args + " " + specialArgs).c_str()) != 0)
  {
    std::cerr << "Couldn't complete the requested task.\n";
    return EXIT_FAILURE;
  }
    
  std::cout << "Finished.\n";

  return EXIT_SUCCESS;
}


