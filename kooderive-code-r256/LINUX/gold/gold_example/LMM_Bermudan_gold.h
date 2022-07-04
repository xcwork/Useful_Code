

#ifndef LMM_BERMUDAN_GOLD_H
#define LMM_BERMUDAN_GOLD_H

void BermudanExample(bool useLogBasis, 
                     bool useCrossTerms,
                     int pathsPerBatch, 
                     int numberOfBatches,
                     int numberOfExtraRegressions, 
                     int duplicate,
                     bool normalise,
                     int pathsPerSecondPassBatch,
                     int numberSecondPassBatches,
                     int choice,
                     int upperPaths,
                     int upperSeed,
                     int pathsPerSubsim,
                     int numberNonCallDates,
                     int innerEstPaths,
                     int innerSeed,
                     int minPathsToUse);
#endif
