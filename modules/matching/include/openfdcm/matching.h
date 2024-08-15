#ifndef OPENFDCM_MATCHING_MATCHING_H
#define OPENFDCM_MATCHING_MATCHING_H

#include "openfdcm/matching/featuremap.h"
#include "openfdcm/matching/matchstrategy.h"
#include "openfdcm/matching/optimizestrategy.h"
#include "openfdcm/matching/penaltystrategy.h"
#include "openfdcm/matching/searchstrategy.h"

// featuremaps
#include "openfdcm/matching/featuremaps/dt3cpu.h"

// matchstrategies
#include "openfdcm/matching/matchstrategies/defaultmatch.h"

// optimizestrategies
#include "openfdcm/matching/optimizestrategies/defaultoptimize.h"
#include "openfdcm/matching/optimizestrategies/indulgentoptimize.h"

// penaltystrategies
#include "openfdcm/matching/penaltystrategies/defaultpenalty.h"
#include "openfdcm/matching/penaltystrategies/exponentialpenalty.h"

// searchstrategies
#include "openfdcm/matching/searchstrategies/concentricrange.h"
#include "openfdcm/matching/searchstrategies/defaultsearch.h"

#endif //OPENFDCM_MATCHING_MATCHING_H
