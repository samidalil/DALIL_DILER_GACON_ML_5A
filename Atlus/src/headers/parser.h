#pragma once

#include <string>
#include "MLPData.h"

MLPData* deserialize(const std::string& data);
std::string serialize(const MLPData* model);
