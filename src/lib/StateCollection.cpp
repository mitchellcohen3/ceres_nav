#include "lib/StateCollection.h"
#include "lib/StateId.h"
#include "utils/Utils.h"

#include <glog/logging.h>

namespace ceres_nav {
bool StateCollection::addState(const StateID &id,
                               std::shared_ptr<ParameterBlockBase> state) {
  // Add the time-varying state to the map
  if (id.hasTimestamp()) {

    auto it = time_varying_states_.find(id.key());
    // If we haven't found this key yet, create a new amp for it
    if (it == time_varying_states_.end()) {
      time_varying_states_.emplace(
          id.key(), std::map<double, std::shared_ptr<ParameterBlockBase>>());
    }

    if (time_varying_states_.at(id.key()).find(id.timestamp().value()) !=
        time_varying_states_.at(id.key()).end()) {
      LOG(ERROR) << "State already exists for key: " << id.key()
                 << " at timestamp: " << id.timestamp().value();
      return false;
    }

    time_varying_states_.at(id.key()).emplace(id.timestamp().value(), state);
  } else {
    // Static state
    if (static_states_.find(id.key()) != static_states_.end()) {
      LOG(ERROR) << "State already exists for key: " << id.key();
      return false;
    }
    static_states_.emplace(id.key(), state);
  }

  return true;
}

bool StateCollection::removeState(const StateID &id) {
  if (id.hasTimestamp()) {
    auto it = time_varying_states_.find(id.key());
    if (it != time_varying_states_.end()) {
      return it->second.erase(id.timestamp().value()) > 0;
    }
    return false;
  } else {
    return static_states_.erase(id.key()) > 0;
  }
}

std::shared_ptr<ParameterBlockBase>
StateCollection::getState(const StateID &id) const {
  if (id.hasTimestamp()) {
    auto it = time_varying_states_.find(id.key());
    if (it != time_varying_states_.end()) {
      auto state_it = it->second.find(id.timestamp().value());
      if (state_it != it->second.end()) {
        return state_it->second;
      }
    }
    return nullptr;
  } else {
    auto it = static_states_.find(id.key());
    if (it != static_states_.end()) {
      return it->second;
    }
    return nullptr;
  }
}

bool StateCollection::hasState(const StateID &state_id) const {
  if (state_id.hasTimestamp()) {
    auto it = time_varying_states_.find(state_id.key());
    if (it != time_varying_states_.end()) {
      auto state_it = it->second.find(state_id.timestamp().value());
      return state_it != it->second.end();
    }
    return false;
  } else {
    return static_states_.find(state_id.key()) != static_states_.end();
  }
}

bool StateCollection::getOldestStamp(const std::string &key,
                                     double &timestamp) const {
  auto it = time_varying_states_.find(key);
  if (it != time_varying_states_.end() && !it->second.empty()) {
    timestamp = it->second.begin()->first;
    return true;
  }
  return false;
}
bool StateCollection::getLatestStamp(const std::string &key,
                                     double &timestamp) const {
  auto it = time_varying_states_.find(key);
  if (it != time_varying_states_.end() && !it->second.empty()) {
    timestamp = it->second.rbegin()->first;
    return true;
  }
  return false;
}

bool StateCollection::getTimesForState(const std::string &key,
                                       std::vector<double> &timestamps) const {
  auto it = time_varying_states_.find(key);
  if (it != time_varying_states_.end() && !it->second.empty()) {
    timestamps.clear();
    for (const auto &pair : it->second) {
      timestamps.push_back(pair.first);
    }
    return true;
  }
  return false;
}

std::shared_ptr<ParameterBlockBase>
StateCollection::getOldestState(const std::string &key) const {
  auto it = time_varying_states_.find(key);
  if (it != time_varying_states_.end() && !it->second.empty()) {
    // Return the first state
    return it->second.begin()->second;
  }
  return nullptr;
}

std::shared_ptr<ParameterBlockBase>
StateCollection::getLatestState(const std::string &key) const {
  auto it = time_varying_states_.find(key);
  if (it != time_varying_states_.end() && !it->second.empty()) {
    // Return the last state
    return it->second.rbegin()->second;
  }
  return nullptr;
}

std::shared_ptr<ParameterBlockBase>
StateCollection::getStateByEstimatePointer(double *ptr) const {
  // Check time-varying states
  for (const auto &type_pair : time_varying_states_) {
    for (const auto &time_pair : type_pair.second) {
      if (time_pair.second->estimatePointer() == ptr) {
        return time_pair.second;
      }
    }
  }

  // Check static states
  for (const auto &static_pair : static_states_) {
    if (static_pair.second->estimatePointer() == ptr) {
      return static_pair.second;
    }
  }

  return nullptr;
}

bool StateCollection::getStateIDByEstimatePointer(double *ptr,
                                                  StateID &state_id) const {
  // Check time-varying states
  for (const auto &type_pair : time_varying_states_) {
    for (const auto &time_pair : type_pair.second) {
      if (time_pair.second->estimatePointer() == ptr) {
        state_id = StateID(type_pair.first, time_pair.first);
        return true;
      }
    }
  }

  // Check static states
  for (const auto &static_pair : static_states_) {
    if (static_pair.second->estimatePointer() == ptr) {
      state_id = StateID(static_pair.first);
      return true;
    }
  }

  return false;
}

} // namespace ceres_nav