#include "lib/StateCollection.h"
#include "lib/StateId.h"
#include "utils/Utils.h"

namespace ceres_nav {

void StateCollection::addState(const std::string &name, double timestamp,
                               std::shared_ptr<ParameterBlockBase> state) {
  int64_t timestamp_key = timestampToKey(timestamp);

  auto it = states_.find(name);
  // If we haven't found this state, create a new entry
  if (it == states_.end()) {
    states_.emplace(name,
                    std::map<int64_t, std::shared_ptr<ParameterBlockBase>>());
  }

  // Check if a state already exists for this name and timestamp
  if (states_.at(name).find(timestamp_key) != states_.at(name).end()) {
    LOG(ERROR) << "State with name: " << name << " and timestamp: " << timestamp
               << " already exists.";
  }

  // Add the state to the map for this name
  states_.at(name).emplace(timestamp_key, state);
}

void StateCollection::addStaticState(
    const std::string &name, std::shared_ptr<ParameterBlockBase> state) {
  auto it = static_states_.find(name);
  if (it != static_states_.end()) {
    LOG(ERROR) << "Static state with name: " << name << " already exists.";
    return;
  }
  static_states_.emplace(name, state);
}

std::shared_ptr<ParameterBlockBase>
StateCollection::getState(const std::string &key, double timestamp) const {
  int64_t timestamp_key = timestampToKey(timestamp);
  auto it1 = states_.find(key);
  if (it1 != states_.end()) {
    auto it2 = it1->second.find(timestamp_key);
    if (it2 != it1->second.end()) {
      return it2->second;
    }
  }
  return nullptr;
}

std::shared_ptr<ParameterBlockBase>
StateCollection::getStaticState(const std::string &key) const {
  auto it = static_states_.find(key);
  if (it != static_states_.end()) {
    return it->second;
  }
  return nullptr;
}

bool StateCollection::hasState(const std::string &key, double timestamp) const {
  int64_t timestamp_key = timestampToKey(timestamp);
  auto it1 = states_.find(key);
  if (it1 != states_.end()) {
    return it1->second.find(timestamp_key) != it1->second.end();
  }
  return false;
}

bool StateCollection::hasStaticState(const std::string &key) const {
  return static_states_.find(key) != static_states_.end();
}

bool StateCollection::hasStateType(const std::string &key) const {
  bool exists = states_.find(key) != states_.end();
  bool static_exists = static_states_.find(key) != static_states_.end();
  return exists || static_exists;
}

void StateCollection::removeState(const std::string &key, double timestamp) {
  int64_t timestamp_key = timestampToKey(timestamp);
  auto it1 = states_.find(key);
  if (it1 != states_.end()) {
    auto it2 = it1->second.find(timestamp_key);
    if (it2 != it1->second.end()) {
      it1->second.erase(it2);
    }
    // If the map is empty, remove the key
    if (it1->second.empty()) {
      states_.erase(it1);
    }
  }
}

void StateCollection::removeStaticState(const std::string &key) {
  auto it = static_states_.find(key);
  if (it != static_states_.end()) {
    static_states_.erase(it);
  }
}

bool StateCollection::getOldestStamp(const std::string &key,
                                     double &timestamp) const {
  auto it = states_.find(key);
  if (it != states_.end() && !it->second.empty()) {
    int64_t timestamp_key = it->second.begin()->first;
    timestamp = keyToTimestamp(timestamp_key);
    return true;
  }
  return false;
}

bool StateCollection::getLatestStamp(const std::string &key,
                                     double &timestamp) const {
  auto it = states_.find(key);
  if (it != states_.end() && !it->second.empty()) {
    int64_t timestamp_key = it->second.rbegin()->first;
    timestamp = keyToTimestamp(timestamp_key);
    return true;
  }
  return false;
}

bool StateCollection::getTimesForState(const std::string &key,
                                       std::vector<double> &times) const {
  auto it = states_.find(key);
  if (it != states_.end()) {
    for (const auto &state : it->second) {
      times.push_back(keyToTimestamp(state.first));
    }
    return true;
  }
  return false;
}

std::shared_ptr<ParameterBlockBase>
StateCollection::getStateByEstimatePointer(double *ptr) const {
  // Loop through all states and check if the pointer matches
  for (auto const &state_map : states_) {
    for (auto const &state : state_map.second) {
      if (state.second->estimatePointer() == ptr) {
        return state.second;
      }
    }
  }

  // Loop through static states as well
  for (auto const &static_state : static_states_) {
    if (static_state.second->estimatePointer() == ptr) {
      return static_state.second;
    }
  }
  return nullptr;
}

bool StateCollection::getStateIDByEstimatePointer(double *ptr,
                                                  StateID &state_id) const {
  for (auto const &state_map_ : states_) {
    for (auto const &state : state_map_.second) {
      if (state.second->estimatePointer() == ptr) {
        state_id = StateID(state_map_.first, keyToTimestamp(state.first));
        return true;
      }
    }
  }
  return false;
}

std::shared_ptr<ParameterBlockBase>
StateCollection::getOldestState(const std::string &key) const {
  auto it = states_.find(key);
  if (it != states_.end() && !it->second.empty()) {
    // Return the first state
    return it->second.begin()->second;
  }
  return nullptr;
}

std::shared_ptr<ParameterBlockBase>
StateCollection::getLatestState(const std::string &key) const {
  auto it = states_.find(key);
  if (it != states_.end() && !it->second.empty()) {
    // Return the last state
    return it->second.rbegin()->second;
  }
  return nullptr;
}

} // namespace ceres_nav