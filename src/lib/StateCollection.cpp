#include "lib/StateCollection.h"
#include "utils/Utils.h"

void StateCollection::addState(const std::string &name, double timestamp,
                               std::shared_ptr<ParameterBlockBase> state) {
  int64_t timestamp_key = timestampToKey(timestamp);

  auto it = states_.find(name);
  // If we haven't found this state, create a new entry
  if (it == states_.end()) {
    states_.emplace(name,
                    std::map<int64_t, std::shared_ptr<ParameterBlockBase>>());
  }
  // Add the state to the map for this name
  states_.at(name).emplace(timestamp_key, state);
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

bool StateCollection::hasState(const std::string &key, double timestamp) const {
  int64_t timestamp_key = timestampToKey(timestamp);
  auto it1 = states_.find(key);
  if (it1 != states_.end()) {
    return it1->second.find(timestamp_key) != it1->second.end();
  }
  return false;
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

std::shared_ptr<ParameterBlockBase> StateCollection::getStateByEstimatePointer(double *ptr) const {
  // Loop through all states and check if the pointer matches
  for (auto const &state_map : states_) {
    for (auto const &state : state_map.second) {
      if (state.second->estimatePointer() == ptr) {
        return state.second;
      }
    }
  }
  return nullptr;
}