/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright 2017  Aetf <aetf@unlimitedcodeworks.xyz>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of
 * the License or (at your option) version 3 or any later version
 * accepted by the membership of KDE e.V. (or its successor approved
 * by the membership of KDE e.V.), which shall act as a proxy
 * defined in Section 14 of version 3 of the license.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "helper.h"

#include "route_guide.grpc.pb.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

using std::string;
using std::stringstream;

namespace routeguide {

string GetDbFileContent(int argc, char **argv)
{
    string db_path;
    string arg_str("--db_path");
    if (argc > 1) {
        string argv_1 = argv[1];
        size_t start_position = argv_1.find(arg_str);
        if (start_position != string::npos) {
            start_position += arg_str.size();
            if (argv_1[start_position] == ' ' || argv_1[start_position] == '=') {
                db_path = argv_1.substr(start_position + 1);
            }
        }
    } else {
        db_path = "route_guide_db.json";
    }
    std::ifstream db_file(db_path);
    if (!db_file.is_open()) {
        std::cout << "Failed to open " << db_path << std::endl;
        return "";
    }
    stringstream db;
    db << db_file.rdbuf();
    return db.str();
}

// A simple parser for the json db file. It requires the db file to have the
// exact form of [{"location": { "latitude": 123, "longitude": 456}, "name":
// "the name can be empty" }, { ... } ... The spaces will be stripped.
class Parser
{
public:
    explicit Parser(const string &db)
        : db_(db)
    {
        // Remove all spaces.
        db_.erase(std::remove_if(db_.begin(), db_.end(), isspace), db_.end());
        if (!Match("[")) {
            SetFailedAndReturnFalse();
        }
    }

    bool Finished() { return current_ >= db_.size(); }

    bool TryParseOne(Feature &feature)
    {
        if (failed_ || Finished() || !Match("{")) {
            return SetFailedAndReturnFalse();
        }
        if (!Match(location_) || !Match("{") || !Match(latitude_)) {
            return SetFailedAndReturnFalse();
        }
        long temp = 0;
        ReadLong(&temp);
        feature.mutable_location()->set_latitude(temp);
        if (!Match(",") || !Match(longitude_)) {
            return SetFailedAndReturnFalse();
        }
        ReadLong(&temp);
        feature.mutable_location()->set_longitude(temp);
        if (!Match("},") || !Match(name_) || !Match("\"")) {
            return SetFailedAndReturnFalse();
        }
        size_t name_start = current_;
        while (current_ != db_.size() && db_[current_++] != '"') {
        }
        if (current_ == db_.size()) {
            return SetFailedAndReturnFalse();
        }
        feature.set_name(db_.substr(name_start, current_ - name_start - 1));
        if (!Match("},")) {
            if (db_[current_ - 1] == ']' && current_ == db_.size()) {
                return true;
            }
            return SetFailedAndReturnFalse();
        }
        return true;
    }

private:
    bool SetFailedAndReturnFalse()
    {
        failed_ = true;
        return false;
    }

    bool Match(const string &prefix)
    {
        bool eq = db_.substr(current_, prefix.size()) == prefix;
        current_ += prefix.size();
        return eq;
    }

    void ReadLong(long *l)
    {
        size_t start = current_;
        while (current_ != db_.size() && db_[current_] != ',' && db_[current_] != '}') {
            current_++;
        }
        // It will throw an exception if fails.
        *l = std::stol(db_.substr(start, current_ - start));
    }

    bool failed_ = false;
    string db_;
    size_t current_ = 0;
    const string location_ = "\"location\":";
    const string latitude_ = "\"latitude\":";
    const string longitude_ = "\"longitude\":";
    const string name_ = "\"name\":";
};

void ParseDb(const string &db, std::vector<Feature> &feature_list)
{
    string db_content(db);
    db_content.erase(std::remove_if(db_content.begin(), db_content.end(), isspace),
                     db_content.end());

    Parser parser(db_content);

    feature_list.clear();
    while (!parser.Finished()) {
        Feature feature;
        if (!parser.TryParseOne(feature)) {
            std::cout << "Error parsing the db file";
            feature_list.clear();
            break;
        }
        feature_list.push_back(feature);
    }
    std::cout << "DB parsed, loaded " << feature_list.size() << " features." << std::endl;
}

} // namespace routeguide
