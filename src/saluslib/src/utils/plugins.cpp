//
// Created by peifeng on 2/23/18.
//

#include "utils/plugins.h"

#include "utils/envutils.h"
#include "utils/stringutils.h"
#include "platform/logging.h"
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/dll/library_info.hpp>
#include <boost/filesystem/path.hpp>
#include <utility>
#include <string_view>

namespace fs = boost::filesystem;
namespace dll = boost::dll;
using namespace std::literals::string_view_literals;

namespace sstl {

PluginLoader::PluginLoader(std::string type)
    : m_type(std::move(type))
{
}

void PluginLoader::forEachPlugin(const boost::filesystem::path &directory, PluginFoundFn cb)
{
    DCHECK(cb);

    std::vector<fs::path> dirsToCheck;
    if (directory.is_absolute()) {
        dirsToCheck.emplace_back(directory);
    } else {
        /*
         * The list to check:
         *  + executable directory (not the current working directory)
         *  + executable directory /lib
         *  + the parent directory of executable directory
         *  + the parent directory of executable directory /lib
         *  + LD_LIBRARY_PATH
         *  + /usr/lib
         *  + /lib
         */

        // executable directory (not the current working directory)
        auto self = dll::program_location().remove_filename();
        dirsToCheck.emplace_back(self);
        dirsToCheck.emplace_back(self / "lib");

        // the parent directory of executable directory
        if (auto pself = self.parent_path(); !pself.empty()) {
            dirsToCheck.emplace_back(self.parent_path());
            dirsToCheck.emplace_back(self.parent_path() / "lib");
        }

        // LD_LIBRARY_PATH
        if (auto ldLibraryPath = sstl::fromEnvVar("LD_LIBRARY_PATH", ""sv); !ldLibraryPath.empty()) {
            for (const auto &p : sstl::splitsv(ldLibraryPath, ";")) {
                dirsToCheck.emplace_back(std::string(p));
            }
        }

        // hard-coded paths
        dirsToCheck.emplace_back("/usr/lib");
        dirsToCheck.emplace_back("/lib");

        // append suffix
        for (auto &p : dirsToCheck) {
            p /= directory;
        }
    }

    for (const auto &dir : dirsToCheck) {
        if (!fs::is_directory(dir)) {
            VLOG(2) << "Skip non exist directory: " << dir;
            continue;
        }
        VLOG(2) << "Try find plugin in directory: " << dir;
        for (const auto &p : fs::directory_iterator(dir)) {
            if (!fs::is_regular_file(p)) {
                continue;
            }
            VLOG(2) << "Found candidate file " << p;
            try {
                dll::library_info li(p);
                cb(fs::absolute(p), li);
            } catch (const std::runtime_error &ex) {
                // ignore
                VLOG(3) << "Ignore candidate file " << p << " due to error: " << ex.what();
            }
        }
    }
}

std::vector<dll::shared_library> PluginLoader::discoverPlugins(const std::string &pluginDirSuffix, bool noDefault)
{
    fs::path suffix(pluginDirSuffix);
    if (noDefault) {
        if (suffix.empty()) {
            // nothing to search
            return {};
        }
        suffix = fs::absolute(suffix);
    }

    std::vector<dll::shared_library> ret;
    forEachPlugin(suffix, [this, &ret](const auto &absPath, auto &li) {
        const auto &symbols = li.symbols();
        if (std::find(symbols.begin(), symbols.end(), "salus_plugin_name_with_type_" + m_type) != symbols.end()) {
            LOG(INFO) << "Loading plugin of type " << m_type << " from " << absPath;
            ret.emplace_back(absPath, dll::load_mode::rtld_lazy | dll::load_mode::rtld_local | dll::load_mode::rtld_deepbind);
        }
    });

    m_plugins.reserve(m_plugins.size() + ret.size());
    m_plugins.insert(m_plugins.end(), ret.begin(), ret.end());
    return ret;
}

} // namespace sstl
