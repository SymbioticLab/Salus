//
// Created by peifeng on 2/23/18.
//

#ifndef SALUS_UTILS_PLUGINS_H
#define SALUS_UTILS_PLUGINS_H

#include <boost/dll/shared_library.hpp>
#include <boost/dll/library_info.hpp>
#include <functional>
#include <vector>

namespace sstl {

/**
 * @brief A simple plugin loader implemented using boost.dll
 *
 */
class PluginLoader
{
public:
    /**
     * @brief Construct the loader
     * @param type The loader will only look for plugins of specific type
     */
    explicit PluginLoader(std::string type);

    /**
     * @brief Discover and load plugins
     *
     * @param pluginDirSuffix suffix
     * @param noDefault Don't search in default paths
     * @return a vector of loaded libraries.
     */
    std::vector<boost::dll::shared_library> discoverPlugins(const std::string &pluginDirSuffix = "plugin", bool noDefault = false);

private:
    using PluginFoundFn = std::function<void (const boost::filesystem::path&, boost::dll::library_info&)>;
    /**
     * @brief Loop through all libraries in @p directory. Only those are loadable libraries are considered.
     *
     * @note The files found are not necessarily loadable by the loader, the only guarantee is they have loadable
     * as determined by boost::dll::library_info
     *
     * If @p dir is relative, appending it as subdirectory for libraries.
     *
     * The default list is
     *  + executable directory (not the current working directory)
     *  + the parent directory of executable directory
     *  + LD_LIBRARY_PATH
     *  + /usr/lib
     *  + /lib
     *
     * @param directory The directory to search. If a relative path is given for @p directory,
     * a default list of paths well be checked with @p directory appended as a subdirectory.
     * If an absolute path is given, only that directory is searched.
     * @param cb callback function for each candidate library
     */
    void forEachPlugin(const boost::filesystem::path &directory, PluginFoundFn cb);

    std::string m_type;
    std::vector<boost::dll::shared_library> m_plugins;
};

} // namespace sstl

#define SSTL_IMPLEMENT_PLUGIN(name, type) \
extern "C" { \
__attribute__((visibility("default"))) \
const char *salus_plugin_name_with_type_##type() { \
    return #name; \
} \
}

#endif // SALUS_UTILS_PLUGINS_H
