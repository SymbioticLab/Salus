#include "helper.h"
#include "route_guide.grpc.pb.h"

#include <grpc++/grpc++.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::Status;

using routeguide::RouteGuide;
using routeguide::Feature;
using routeguide::Point;
using routeguide::RouteSummary;
using routeguide::RouteNote;

using std::chrono::system_clock;

std::string GetFeatureName(const Point &point, const std::vector<Feature> &feature_list)
{
    for (const Feature &f : feature_list) {
        if (f.location().latitude() == point.latitude()
            && f.location().longitude() == point.longitude()) {
            return f.name();
        }
    }
    return "";
}

float ConvertToRadians(float num)
{
    return num * 3.1415926 / 180;
}

float GetDistance(const Point &start, const Point &end)
{
    const float kCoordFactor = 10000000.0;
    float lat_1 = start.latitude() / kCoordFactor;
    float lat_2 = end.latitude() / kCoordFactor;
    float lon_1 = start.longitude() / kCoordFactor;
    float lon_2 = end.longitude() / kCoordFactor;
    float lat_rad_1 = ConvertToRadians(lat_1);
    float lat_rad_2 = ConvertToRadians(lat_2);
    float delta_lat_rad = ConvertToRadians(lat_2 - lat_1);
    float delta_lon_rad = ConvertToRadians(lon_2 - lon_1);

    float a = pow(sin(delta_lat_rad / 2), 2)
              + cos(lat_rad_1) * cos(lat_rad_2) * pow(sin(delta_lon_rad / 2), 2);
    float c = 2 * atan2(sqrt(a), sqrt(1 - a));
    int R = 6371000; // metres

    return R * c;
}

class RouteGuideImpl final : public RouteGuide::Service
{
public:
    explicit RouteGuideImpl(const std::string &db)
    {
        routeguide::ParseDb(db, m_feature_list);
    }

    Status GetFeature(ServerContext *context, const Point *point, Feature *feature) override
    {
        feature->set_name(GetFeatureName(*point, m_feature_list));
        feature->mutable_location()->CopyFrom(*point);
        return Status::OK;
    }

    Status ListFeatures(ServerContext *context, const routeguide::Rectangle *rectangle,
                        ServerWriter<Feature> *writer) override
    {
        auto lo = rectangle->lo();
        auto hi = rectangle->hi();
        long left = std::min(lo.longitude(), hi.longitude());
        long right = std::max(lo.longitude(), hi.longitude());
        long top = std::max(lo.latitude(), hi.latitude());
        long bottom = std::min(lo.latitude(), hi.latitude());
        for (const Feature &f : m_feature_list) {
            if (f.location().longitude() >= left && f.location().longitude() <= right
                && f.location().latitude() >= bottom && f.location().latitude() <= top) {
                writer->Write(f);
            }
        }
        return Status::OK;
    }

    Status RecordRoute(ServerContext *context, ServerReader<Point> *reader,
                       RouteSummary *summary) override
    {
        Point point;
        int point_count = 0;
        int feature_count = 0;
        float distance = 0.0;
        Point previous;

        system_clock::time_point start_time = system_clock::now();
        while (reader->Read(&point)) {
            point_count++;
            if (!GetFeatureName(point, m_feature_list).empty()) {
                feature_count++;
            }
            if (point_count != 1) {
                distance += GetDistance(previous, point);
            }
            previous = point;
        }
        system_clock::time_point end_time = system_clock::now();
        summary->set_point_count(point_count);
        summary->set_feature_count(feature_count);
        summary->set_distance(static_cast<long>(distance));
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        summary->set_elapsed_time(secs.count());

        return Status::OK;
    }

    Status RouteChat(ServerContext *context,
                     ServerReaderWriter<RouteNote, RouteNote> *stream) override
    {
        std::vector<RouteNote> received_notes;
        RouteNote note;
        while (stream->Read(&note)) {
            for (const RouteNote &n : received_notes) {
                if (n.location().latitude() == note.location().latitude()
                    && n.location().longitude() == note.location().longitude()) {
                    stream->Write(n);
                }
            }
            received_notes.push_back(note);
        }

        return Status::OK;
    }

private:
    std::vector<Feature> m_feature_list;
};

void RunServer(const std::string &db_path)
{
    std::string server_address("0.0.0.0:50051");
    RouteGuideImpl service(db_path);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}

int main(int argc, char **argv)
{
    // Expect only arg: --db_path=path/to/route_guide_db.json.
    std::string db = routeguide::GetDbFileContent(argc, argv);
    RunServer(db);

    return 0;
}
