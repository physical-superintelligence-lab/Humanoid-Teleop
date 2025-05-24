#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <vector>
#include <cstring>
#include <cmath> 
#include <csignal>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/compression/octree_pointcloud_compression.h>
#include <zlib.h>

#include "livox_lidar_api.h"
#include "livox_lidar_def.h"

namespace fs = std::filesystem;

struct Point {
    float x;
    float y;
    float z;
};

std::vector<Point> cloud;
std::mutex cloud_mutex;

volatile uint32_t g_lidar_handle = 0;
volatile std::sig_atomic_t g_stop = 0;

// Signal handler for SIGINT (Ctrl+C)
void SignalHandler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nSIGINT received. Stopping recording..." << std::endl;
        g_stop = 1;
    }
}

std::string GetTimestampFilepath(std::string output_dir) {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;


    double time = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    std::string timeStr = std::to_string(time);
 

    // Build filepath with milliseconds appended.
    std::stringstream ss;
    ss << timeStr << ".pcd";
    return output_dir + ss.str();
}

// New helper: Save a given vector of points to an ASCII PCD file.
void SavePointsToPCD(const std::vector<Point>& points, const std::string& filepath) {

    fs::path dir = fs::path(filepath).parent_path();

    if (!fs::exists(dir)) {
        if (!fs::create_directories(dir)) {
            std::cerr << "Failed to create directory: " << dir << std::endl;
            return;
        }
    }

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return;
    }
    ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
    ofs << "VERSION 0.7\n";
    ofs << "FIELDS x y z\n";
    ofs << "SIZE 4 4 4\n";
    ofs << "TYPE F F F\n";
    ofs << "COUNT 1 1 1\n";
    ofs << "WIDTH " << points.size() << "\n";
    ofs << "HEIGHT 1\n";
    ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
    ofs << "POINTS " << points.size() << "\n";
    ofs << "DATA ascii\n";

    for (const auto& point : points) {
        ofs << point.x << " " << point.y << " " << point.z << "\n";
    }
    ofs.close();
    /*std::cout << "Saved " << points.size() << " points to " << filepath << std::endl;*/
}

void SavePointsToBin(const std::vector<Point>& points, const std::string& filepath) {
    fs::path dir = fs::path(filepath).parent_path();

    if (!fs::exists(dir)) {
        if (!fs::create_directories(dir)) {
            std::cerr << "Failed to create directory: " << dir << std::endl;
            return;
        }
    }

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return;
    }
    // Create a PCL point cloud.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_out->width = static_cast<uint32_t>(points.size());
    cloud_out->height = 1;
    cloud_out->is_dense = true;
    cloud_out->points.resize(points.size());

    // Copy data from your vector into the PCL cloud.
    for (size_t i = 0; i < points.size(); ++i) {
        cloud_out->points[i].x = points[i].x;
        cloud_out->points[i].y = points[i].y;
        cloud_out->points[i].z = points[i].z;
    }

    // Save the cloud in binary format.
    if (pcl::io::savePCDFileBinary(filepath, *cloud_out) == -1) {
        std::cerr << "Error saving PCD file: " << filepath << std::endl;
    } else {
        std::cout << "Saved " << points.size() << " points to " << filepath << std::endl;
    }
}

void PointCloudCallback(const uint32_t handle, const uint8_t dev_type, 
                        LivoxLidarEthernetPacket* packet, void* client_data) {
    if (g_lidar_handle == 0) {
        g_lidar_handle = handle;
        std::cout << "Discovered LiDAR handle: " << g_lidar_handle << std::endl;
    }

    /*std::cout << "Received packet from LiDAR handle: " << handle*/
    /*          << ", dev_type: " << static_cast<int>(dev_type) << std::endl;*/
    
    if (!packet || !packet->data) {
        std::cerr << "Packet data is null!" << std::endl;
        return;
    }
    
    uint16_t num_points = packet->dot_num;
    
    // Process the point cloud data based on the point data type.
    // Lock the mutex while appending points.
    std::lock_guard<std::mutex> lock(cloud_mutex);
    if (packet->data_type == kLivoxLidarCartesianCoordinateHighData) {
        // High precision Cartesian: points in mm.
        LivoxLidarCartesianHighRawPoint* points = 
            reinterpret_cast<LivoxLidarCartesianHighRawPoint*>(packet->data);
        for (uint16_t i = 0; i < num_points; ++i) {
            Point point;
            point.x = static_cast<float>(points[i].x);
            point.y = static_cast<float>(points[i].y);
            point.z = static_cast<float>(points[i].z);
            cloud.push_back(point);
        }
    }
    else if (packet->data_type == kLivoxLidarCartesianCoordinateLowData) {
        // Low precision Cartesian: points in cm.
        LivoxLidarCartesianLowRawPoint* points = 
            reinterpret_cast<LivoxLidarCartesianLowRawPoint*>(packet->data);
        for (uint16_t i = 0; i < num_points; ++i) {
            Point point;
            point.x = static_cast<float>(points[i].x);
            point.y = static_cast<float>(points[i].y);
            point.z = static_cast<float>(points[i].z);
            cloud.push_back(point);
        }
    }
    else if (packet->data_type == kLivoxLidarSphericalCoordinateData) {
        // Spherical coordinates: conversion to Cartesian is needed.
        LivoxLidarSpherPoint* points = 
            reinterpret_cast<LivoxLidarSpherPoint*>(packet->data);
        for (uint16_t i = 0; i < num_points; ++i) {
            Point point;
            float depth = static_cast<float>(points[i].depth);
            // Assuming theta and phi are provided in degrees.
            float theta = static_cast<float>(points[i].theta) * (3.14159265358979323846f / 180.0f);
            float phi   = static_cast<float>(points[i].phi)   * (3.14159265358979323846f / 180.0f);
            point.x = depth * sinf(phi) * cosf(theta);
            point.y = depth * sinf(phi) * sinf(theta);
            point.z = depth * cosf(phi);
            cloud.push_back(point);
        }
    }
    else {
        std::cerr << "Unknown point data type: " << static_cast<int>(packet->data_type) << std::endl;
    }
}

int main(int argc, const char* argv[]) {
    // Install the SIGINT handler.
    std::signal(SIGINT, SignalHandler);

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <config_path> " << "<output_path>"<< std::endl;
        return -1;
    }

    std::string config_path = argv[1];
    std::string output_dir = std::string(argv[2]) + "/";

    if (!LivoxLidarSdkInit(config_path.c_str())) {
        std::cerr << "Failed to initialize Livox LiDAR SDK!" << std::endl;
        return -1;
    }

    SetLivoxLidarPointCloudCallBack(PointCloudCallback, nullptr);

    if (!LivoxLidarSdkStart()) {
        std::cerr << "Failed to start Livox LiDAR SDK!" << std::endl;
        LivoxLidarSdkUninit();
        return -1;
    }

    std::cout << "Waiting for LiDAR handle discovery..." << std::endl;
    for (int i = 0; i < 50 && g_lidar_handle == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (g_lidar_handle == 0) {
        std::cerr << "Failed to get LiDAR handle from callback." << std::endl;
        LivoxLidarSdkUninit();
        return -1;
    }

    if (EnableLivoxLidarPointSend(g_lidar_handle, nullptr, nullptr) != kLivoxLidarStatusSuccess) {
        std::cerr << "Failed to enable point cloud data sending!" << std::endl;
        LivoxLidarSdkUninit();
        return -1;
    }

    // periodically save .pcd (per every 33ms)
  std::thread saving_thread([output_dir]() {
    using clock = std::chrono::steady_clock;
    /*const std::chrono::milliseconds period(1/30); // fixed period of 33ms*/
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / 30));
    while (true) {
      auto now = std::chrono::system_clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 100;

        if (ms.count() % 33 == 0) {
            std::cout << "starting at " << ms.count() << std::endl;
            break;
        }
    }

    cloud.clear();
    auto next_tick = clock::now() + period;

    // TODO: remove first data
    while (!g_stop) {
      std::string filepath = GetTimestampFilepath(output_dir);

      std::vector<Point> points_to_save;
      {
        std::lock_guard<std::mutex> lock(cloud_mutex);
        if (!cloud.empty()) {
          points_to_save.swap(cloud);
        }
      }
      if (!points_to_save.empty()) {
        SavePointsToBin(points_to_save, filepath);
      }

      std::this_thread::sleep_until(next_tick);
      next_tick += period;
    }
  });

    std::cout << "Collecting point cloud data. Press Ctrl+C to stop recording." << std::endl;

    // main thread waits until SIGINT is received.
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (saving_thread.joinable()) {
        saving_thread.join();
    }

    LivoxLidarSdkUninit();
    std::cout << "Livox LiDAR SDK Uninitialized. Recording complete." << std::endl;

    return 0;
}
