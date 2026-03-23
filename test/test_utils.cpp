#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
// #include <dirent.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "test_utils.h"

std::vector<std::string> classes{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

void readYoloLabelFile(const std::string &path, int imgWidth, int imgHeight, std::vector<Object> &objects)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open label file: " << path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        int class_id;
        float x_center, y_center, width, height, conf;

        ss >> class_id >> x_center >> y_center >> width >> height >> conf;

        float x = (x_center - width / 2.0f) * imgWidth;
        float y = (y_center - height / 2.0f) * imgHeight;
        float w = width * imgWidth;
        float h = height * imgHeight;

        Object obj;
        obj.label = class_id;
        obj.prob = conf;
        obj.rect = cv::Rect_<float>(x, y, w, h);
        objects.push_back(obj);
    }
}

void drawTracklets(cv::Mat &image, const std::vector<STrack> &tracklets)
{
    for (const STrack &tracklet : tracklets)
    {
        cv::Rect box(
            static_cast<int>(tracklet.tlwh[0]),
            static_cast<int>(tracklet.tlwh[1]),
            static_cast<int>(tracklet.tlwh[2]),
            static_cast<int>(tracklet.tlwh[3]));

        cv::Scalar color = tracklet.get_color();

        cv::rectangle(image,
                      box,
                      color,
                      2,
                      cv::LINE_AA,
                      false);

        cv::Point2i textLocation(
            static_cast<int>(tracklet.tlwh[0]),
            static_cast<int>(tracklet.tlwh[1]) - 5);
        cv::putText(image,
                    classes[tracklet.class_id] + ", track_id: " + std::to_string(tracklet.track_id),
                    textLocation,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv::LINE_AA,
                    false);
    }
}