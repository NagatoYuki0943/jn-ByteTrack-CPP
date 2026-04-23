#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "BYTETracker.h"
#include "test_utils.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Please provide the required arguments: video_path, labels_path, output_path." << std::endl;
        std::cout << "  video_path   : Input video path." << std::endl;
        std::cout << "  labels_path  : Path to the detection model predictions (YOLO format)" << std::endl;
        std::cout << "  output_path  : Path to save the output video" << std::endl;
        return 1;
    }

    const std::string video_path = argv[1];
    const fs::path labels_path = argv[2];
    const std::string output_path = argv[3];

    if (!fs::exists(video_path))
    {
        std::cout << "video_path: " << video_path << "not exist" << std::endl;
        return -1;
    }

    if (!fs::exists(labels_path))
    {
        std::cout << "video_path: " << labels_path << "not exist" << std::endl;
        return -1;
    }

    // Initialize video capture
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
        return 1;
    // Input video information
    int img_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int img_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Output video writer
    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(img_w, img_h));

    // Initialize tracker
    // max_time_lost 死亡倒计时，目标丢失（未匹配到检测框）后，在内存中保留等待重新出现的总帧数
    int max_time_lost = 60;
    // track_high_thresh 高分界线，得分大于此值的框为“高分框”，参与第一轮常规匹配。
    float track_high_thresh = 0.3;
    // track_low_thresh 低分界线，得分在此值与高分界线之间的为“低分框”，参与第二轮遮挡修补；低于此值的框直接丢弃。
    float track_low_thresh = 0.1;
    // new_track_thresh 出生门槛，只有得分大于此值的检测框，才能被初始化为全新的追踪目标。
    float new_track_thresh = 0.3;
    // high_match_thresh/low_match_thresh/unconfirmed_match_thresh 认亲标准，判定检测框与已有轨迹“是否为同一目标”的匹配代价容忍度（通常基于 IoU）。越高，越容易匹配；越低，越难匹配。
    //  什么时候应该调高（比如 0.8 ~ 0.9）？
    //      视频帧率（FPS）较低时。
    //      目标运动速度极快时。
    //      原因： 这种情况下，目标在相邻两帧之间的位移跨度很大，导致上一帧的预测框和这一帧的检测框交集 (IoU) 很小。如果不把容忍度调高，追踪器会认为老目标消失了，新目标出现了，导致疯狂闪烁和切换 ID。
    //  什么时候应该调低（比如 0.4 ~ 0.5）？
    //      画面极其拥挤密集时（例如密集的车流、十字路口的人群）。
    //      原因： 当多个目标靠得非常近时，如果追踪器太宽容，很容易发生 ID 劫持（比如 A 车和 B 车并排，追踪器把 A 的 ID 错误地连到了 B 的检测框上）。降低阈值能逼迫追踪器变得“严谨”，只认准那个跟历史轨迹重合度最高的目标。
    // high_match_thresh 已追踪到的轨迹+丢失的规矩 vs 高分目标 的阈值
    // low_match_thresh 剩余已追踪到的轨迹 vs 低分目标 的阈值
    // unconfirmed_match_thresh 候选轨迹 vs 剩余高分目标 的阈值
    float high_match_thresh = 0.8;
    float low_match_thresh = 0.5;
    float unconfirmed_match_thresh = 0.8;
    // min_hits 连续追踪到多少帧才认定是追踪目标
    int min_hits = 3;
    ByteTrack::BYTETracker tracker(
        max_time_lost,
        track_high_thresh,
        track_low_thresh,
        new_track_thresh,
        high_match_thresh,
        low_match_thresh,
        unconfirmed_match_thresh,
        min_hits);
    std::vector<ByteTrack::Object> objects;
    std::vector<ByteTrack::STrack> tracklets;
    std::vector<ByteTrack::STrack> lostTracklets;
    std::vector<ByteTrack::STrack> removedTracklets;

    int frame_idx = 1;
    cv::Mat frame;
    cv::Mat outputFrame;
    while (true)
    {
        // Read frame
        if (!cap.read(frame))
            break;
        outputFrame = frame;
        std::cout << "Processing frame " << frame_idx << " / " << nFrame << std::endl;

        // Read and process model predictions
        fs::path labelFile = labels_path / (std::to_string(frame_idx) + ".txt");
        if (fs::exists(labelFile))
        {
            objects.clear();
            tracklets.clear();

            // Read labels
            readYoloLabelFile(labelFile.string(), img_w, img_h, objects);

            // Tracking
            tracker.update(objects, tracklets, lostTracklets, removedTracklets);

            std::cout << "objects size: " << objects.size() << std::endl;
            std::cout << "Tracklets size: " << tracklets.size() << std::endl;
            std::cout << "lostTracklets size: " << lostTracklets.size() << std::endl;
            std::cout << "removedTracklets size: " << removedTracklets.size() << std::endl;

            // Draw boxes
            drawTracklets(outputFrame, tracklets);
        }
        else
        {
            std::cout << "No labels for frame " << frame_idx << std::endl;
        }

        // Write output frame
        writer.write(outputFrame);

        std::cout << std::endl;

        frame_idx++;
    }

    // Release video capture and writer
    cap.release();
    writer.release();

    return 0;
}
