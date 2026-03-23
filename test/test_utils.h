#include <string>
#include <vector>

#include "BYTETracker.h"


void readYoloLabelFile(const std::string& path, int imgWidth, int imgHeight, std::vector<Object>& objects);

void drawTracklets(cv::Mat& image, const std::vector<STrack>& tracklets);
