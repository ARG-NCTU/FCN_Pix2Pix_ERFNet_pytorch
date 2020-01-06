#include <ros/ros.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Pose.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
// create folder
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
//
#include "subt_msgs/arti_input.h"
#include "subt_msgs/masks.h"
#include "subt_msgs/mask_center.h"
#include "subt_msgs/ArtifactPoseArray.h"
#include "subt_msgs/ArtifactPose.h"
using namespace ros;
using namespace std;
using namespace cv;
using namespace pcl;
using namespace message_filters;


class pub_arti_pose{
  public:
    pub_arti_pose();
    void callback(const subt_msgs::masks);
    void getXYZ(float* , float* ,float );
  private:
    ros::Publisher pub_pose;
    ros::Subscriber mask_result;
    PointCloud<PointXYZRGB>::Ptr pc;
    float fx;
    float fy;
    float cx;
    float cy;
};