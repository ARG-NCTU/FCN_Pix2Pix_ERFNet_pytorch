#include "pub_arti_pose.h"

void pub_arti_pose::getXYZ(float* a, float* b,float zc){

	float inv_fx = 1.0/fx;
	float inv_fy = 1.0/fy;
	*a = (*a - cx) * zc * inv_fx;
	*b = (*b - cy) * zc * inv_fy;
	return;
}

void pub_arti_pose::callback(const subt_msgs::masks msg){
    pc.reset(new PointCloud<PointXYZRGB>());
    cv_bridge::CvImagePtr img_ptr_depth = cv_bridge::toCvCopy(msg.depth, sensor_msgs::image_encodings::TYPE_16UC1);

    //// artifact pose ////
    subt_msgs::ArtifactPoseArray arti_pose_arr = subt_msgs::ArtifactPoseArray();
    arti_pose_arr.header = msg.header;
	arti_pose_arr.count = msg.count;
	arti_pose_arr.camera = msg.camera;
    cout << "Total artifact: "<< msg.count << endl;
	cout << "Camera Side: " << msg.camera << endl;
    for (int i = 0; i < msg.count; i++){
        float z;
        subt_msgs::ArtifactPose arti_pose = subt_msgs::ArtifactPose();
        geometry_msgs::Pose pose;
        cout << "Class: " << msg.masks[i].Class << endl;
		cout << "center_x: " << msg.masks[i].y << endl;
		cout << "center_y: " << msg.masks[i].x << endl;
        

		arti_pose.Class = msg.masks[i].Class;

        float* y = new float (msg.masks[i].x);
		float* x = new float (msg.masks[i].y);
        if(0 <= *x && *x < 480 && 0 <= *y && *y < 640)
			z = float(img_ptr_depth->image.at<unsigned short int>(*x, *y))/1000.;
		else 
			z = 0;

        getXYZ(y,x,z);
        cout << "x: " << z << " y: " << -*y << " z: " << -*x << endl;
        
        pose.position.x = z;
        pose.position.y = -*y;
		pose.position.z = -*x;
		arti_pose.pose = pose;
	    free(x);
		free(y);
		arti_pose_arr.pose_array.push_back(arti_pose);
	}

	pub_pose.publish(arti_pose_arr);
	return;
}

pub_arti_pose::pub_arti_pose(){
	NodeHandle nh;

	sensor_msgs::CameraInfo::ConstPtr msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/color/camera_info",ros::Duration());
	fx = msg->P[0];
	fy = msg->P[5];
	cx = msg->P[2];
	cy = msg->P[6];

	pub_pose = nh.advertise<subt_msgs::ArtifactPoseArray> ("/artifact_pose", 10);
	mask_result = nh.subscribe<subt_msgs::masks>("/fcn_pix2pix_prediction/masks", 1, &pub_arti_pose::callback,this); 
}

int main(int argc, char** argv){
	init(argc, argv, "pub_arti_pose");
	pub_arti_pose pub_arti_pose;
	spin();
	return 0;
}