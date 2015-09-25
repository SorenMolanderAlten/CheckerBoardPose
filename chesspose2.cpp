//
// Calculate camera extrinsic calibration given a chessboard (classic PnP - problem)
// Inputs:
// - chessboard specification file (xml)
// - Camera calibration file (xml) with camera matrix in Hartley Zisserman form)
// - desired vertical image crop factors (0, 0 -> no cropping)
// Output:
// - Projection matrix (K * (R | t))
// - Visual projection of chessboard points and coordinate axes
//
// The main openCV algorithm is solvPnP which solves for the rotation and translation matrix given
// corresponding image and 3D object points. For higher precision a sub-pixel interpolation scheme is used.
// Also a simple time-based smoothing is applied to the result to compensate for picel noise.
//
// Instruction:
// Orient the chessboard and align the sides along the direction of the camera, press escape when finished.
// The output projection matrix is saved in "ProjctionMatrix.xml"
// 
// Sören Molander, Alten AB Linköping, 2015-07-02

#define _USE_MATH_DEFINES
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

//Joel har lagt till en kommentar här för att se att github fungerar

double L2norm3D(Mat& vec) {
    Mat v(1,3,CV_64F);
    v = vec;
    double vx = v.at<double>(0,0);
    double vy = v.at<double>(0,1);
    double vz = v.at<double>(0,2);
    return sqrt(vx*vx+vy*vy+vz*vz);
}

static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners) {
    for( int i = 0; i < boardSize.height; ++i )
        for( int j = 0; j < boardSize.width; ++j )
            corners.push_back(Point3f(float( j*squareSize ), float( i*squareSize ), 0));

}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


static vector<double> toEuler(Mat& rot) {
    double RAD2DEG = 180.0/M_PI;
    double m_00 = rot.at<double>(0,0);
    double m_10 = rot.at<double>(1,0);
    double m_20 = rot.at<double>(2,0);
    double m_21 = rot.at<double>(2,1);
    double m_22 = rot.at<double>(2,2);
    vector<double> euler;
    double x_rot = atan2(m_21,m_22)*RAD2DEG;
    double y_rot = atan2(-m_20,sqrt(m_21*m_21+m_22*m_22))*RAD2DEG;
    double z_rot = atan2(m_10,m_00)*RAD2DEG;
    euler.push_back(y_rot);
    euler.push_back(x_rot);
    euler.push_back(z_rot);
    return euler;
}


void cropImageVertically(double cropFactor, Mat& in, Mat& out) {
    int imwidth = in.cols;
    int imheight = in.rows;
    Rect roi(0,int(imheight*cropFactor),imwidth-1,int(imheight*(1-cropFactor)-1));
    Mat cropped_ref(in,roi);
    cropped_ref.copyTo(out);
}


void smoothSolution(vector<Point2f>& in, vector<Point2f>& prev_in) {
    float a = float(0.995);
    for (unsigned int i=0;i<in.size();i++) {
        prev_in[i].x = a*prev_in[i].x + (1-a)*in[i].x;
        prev_in[i].y = a*prev_in[i].y + (1-a)*in[i].y; 
    }
}


int main(int argc, char* argv[]) {
    // help();
    if (argc < 5) {
        cout << "chess pose <board geometry xml file> <camera calibration xml> <crop x factor <crop y factor>" << endl;
        cout << "Crop x or y: 1 = no image, 0 entire image. This is used to adjust principal points" << endl;
        exit(0);
    }
    string boardfile = string(argv[1]);
    string cammatrixfile = string(argv[2]);
    double crop_x = atof(argv[3]);
    double crop_y = atof(argv[4]);
    if (crop_x > 1 || crop_y > 1) {
        cout << "crop factor must be in the range 0..1" << endl;
        exit(-1);
    }

    
    FileStorage boardparams(boardfile, FileStorage::READ);
    FileStorage camparams(cammatrixfile, FileStorage::READ); 
    
    if (!boardparams.isOpened()) {
        cout << "Could not open the configuration file: \"" << boardfile << "\"" << endl;
        return -1;
    }
    // boardparams.release();                                         

    if (!camparams.isOpened()) {
        cout << "Could not open the camera calibration file: \"" << cammatrixfile << "\"" << endl;
        return -1;
    }
    // camparams.release();                                         
    
    Mat cameraMatrix, distCoeffs;
    int boardsize_width, boardsize_height;
    float square_size;
    bool first_time = true;
    FileNode bp = boardparams.getFirstTopLevelNode();

    // Extract camera matrix and chessboard properties
    camparams["Camera_Matrix"] >> cameraMatrix;
    camparams["Distortion_Coefficients"] >> distCoeffs;
    bp["BoardSize_Width"] >> boardsize_width;
    bp["BoardSize_Height"] >> boardsize_height;
    bp["Square_Size"] >> square_size;

    // Need this to properly access camera matrix elements
    Mat K(3,3,CV_64FC1);
    K = cameraMatrix;
    // string mat_type = type2str(K.type());
    // double px = K.at<double>(0,2);
    // double py = K.at<double>(1,2);
    // double px_new = px*(1-crop_x);
    // double py_new = px*(1-crop_y);
    // K.at<double>(0,2) = px_new;
    // K.at<double>(1,2) = py_new;
    
    cout << "camera params: " << endl;
    cout << "K =  " << K << endl;
    cout << "dist " << distCoeffs << endl;
    cout << "Boardsize width = " << boardsize_width << endl;
    cout << "Boardsize height = " << boardsize_height << endl;
    cout << "Square size = " << square_size << endl;

    // Grab video
    VideoCapture cap(0);
    namedWindow("chess", WINDOW_AUTOSIZE);
    Mat image, imageGray, imageCrop;
    Mat rvec, tvec;
    bool found, pose_found = false;
    vector<Point2f> imagePoints, prevPoints, smoothedPoints, imagePointsProj;
    vector< vector<Point3f> > objectPoints(1);
    double dist;
    vector<double> rodrigues_rot;
    Mat rot, Proj3, Proj34;
    double err;
    bool done=false;
    vector<double> euler;
    int cols = K.cols, rows = K.rows;

    // Define coordinate axes for visual display of coordinate axes
    vector<Point3f> poseAngles;
    float poseAxisLen = 100;
    poseAngles.push_back(Point3f(0,0,0));
    poseAngles.push_back(Point3f(-poseAxisLen,0,0));
    poseAngles.push_back(Point3f(0,-poseAxisLen,0));
    poseAngles.push_back(Point3f(0,0,-poseAxisLen));
    Size boardSize(boardsize_width, boardsize_height);
    calcBoardCornerPositions(boardSize, square_size, objectPoints[0]);

    while (!done) {
        bool ok = cap.read(image);
        cropImageVertically(crop_y, image, imageCrop);
        cvtColor(imageCrop, imageGray, CV_BGR2GRAY);
        smoothedPoints = imagePoints;
        found = findChessboardCorners( imageCrop, boardSize, imagePoints,
                                       CV_CALIB_CB_ADAPTIVE_THRESH |
                                       CV_CALIB_CB_FAST_CHECK |
                                       CV_CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            if (first_time) { // initialize points for smoothing filter
                smoothedPoints = imagePoints;
                first_time = false;
            } else
                smoothSolution(imagePoints, smoothedPoints);
            cornerSubPix( imageGray, smoothedPoints, Size(11,11),
                          Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
            drawChessboardCorners(imageCrop, boardSize, Mat(smoothedPoints), found );
            objectPoints.resize(imagePoints.size(),objectPoints[0]);
            // pose_found = solvePnP(objectPoints[0], imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false);
            pose_found = solvePnP(objectPoints[0], imagePoints, K, distCoeffs, rvec, tvec, false);
            if (!pose_found)
                cout << "Warning: Pose not found" << endl;

            Rodrigues(rvec,rot);
            projectPoints( Mat(objectPoints[0]), rvec, tvec, K, distCoeffs, imagePointsProj);
            err = norm(Mat(smoothedPoints), Mat(imagePointsProj), CV_L2);
            projectPoints(Mat(poseAngles), rvec, tvec, K, distCoeffs, imagePointsProj);
            line(imageCrop,imagePointsProj[0],imagePointsProj[1],Scalar(255,0,255),2);
            line(imageCrop,imagePointsProj[0],imagePointsProj[2],Scalar(255,0,255),2);
            line(imageCrop,imagePointsProj[0],imagePointsProj[3],Scalar(255,0,255),2);
            euler = toEuler(rot);
            dist = L2norm3D(tvec);
            cout <<"Dist (mm) = " << dist
                  <<" Angles (deg), yaw = " << euler[0] << ", pitch = " << euler[1] << ", roll " << euler[2] << endl;

        } else
            first_time = true;

        imshow("chess", imageCrop);
        if (waitKey(10) == 27) {
            cap.release();
            done = true;
        }
    }
    if (pose_found) {
        FileStorage out ("ProjectionMatrix.xml", FileStorage::WRITE);
        Proj3 = cameraMatrix * rot;
        hconcat(Proj3,tvec,Proj34);
        cout << "Projection matrix = " << Proj34 << endl;
        out << "Projection_Matrix" << Proj34;
        out.release();
    }
    exit(0);
}    
    

