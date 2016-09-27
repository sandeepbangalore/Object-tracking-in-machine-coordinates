/*
	The following program is used to detect and track objects.
	Upon detection it can inform as to how much more the camera
	has to move in a particular direction in order to center the
	object with respect to the camera.

	The source folder contains a file named "cameracalib.xml", this
	file contains the settings in order to perform camera calibration
	task.

	The camera is presently calibrated at z = 0 and the file named 
	"out_camera_data.xml" has the camera parameters at the orientaion.
	If there is a need to change the orientation and the zero reference
	of the camera, one has to run the camera calibration routine.

	Press  'c'   for  enabling  camera calibration. In oder to perform
	camera calibration one has to take a print out of the file "chess board.png"
	available in the souce folder. As per the "cameracalib.xml" file the chess board
	image should have 9x6 squares. The values can be changed if required by editing 
	the following lines:

	<BoardSize_Width> 9</BoardSize_Width>
    <BoardSize_Height>6</BoardSize_Height>

	Similarly the size of the squares in real world can be changes here:

	<Square_Size>25</Square_Size> // size is in millimeter scale.

	The method required a need for at least 20 different orientations of 
	the chess board as seen by the camera. Cureently it is set to 25, 
	this can be changed  at:

	Calibrate_NrOfFrameToUse>25</Calibrate_NrOfFrameToUse>

	and the delay between two frames can be set here:

	<Input_Delay>5000</Input_Delay> //Time is in milliseconds

	Once the calibration is completed an output file containing all the
	camera parameters will be created for further use.

	Taking a new template image:

	Show the template image to the camera and press 't'. The screen will
	be  frozen.  Using the mouse  click the  upper-left  and lower-right
	corners of the  object which you want to  detect. The application will
	learn the image you show immediately and will be ready to detect it.

	This routine uses a combination of FAST keypoint detectors and BRIEF 
	descriptors. FAST (Features from Accelerated Segment Test) algorithm 
	was proposed by Edward Rosten and Tom Drummond in their paper “Machine
	learning for high-speed corner detection” in 2006 (Later revised it in 2010).
	Descriptors on the other hand are the way to compare the keypoints.
	They summarize, in vector format (of constant length) some characteristics 
	about the keypoints. For example, it could be their intensity in the direction
	of their most pronounced orientation. It's assigning a numerical description 
	to the area of the image the keypoint refers to.

	One important point is that BRIEF is a feature descriptor, it doesn’t provide
	any method to find the features. So we will have to use any other feature 
	detectors like FAST, SIFT, SURF etc. BRIEF is used for faster method feature 
	descriptor calculation and matching. It also provides high recognition rate 
	unless there is large in-plane rotation. In order to get past this issue we 
	rotate the template image by 360° with an increment of 20° and find the keypoints
	and descriptors at different orientations and match the best.

	Enable - Disable rotation and scale invariance:
	
	Press  'd' for enabling or disabling the rotation and scale invariance. Disabling
	rotation  and  scale invariance  mean that  the detection will be done by only on
	the original template image, instead of  trying the  incoming frame  with several
	different  rotations and scales of this template.

	Quit:
	Press Q.

*/

#include <stdafx.h>
#include "BRIEF.h"

#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "opencv\cvaux.h"
#include <vector>
#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef _CRT_SECURE_NO_WARNINGS
# define _CRT_SECURE_NO_WARNINGS
#endif

#define M_PI 3.1416;
using namespace std;
using namespace cv;
/************************************ GLOBAL CONSTANTES ***************************************/

// Frame width and height of the capture
static const int FRAME_WIDTH = 640;
static const int FRAME_HEIGHT = 480;

// Maximum number of keypoint matches allowed by the program
static const int MAXIMUM_NUMBER_OF_MATCHES = 500;

// Minimum  scale ratio of  the program.  It indicates  that templates
// having scales  [0.5, 1]  of the original  template will be  used to
// generate new templates. Scales are determined in a logarithm base.
static const float SMALLEST_SCALE_CHANGE = 0.5;

// Number of different scales used to generate the templates.
static const int NUMBER_OF_SCALE_STEPS = 3;

// Number  of   different  rotation   angles  used  to   generate  the
// templates. 18 indicates that  [0 20 ... 340] degree-rotated samples
// will be stored in the 2-D array for each scale.
static const int NUMBER_OF_ROTATION_STEPS = 18;

static const double zInitialCoordinates = 300.0; // The z Coordinates as per calibration


/************************************ GLOBAL VARIABLES ***************************************/
//Checks if it is the first time we are allocating memory or if it is 
//duplicating it
bool firstTime = true; 
bool printZ = false;
//Confirms if the object is centered or not
bool isCentered = false;
// Brief object which contains methods describing keypoints with BRIEF
// descriptors.
CVLAB::BRIEF brief;

// 2-D   array   storing  detected   keypoints   for  every   template
// taken. Templates  are generated  rotating and scaling  the original
// image   for   each  rotation   angle   and   scale  determined   by
// NUMBER_OF_SCALE STEPS and NUMBER_OF_ROTATION_STEPS.
vector<cv::KeyPoint> templateKpts[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

// 2-D array  storing BRIEF descriptors  of corresponding templateKpts
// elements
vector< bitset<CVLAB::DESC_LEN> > templateDescs[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

// 2-D array storing  the coordinates of the corners  of the generated
// templates in original image coordinate system.
CvMat templateObjectCorners[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

// Data part of the templateObjectCorners
double* templateObjectCornersData[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

// The coordinates  of the keypoints  matching each other in  terms of
// Hamming  Distance between  BRIEF descriptors  of  them.  match1Data
// represents the  keypoint coordinates  of the matching  template and
// match2Data  represents  the  matching  keypoints  detected  in  the
// current frame. Elements with even indices contain x coordinates and
// those  with odd  indices  contain y  coordinates  of the  keypoints
// detected:
double match1Data[2 * MAXIMUM_NUMBER_OF_MATCHES];
double match2Data[2 * MAXIMUM_NUMBER_OF_MATCHES];
double zPresentCoordinates; // The z coordinates as entered by user
Mat calibrationMatrix;					// The cameras calibration matrix
Mat calibrationDistortionCoefficients;	// The cameras distortion coefficients 
Mat rvec = Mat(Size(3,1), CV_64F);		// rotation vector
Mat tvec = Mat(Size(3,1), CV_64F);		// translation vector
vector<Point3d> framePoints;			
vector<Point2d> imageFramePoints;
Mat rotation;
float present_scale;					

// Holds the mode of the application.
enum APPLICATION_MODE
  {
    DO_NOTHING, // Application only captures a frame and shows it
	DO_NOTHING_CALIB, // Application only captures a frame and performs camera calibration
	DETECTION, // Application detects the planar object whose features are stored
    TAKING_NEW_TEMPLATE, // Application captures a new template
    END // Application quits
  } appMode;


// Holds if any template has taken before or not
bool hasAnyTemplateChosenBefore = false;


// Indicates  either all the  BRIEF descriptors  stored or  only BRIEF
// descriptors  of the  original template  will be  used  for template
// matching.
bool doDBSearch = true;


// Template image captured in RGB.
IplImage* templateImageRGBFull;


// The part of the  templateImageRGBFull which is inside the rectangle
// drawn by the user. Both in RGB and Grayscale
IplImage* templateImageRGB = NULL;
IplImage* templateImageGray = NULL;


// Last frame captured by the camera in RGB and Grayscale
IplImage* newFrameRGB;
IplImage* newFrameGray;


// Copy of the newFrameRGB for further processing.
IplImage* outputImage;


// Last frame  taken and the original  template image are  put side by
// side in order to show the result.
IplImage* sideBySideImage;


// Object used for capturing image frames
CvCapture* capture;


// Threshold value given to FAST detector:
int fastThreshold;


// Number  of the  points selected  by the  user on  the  new template
// image:
int noOfPointsPickedOnTemplateImage;


// Image coordinates  of the  points selected by  the user in  the new
// template image:
CvPoint templateCorners[2];


// Coordinate of  the top-left  corner of the  rectangle drawn  by the
// user on the new template image:
int templateROIX;
int templateROIY;


// Coordinates of the keypoints of the template which fits best to the
// frame captured by the camera.   These are the coordinates which are
// transformed back to the original template image coordinates:
double pointsOnOriginalImage[MAXIMUM_NUMBER_OF_MATCHES];


// Font which is used to write on the image
CvFont font;


// Number of the frames processed per second in the application
int fps;


// Time elapsed:
double keypointExtractTime; // by FAST detector
double bitDescTime; // to describe all keypoints with BRIEF descriptor
double matchTime; // to find the matches between 2 BRIEF descriptor vector
double hmgEstTime; // to estimate the Homography matrix between 2 images
double totalMatchTime; // to match the BRIEF descriptors of the incoming frame with all
double AxesTime; // to generate the 3D axes




// (# of Matching Keypoints / #  of the Keypoints) * 100, for the best fit:
int matchPercentage;

/*************************************** FUNCTIONS DECLARATION ************************************/
// Main function that starts the camera calibration routine
int cameraCalib();

/**************************************************************************************************/

/****************************************** INLINE FUNCTIONS **************************************/

// Returns radian equivalent of an angle in degrees
inline float degreeToRadian(const float d)
{
  return (d / 180.0) * M_PI;
}

// Converts processor tick counts to milliseconds
inline double toElapsedTime(const double t)
{
  return t / (1e3 * cvGetTickFrequency());
}

/**************************************************************************************************/

// Function for handling keyboard inputs
void waitKeyAndHandleKeyboardInput(int timeout)
{
  // Wait for the keyboard input
  const char key = cvWaitKey(timeout);
  // Change the application mode according to the keyboard input
  switch (key) {
   case 'c': case 'C':
    appMode = DO_NOTHING_CALIB;
    break;
  case 's': case 'S':
    appMode = DO_NOTHING;
    break;
  case 'q': case 'Q':
    appMode = END;
    break;
  case 't': case 'T':
    if (appMode == TAKING_NEW_TEMPLATE) {
      noOfPointsPickedOnTemplateImage = 0;
      // if a template has been taken before, go back to detection of last template
      // otherwise
      appMode = hasAnyTemplateChosenBefore ? DETECTION : DO_NOTHING;
    }
    else
      appMode = TAKING_NEW_TEMPLATE;
    break;
  case 'd': case 'D':
    doDBSearch = !doDBSearch;
    break;
  }
}

// Function for handling mouse inputs
void mouseHandler(int event, int x, int y, int flags, void* params)
{
  if (appMode == TAKING_NEW_TEMPLATE) {
    templateCorners[1] = cvPoint(x, y);
    switch (event) {
    case CV_EVENT_LBUTTONDOWN:
      templateCorners[noOfPointsPickedOnTemplateImage++] = cvPoint(x, y);
      break;
    case CV_EVENT_RBUTTONDOWN:
      break;
    case CV_EVENT_MOUSEMOVE:
      if (noOfPointsPickedOnTemplateImage == 1)
	templateCorners[1] = cvPoint(x, y);
      break;
    }
  }
}

// Draws a quadrangle on an image given (u, v) coordinates, color and thickness
void drawQuadrangle(IplImage* frame,
		    const int u0, const int v0,
		    const int u1, const int v1,
		    const int u2, const int v2,
		    const int u3, const int v3,
		    const CvScalar color, const int thickness)
{
  cvLine(frame, cvPoint(u0, v0), cvPoint(u1, v1), color, thickness);
  cvLine(frame, cvPoint(u1, v1), cvPoint(u2, v2), color, thickness);
  cvLine(frame, cvPoint(u2, v2), cvPoint(u3, v3), color, thickness);
  cvLine(frame, cvPoint(u3, v3), cvPoint(u0, v0), color, thickness);
}

// Draws a quadrangle with the corners of the object detected on img
void markDetectedObject(IplImage* frame, const double * detectedCorners)
{
  drawQuadrangle(frame,
		 detectedCorners[0], detectedCorners[1],
		 detectedCorners[2], detectedCorners[3],
		 detectedCorners[4], detectedCorners[5],
		 detectedCorners[6], detectedCorners[7],
		 cvScalar(255, 255, 255), 3);
}

// Draws a plus sign on img given (x, y) coordinate
void drawAPlus(IplImage* img, const int x, const int y)
{
  cvLine(img, cvPoint(x - 5, y), cvPoint(x + 5, y), CV_RGB(255, 0, 0));
  cvLine(img, cvPoint(x, y - 5), cvPoint(x, y + 5), CV_RGB(255, 0, 0));
}

// Marks the keypoints with plus signs on img
void showKeypoints(IplImage* img, const vector<cv::KeyPoint>& kpts)
{
  for (unsigned int i = 0; i < kpts.size(); ++i)
    drawAPlus(img, kpts[i].pt.x, kpts[i].pt.y);
}

// Captures a new frame. Returns if capture is taken without problem or not.
bool takeNewFrame(void)
{
  if ((newFrameRGB = cvQueryFrame(capture)))
    cvCvtColor(newFrameRGB, newFrameGray, CV_BGR2GRAY);
  else
    return false;
  return true;
}

// Puts img1 and img2 side by side and stores into result
void putImagesSideBySide(IplImage* result, const IplImage* img1, const IplImage* img2)
{
  // widthStep of the resulting image
  const int bigWS = result->widthStep;
  // half of the widthStep of the resulting image
  const int bigHalfWS = result->widthStep >> 1;
  // widthStep of the image which will be put in the left
  const int lWS = img1->widthStep;
  // widthStep of the image which will be put in the right
  const int rWS = img2->widthStep;

  // pointer to the beginning of the left image
  char *p_big = result->imageData;
  // pointer to the beginning of the right image
  char *p_bigMiddle = result->imageData + bigHalfWS;
  // pointer to the image data which will be put in the left
  const char *p_l = img1->imageData;
  // pointer to the image data which will be put in the right
  const char *p_r = img2->imageData;

  for (int i = 0; i < FRAME_HEIGHT; ++i, p_big += bigWS, p_bigMiddle += bigWS) {
    // copy a row of the left image till the half of the resulting image
    memcpy(p_big, p_l + i*lWS, lWS);
    // copy a row of the right image from the half of the resulting image to the end of it
    memcpy(p_bigMiddle, p_r + i*rWS, rWS);
  }
}

// Marks the matching keypoints on two images which were put side by side
void showMatches(const int matchCount)
{
  const int iterationEnd = 2 * matchCount;

  for (int xCoor = 0, yCoor = 1; xCoor < iterationEnd; xCoor += 2, yCoor += 2) {
    // Draw a line between matching keypoints
    cvLine(sideBySideImage,
	   cvPoint(match2Data[xCoor], match2Data[yCoor]),
	   cvPoint(pointsOnOriginalImage[xCoor] + templateROIX + FRAME_WIDTH,
		   pointsOnOriginalImage[yCoor] + templateROIY),
	   cvScalar(0, 255, 0), 1);
  }
}

// Returns whether H is a nice homography matrix or not
bool niceHomography(const CvMat * H)
{
  const double det = cvmGet(H, 0, 0) * cvmGet(H, 1, 1) - cvmGet(H, 1, 0) * cvmGet(H, 0, 1);
  if (det < 0)
    return false;

  const double N1 = sqrt(cvmGet(H, 0, 0) * cvmGet(H, 0, 0) + cvmGet(H, 1, 0) * cvmGet(H, 1, 0));
  if (N1 > 4 || N1 < 0.1)
    return false;

  const double N2 = sqrt(cvmGet(H, 0, 1) * cvmGet(H, 0, 1) + cvmGet(H, 1, 1) * cvmGet(H, 1, 1));
  if (N2 > 4 || N2 < 0.1)
    return false;

  const double N3 = sqrt(cvmGet(H, 2, 0) * cvmGet(H, 2, 0) + cvmGet(H, 2, 1) * cvmGet(H, 2, 1));
  if (N3 > 0.002)
    return false;

  return true;
}

// Rotates src around center with given angle and assigns the result to dst
void rotateImage(IplImage* dst, IplImage* src, const CvPoint2D32f& center, float angle)
{
  static CvMat *rotMat = cvCreateMat(2, 3, CV_32FC1);
  cv2DRotationMatrix(center, angle, 1.0, rotMat);
  cvWarpAffine(src, dst, rotMat);
}

// Transforms the coordinates of the keypoints of a template image whose matrix index is
// (scaleInd, rotInd) into the original template image's (scale = 1, rotation angle = 0) coordinates
void transformPointsIntoOriginalImageCoordinates(const int matchNo, const int scaleInd, const int rotInd)
{
  // Difference between the angles of two consecutive samples
  static const float ROT_ANGLE_INCREMENT = 360.0 / NUMBER_OF_ROTATION_STEPS;

  // Take the scale samples in a logarithmic base
  static const float k = exp(log(SMALLEST_SCALE_CHANGE) / (NUMBER_OF_SCALE_STEPS - 1));
  const float scale = pow(k, scaleInd);

  // Center of the original image
  const float orgCenterX = templateImageGray->width / 2.0;
  const float orgCenterY = templateImageGray->height / 2.0;

  // Center of the scaled image
  const float centerX = orgCenterX * scale;
  const float centerY = orgCenterY * scale;

  // Rotation angle for the template
  const float angle = ROT_ANGLE_INCREMENT * rotInd;
  // Avoid repeatition of the trigonometric calculations
  const float cosAngle = cos(degreeToRadian(-angle));
  const float sinAngle = sin(degreeToRadian(-angle));

  const float iterationEnd = 2 * matchNo;
  for (int xCoor = 0, yCoor = 1; xCoor < iterationEnd; xCoor += 2, yCoor += 2) {
    // Translate the point so that the origin is in the middle of the image
    const float translatedX = match1Data[xCoor] - centerX;
    const float translatedY = match1Data[yCoor] - centerY;

    // Rotate the point so that the angle between this template and the original template will be zero
    const float rotatedBackX = translatedX * cosAngle - translatedY * sinAngle;
    const float rotatedBackY = translatedX * sinAngle + translatedY * cosAngle;

    // Scale the point so that the size of this template will be equal to the original one
    pointsOnOriginalImage[xCoor] = rotatedBackX / scale + orgCenterX;
    pointsOnOriginalImage[yCoor] = rotatedBackY / scale + orgCenterY;
	
  }

}

// Estimates the fps of the application
void fpsCalculation(void)
{
  static int64 currentTime, lastTime = cvGetTickCount();
  static int fpsCounter = 0;
  currentTime = cvGetTickCount();
  ++fpsCounter;
  
  // If 1 second has passed since the last FPS estimation, update the fps
  if (currentTime - lastTime > 1e6 * cvGetTickFrequency()) {
    fps = fpsCounter;
    lastTime = currentTime;
    fpsCounter = 0;
  }
}

// Writes the statistics showing the performance of the application to img
void showOutput(IplImage* img)
{
  static char text[256];

  if (appMode != TAKING_NEW_TEMPLATE) {
    sprintf(text, "FPS: %d", fps);
    cvPutText(img, text, cvPoint(10, 30), &font, cvScalar(255, 0, 0));

    sprintf(text, "KP Extract: %f", toElapsedTime(keypointExtractTime));
    cvPutText(img, text, cvPoint(10, 50), &font, cvScalar(255, 0, 0));

    sprintf(text, "Bit Desc: %f", toElapsedTime(bitDescTime));
    cvPutText(img, text, cvPoint(10, 70), &font, cvScalar(255, 0, 0));

    sprintf(text, "Match Time: %f", toElapsedTime(matchTime));
    cvPutText(img, text, cvPoint(10, 90), &font, cvScalar(255, 0, 0));

    sprintf(text, "Total Matching Time: %f", toElapsedTime(totalMatchTime));
    cvPutText(img, text, cvPoint(10, 110), &font, cvScalar(255, 0, 0));

    sprintf(text, "RANSAC: %f", toElapsedTime(hmgEstTime));
    cvPutText(img, text, cvPoint(10, 130), &font, cvScalar(255, 0, 0));

    sprintf(text, "Match Percentage: %d%%", matchPercentage);
    cvPutText(img, text, cvPoint(10, 150), &font, cvScalar(255, 0, 0));

	sprintf(text, "Axes Time: %f", toElapsedTime(AxesTime));
    cvPutText(img, text, cvPoint(10, 170), &font, cvScalar(255, 0, 0));

	sprintf(text, "Scale: %f", present_scale);
    cvPutText(img, text, cvPoint(10, 190), &font, cvScalar(255, 0, 0));

	sprintf(text, "Centered?: %d", isCentered);
    cvPutText(img, text, cvPoint(10, 210), &font, cvScalar(255, 0, 0));

	if(isCentered == true)
	{
	sprintf(text, "POI: %f, %f", imageFramePoints[0].x, imageFramePoints[0].y);
    cvPutText(img, text, cvPoint(10, 230), &font, cvScalar(255, 0, 0));
	}
	
  }
  
  cvShowImage("BRIEF", img);
}

// Detect keypoints of img with FAST and store them to kpts given the threshold kptDetectorThreshold.
int extractKeypoints(vector< cv::KeyPoint >& kpts, int kptDetectorThreshold, IplImage* img)
{
  CvRect r = cvRect(CVLAB::IMAGE_PADDING_LEFT, CVLAB::IMAGE_PADDING_TOP,
		    CVLAB::SUBIMAGE_WIDTH(img->width), CVLAB::SUBIMAGE_HEIGHT(img->height));

  // Don't detect keypoints on the image borders:
  cvSetImageROI(img, r);
  
  // Use FAST corner detector to detect the image keypoints
  cv::FAST((cv::Mat)img, kpts, kptDetectorThreshold, true);

  // Get the borders back:
  cvResetImageROI(img);

  // Transform the points to their actual image coordinates:
  for (unsigned int i = 0, sz = kpts.size(); i < sz; ++i)
    kpts[i].pt.x += CVLAB::IMAGE_PADDING_LEFT, kpts[i].pt.y += CVLAB::IMAGE_PADDING_TOP;

  return kpts.size();
}

// Tries to find a threshold for FAST that gives a number of keypoints between lowerBound and upperBound:
int chooseFASTThreshold(const IplImage* img, const int lowerBound, const int upperBound)
{
  static vector<cv::KeyPoint> kpts;

  int left = 0;
  int right = 255;
  int currentThreshold = 128;
  int currentScore = 256;

  IplImage* copyImg = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
  cvCopyImage(img, copyImg);

  while (currentScore < lowerBound || currentScore > upperBound) {
    currentScore = extractKeypoints(kpts, currentThreshold, copyImg);

    if (lowerBound > currentScore) {
      // we look for a lower threshold to increase the number of corners:
      right = currentThreshold;
      currentThreshold = (currentThreshold + left) >> 1;
      if (right == currentThreshold)
	break;
    } else {
      // we look for a higher threshold to decrease the number of corners:
      left = currentThreshold;
      currentThreshold = (currentThreshold + right) >> 1;
      if (left == currentThreshold)
	break;
    }
  }
  cvReleaseImage(&copyImg);

  return currentThreshold;
}

// Saves the coordinates of the corners of the rectangle drawn by the user when
// capturing a new template image
void saveCornerCoors(void)
{
  const double templateWidth = templateImageGray->width;
  const double templateHeight = templateImageGray->height;

  double* corners = templateObjectCornersData[0][0];
  corners[0] = 0;
  corners[1] = 0;
  corners[2] = templateWidth;
  corners[3] = 0;
  corners[4] = templateWidth;
  corners[5] = templateHeight;
  corners[6] = 0;
  corners[7] = templateHeight;
}

// Saves the image inside the rectangle drawn by the user as the new template image.
// Returns if it can be used as a new template or not.
bool saveNewTemplate(void)
{
  // Calculate the size of the new template
  const int templateWidth = templateCorners[1].x - templateCorners[0].x;
  const int templateHeight = templateCorners[1].y - templateCorners[0].y;

  // If the size of of the new template is illegal, return false
  if ((SMALLEST_SCALE_CHANGE * templateWidth) < CVLAB::IMAGE_PADDING_TOTAL ||
      (SMALLEST_SCALE_CHANGE * templateHeight) < CVLAB::IMAGE_PADDING_TOTAL)
    return false;

  // Store the upper left corner coordinate of the rectangle (ROI)
  templateROIX = templateCorners[0].x, templateROIY = templateCorners[0].y;

  const CvSize templateSize = cvSize(templateWidth, templateHeight);
  const CvRect templateRect = cvRect(templateCorners[0].x, templateCorners[0].y, 
				     templateWidth, templateHeight);
  
  // Store the original version of the new template(all image)
  cvCopyImage(newFrameRGB, templateImageRGBFull);

  cvReleaseImage(&templateImageRGB);
  templateImageRGB = cvCreateImage(templateSize, IPL_DEPTH_8U, 3);

  cvReleaseImage(&templateImageGray);
  templateImageGray = cvCreateImage(templateSize, IPL_DEPTH_8U, 1);

  // Store a Grayscale version of the new template(only ROI)
  cvSetImageROI(newFrameGray, templateRect);
  cvCopyImage(newFrameGray, templateImageGray);
  cvResetImageROI(newFrameGray);


  // Store an RGB version of the new template(only ROI)
  cvSetImageROI(newFrameRGB, templateRect);
  cvCopyImage(newFrameRGB, templateImageRGB);
  cvResetImageROI(newFrameRGB);

  saveCornerCoors();

  return true;
}

// Finds the coordinates of the corners of the template image after scaling and rotation applied
void estimateCornerCoordinatesOfNewTemplate(int scaleInd, int rotInd, float scale, float angle)
{
  static double* corners = templateObjectCornersData[0][0];

  // Center of the original image
  const float orgCenterX = templateImageGray->width / 2.0, orgCenterY = templateImageGray->height / 2.0;
  // Center of the scaled image
  const float centerX = orgCenterX * scale, centerY = orgCenterY * scale;

  const float cosAngle = cos(degreeToRadian(angle));
  const float sinAngle = sin(degreeToRadian(angle));

  for (int xCoor = 0, yCoor = 1; xCoor < 8; xCoor += 2, yCoor += 2) {
    // Scale the point and translate it so that the origin is in the middle of the image
    const float resizedAndTranslatedX = (corners[xCoor] * scale) - centerX,
      resizedAndTranslatedY = (corners[yCoor] * scale) - centerY;

    // Rotate the point with the given angle
    templateObjectCornersData[scaleInd][rotInd][xCoor] =
      (resizedAndTranslatedX * cosAngle - resizedAndTranslatedY * sinAngle) + centerX;
    templateObjectCornersData[scaleInd][rotInd][yCoor] =
      (resizedAndTranslatedX * sinAngle + resizedAndTranslatedY * cosAngle) + centerY;
  }
}

// Generates new templates with different scales and orientations and stores their keypoints and
// BRIEF descriptors.
void learnTemplate(void)
{
  static const float ROT_ANGLE_INCREMENT = 360.0 / NUMBER_OF_ROTATION_STEPS;
  static const float k = exp(log(SMALLEST_SCALE_CHANGE) / (NUMBER_OF_SCALE_STEPS - 1));

  // Estimate a feasible threshold value for FAST keypoint detector
  fastThreshold = chooseFASTThreshold(templateImageGray, 200, 250);

  // For every scale generate templates
  for (int scaleInd = 0; scaleInd < NUMBER_OF_SCALE_STEPS; ++scaleInd) {
    // Calculate the template size in a log basis
    const float currentScale = pow(k, scaleInd);

    // Scale the image
    IplImage* scaledTemplateImg = cvCreateImage(cvSize(templateImageGray->width * currentScale,
						       templateImageGray->height * currentScale),
						IPL_DEPTH_8U, 1);
    cvResize(templateImageGray, scaledTemplateImg);

    const CvPoint2D32f center = cvPoint2D32f(scaledTemplateImg->width >> 1, scaledTemplateImg->height >> 1);

    // For a given scale, generate templates with several rotations
    float currentAngle = 0.0;
    for (int rotInd = 0; rotInd < NUMBER_OF_ROTATION_STEPS; ++rotInd, currentAngle += ROT_ANGLE_INCREMENT) {
      // Rotate the image
      IplImage* rotatedImage = cvCreateImage(cvGetSize(scaledTemplateImg),
					     scaledTemplateImg->depth,
					     scaledTemplateImg->nChannels);
      rotateImage(rotatedImage, scaledTemplateImg, center, -currentAngle);

      // Detect FAST keypoints
      extractKeypoints(templateKpts[scaleInd][rotInd], fastThreshold, rotatedImage);

      // Describe the keypoints with BRIEF descriptors
      brief.getBriefDescriptors(templateDescs[scaleInd][rotInd],
				templateKpts[scaleInd][rotInd],
				rotatedImage);

      // Store the scaled and rotated template corner coordinates
      estimateCornerCoordinatesOfNewTemplate(scaleInd, rotInd, currentScale, currentAngle);

      cvReleaseImage(&rotatedImage);
    }
    cvReleaseImage(&scaledTemplateImg);
  }
}

// Manages the capture of the new template image according to the points picked by the user
void takeNewTemplateImage(void)
{
  cvCopyImage(newFrameRGB, outputImage);
  switch (noOfPointsPickedOnTemplateImage) {
  case 1:
    cvRectangle(outputImage, templateCorners[0], templateCorners[1], cvScalar(0, 255, 0), 3);
    break;
  case 2:
    if (saveNewTemplate()) {
      learnTemplate();
      appMode = DETECTION;
      hasAnyTemplateChosenBefore = true;
    }
    noOfPointsPickedOnTemplateImage = 0;
    break;
  default:
    break;
  }
}

// Matches Brief descriptors descs1 and descs2 in terms of Hamming Distance.
int matchDescriptors(
		     CvMat& match1, CvMat& match2,
		     const vector< bitset<CVLAB::DESC_LEN> > descs1,
		     const vector< bitset<CVLAB::DESC_LEN> > descs2,
		     const vector<cv::KeyPoint>& kpts1,
		     const vector<cv::KeyPoint>& kpts2)
{
  // Threshold value for matches.
  static const int MAX_MATCH_DISTANCE = 50;

  int numberOfMatches = 0;
  // Index of the best BRIEF descriptor match on descs2
  int bestMatchInd2 = 0;
  
  // For every BRIEF descriptor in descs1 find the best fitting BRIEF descriptor in descs2
  for (unsigned int i = 0; i < descs1.size() && numberOfMatches < MAXIMUM_NUMBER_OF_MATCHES; ++i) {
    int minDist = CVLAB::DESC_LEN;
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < descs2.size(); ++j) {
	    const int dist = CVLAB::HAMMING_DISTANCE(descs1[i], descs2[j]);
      
      // If dist is less than the optimum one observed so far, the new optimum one is current BRIEF descriptor
      if (dist < minDist) {
	minDist = dist;
	bestMatchInd2 = j;
      }
    }
    // If the Hamming Distance is greater than the threshold, ignore this match
    if (minDist > MAX_MATCH_DISTANCE)
      continue;

    // Save the matching keypoint coordinates
    const int xInd = 2 * numberOfMatches;
    const int yInd = xInd + 1;

    match1Data[xInd] = kpts1[i].pt.x;
    match1Data[yInd] = kpts1[i].pt.y;

    match2Data[xInd] = kpts2[bestMatchInd2].pt.x;
    match2Data[yInd] = kpts2[bestMatchInd2].pt.y;
    
    numberOfMatches++;
  }

  if (numberOfMatches > 0) {
    cvInitMatHeader(&match1, numberOfMatches, 2, CV_64FC1, match1Data);
    cvInitMatHeader(&match2, numberOfMatches, 2, CV_64FC1, match2Data);
  }

  return numberOfMatches;
}

// Initializes the application
int init(void)
{
	
	capture = cvCaptureFromCAM(0); // capture from video device #0
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
	if(firstTime){
	// Memory Allocations
	newFrameGray = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 1);
	outputImage = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
	templateImageRGBFull = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
	sideBySideImage = cvCreateImage(cvSize(2 * FRAME_WIDTH, FRAME_HEIGHT), IPL_DEPTH_8U, 3);
	templateImageRGB = cvCreateImage(cvSize(1, 1), IPL_DEPTH_8U, 3);
	templateImageGray = cvCreateImage(cvSize(1, 1), IPL_DEPTH_8U, 1);
	for (int s = 0; s < NUMBER_OF_SCALE_STEPS; s++) {
		for (int r = 0; r < NUMBER_OF_ROTATION_STEPS; r++) {
		templateObjectCornersData[s][r] = new double[8];
		templateObjectCorners[s][r] = cvMat(1, 4, CV_64FC2, templateObjectCornersData[s][r]);
		}
	}
	firstTime = false;
	}
    const string inputSettingsFile = "out_camera_data.xml"; // load camera parameters
    FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the parameters
    if (!fs.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
		cout << "Please run the camera calibration first" << endl;
		appMode = DO_NOTHING_CALIB;
        return 0;
    }
    fs["Camera_Matrix"] >> calibrationMatrix; // Intrincis camera parameters 
	fs["Distortion_Coefficients"] >> calibrationDistortionCoefficients; // Camera distortion coefficients
    fs.release();                               



  // Seed the random number generator
  srand(time(NULL));

  // In the beginning, only capture a frame and show it to the user
  appMode = DO_NOTHING;

 

  cvNamedWindow("BRIEF", CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback("BRIEF", mouseHandler, NULL);

  cvInitFont(&font, CV_FONT_HERSHEY_PLAIN | CV_FONT_ITALIC, 1, 1, 0, 1);
  return 0;
}

// Detects the template object in the incoming frame
void doDetection(void)
{
  // Variables for elapsed time estimations
  static int64 startTime, endTime;

  // Homography Matrix
  static CvMat* H = cvCreateMat(3, 3, CV_64FC1);

  // Corners of the detected planar object
  static double detectedObjCornersData[8];
  static CvMat detectedObjCorners = cvMat(1, 4, CV_64FC2, detectedObjCornersData);
  
	   
  // Keypoints of the incoming frame
  vector<cv::KeyPoint> kpts;
  
  // BRIEF descriptors of kpts
  vector< bitset<CVLAB::DESC_LEN> > descs;

  // Coordinates of the matching keypoints
  CvMat match1, match2;

  float maxRatio = 0.0;
  int maxScaleInd = 0;
  int maxRotInd = 0;
  int maximumNumberOfMatches = 0;

  // If !doDBSearch then only try to match the original template image
  const int dbScaleSz = doDBSearch ? NUMBER_OF_SCALE_STEPS : 1;
  const int dbRotationSz = doDBSearch ? NUMBER_OF_ROTATION_STEPS : 1;

  startTime = cvGetTickCount();
  // Detect the FAST keypoints of the incoming frame
  extractKeypoints(kpts, fastThreshold, newFrameGray);
  endTime = cvGetTickCount();
  keypointExtractTime = endTime - startTime;


  startTime = cvGetTickCount();
  // Describe the keypoints with BRIEF descriptors
  brief.getBriefDescriptors(descs, kpts, newFrameGray);
  endTime = cvGetTickCount();
  bitDescTime = endTime - startTime;

  startTime = cvGetTickCount();
  // Search through all the templates
  #pragma omp parallel
  #pragma omp for
  for (int scaleInd = 0; scaleInd < dbScaleSz; ++scaleInd) {
     for (int rotInd = 0; rotInd < dbRotationSz; ++rotInd) {
      const int numberOfMatches = matchDescriptors(match1, match2,
						   templateDescs[scaleInd][rotInd], descs,
						   templateKpts[scaleInd][rotInd], kpts);

      // Since RANSAC needs at least 4 points, ignore this match
      if (numberOfMatches < 4)
	continue;

      // Save the matrix index of the best fitting template to the incoming frame
      const float currentRatio = float(numberOfMatches) / templateKpts[scaleInd][rotInd].size();
      if (currentRatio > maxRatio) {
	maxRatio = currentRatio;
	maxScaleInd = scaleInd;
	maxRotInd = rotInd;
	static const float k = exp(log(SMALLEST_SCALE_CHANGE) / (NUMBER_OF_SCALE_STEPS - 1));
    const float scale = pow(k, scaleInd);
	present_scale = scale;
	maximumNumberOfMatches = numberOfMatches;
      }
    }
  }
  endTime = cvGetTickCount();
  totalMatchTime = endTime - startTime;


  matchPercentage = int(maxRatio * 100.0);

  if (maximumNumberOfMatches > 3) {
    startTime = cvGetTickCount();
    // Match the best fitting template's BRIEF descriptors with the incoming frame
    matchDescriptors(match1, match2,
		     templateDescs[maxScaleInd][maxRotInd], descs,
		     templateKpts[maxScaleInd][maxRotInd], kpts);
    endTime = cvGetTickCount();
    matchTime = endTime - startTime;
	startTime = cvGetTickCount();
    // Calculate the homography matrix via RANSAC
    cvFindHomography(&match1, &match2, H, CV_RANSAC, 10, 0);

    // If H is not a feasible homography matrix, ignore it
    if (niceHomography(H)) {
      startTime = cvGetTickCount();
      // Transform the coordinates of the corners of the template into image coordinates
      cvPerspectiveTransform(&templateObjectCorners[maxScaleInd][maxRotInd], &detectedObjCorners, H);
      endTime = cvGetTickCount();
      hmgEstTime = endTime - startTime;

      // Draw the detected object on the image
      markDetectedObject(sideBySideImage, detectedObjCornersData);

      // Scale and rotate the coordinates of the template keypoints to transform them into the original
      // template image's coordinates to show the matches
      transformPointsIntoOriginalImageCoordinates(maximumNumberOfMatches, maxScaleInd, maxRotInd);
	  

	  // Finding the extrinsic parameters in order to represent 3D coordinates on 2D
	typedef double precision;
	int precisionType = CV_64FC1;

	Mat inverseCalibrationMatrix = calibrationMatrix.inv(DECOMP_SVD);
	
	// Split the homography matrix into 3 vectors.
	double h1A[3][1] = {{cvGetReal2D(H,0,0)} , {cvGetReal2D(H,1,0)} , {cvGetReal2D(H,2,0)}};
	Mat h1(3, 1, precisionType, h1A);
	
	double h2A[3][1] = {{cvGetReal2D(H,0,1)} , {cvGetReal2D(H,1,1)} , {cvGetReal2D(H,2,1)}};
	Mat h2(3, 1, precisionType, h2A);

	double h3A[3][1] = {{cvGetReal2D(H,0,2)} , {cvGetReal2D(H,1,2)} , {cvGetReal2D(H,2,2)}};
	Mat h3(3, 1, precisionType, h3A);
	
	// Remove the calibration paramaters before computing the scale length.
	Mat scaleVector = inverseCalibrationMatrix * h1;

	// Calculate the the length of H1 for normalising.
	double scale1 = sqrt(scaleVector.at<precision>(0,0)*scaleVector.at<precision>(0,0) +
						scaleVector.at<precision>(1,0)*scaleVector.at<precision>(1,0) +
						scaleVector.at<precision>(2,0)*scaleVector.at<precision>(2,0));

	if(scale1 != 0)
	{
		scale1 = 1/scale1;

		// Normalise the inverseCalibrationMatrix
		inverseCalibrationMatrix = inverseCalibrationMatrix * scale1;

		// Remove the calibration paramaters from the translation.
		tvec = inverseCalibrationMatrix * h3;

		Mat r1 = inverseCalibrationMatrix * h1;
		Mat r2 = inverseCalibrationMatrix * h2;
		Mat r3 = r1.cross(r2);			// Find the vector perpendicular (orthogonal) to the other 2 rotation vectors.
   
		precision rotationMatrixA[3][3] = {{r1.at<precision>(0,0) , r2.at<precision>(0,0) , r3.at<precision>(0,0)},
												  {r1.at<precision>(1,0) , r2.at<precision>(1,0) , r3.at<precision>(1,0)},
												  {r1.at<precision>(2,0) , r2.at<precision>(2,0) , r3.at<precision>(2,0)}};
		Mat rotationMatrix(3, 3, precisionType, rotationMatrixA);
	
		// Enforce RT*R = R*RT = I (Where RT is R transpose), by setting D to I (I = identity matrix).
		SVD decomposed(rotationMatrix);
		rotation = decomposed.u * decomposed.vt;
	}
	else 
	{
		// Can't divide by a zero.
		tvec = Mat::zeros(3, 1, precisionType) ;
		rotation = Mat::eye(3, 3, precisionType) ;
	}
	Rodrigues(rotation, rvec);
	Mat temp = cv::Mat(&templateObjectCorners[maxScaleInd][maxRotInd], true);
	float xPoint = ((temp.at<double>(0,0)+temp.at<double>(0,2)+temp.at<double>(0,4)+temp.at<double>(0,6)))/4;
	float yPoint = ((temp.at<double>(0,1)+temp.at<double>(0,3)+temp.at<double>(0,5)+temp.at<double>(0,7)))/4;
	framePoints.clear();
	//Coordinates for 3D axes with origin at the center of the marked object
	framePoints.push_back( Point3d( xPoint, yPoint, 0.0 ) );
	framePoints.push_back( Point3d( (xPoint)+100.0, yPoint, 0.0 ) );
	framePoints.push_back( Point3d( xPoint, (yPoint)+100.0, 0.0 ) );
	framePoints.push_back( Point3d( xPoint, yPoint, -100.0 ) );
	//Coordinates of the 4 corners of the object
	framePoints.push_back( Point3d( temp.at<double>(0,0), temp.at<double>(0,1), 0.0 ) );
	framePoints.push_back( Point3d( temp.at<double>(0,2), temp.at<double>(0,3), 0.0 ) );
	framePoints.push_back( Point3d( temp.at<double>(0,4), temp.at<double>(0,5), 0.0 ) );
	framePoints.push_back( Point3d( temp.at<double>(0,6), temp.at<double>(0,7), 0.0 ) );
	framePoints.push_back( Point3d( xPoint, yPoint, 100.0 ) );
	projectPoints(framePoints, rvec/present_scale, tvec/present_scale, calibrationMatrix, calibrationDistortionCoefficients, imageFramePoints );
	printZ = true;
	
	line((Mat)sideBySideImage, imageFramePoints[0], imageFramePoints[1], CV_RGB(255,0,0), 2 );
	line((Mat)sideBySideImage, imageFramePoints[0], imageFramePoints[2], CV_RGB(0,255,0), 2 );
	line((Mat)sideBySideImage, imageFramePoints[0], imageFramePoints[3], CV_RGB(0,0,255), 2 );
	circle((Mat)sideBySideImage, imageFramePoints[0], 10, CV_RGB(0,255,255));
	circle((Mat)sideBySideImage, imageFramePoints[4], 10, CV_RGB(0,0,255));
	circle((Mat)sideBySideImage, imageFramePoints[5], 10, CV_RGB(0,0,255));
	circle((Mat)sideBySideImage, imageFramePoints[6], 10, CV_RGB(0,255,255));
	circle((Mat)sideBySideImage, imageFramePoints[7], 10, CV_RGB(0,255,255));
	//Represents a point in the center if the object is centered. 
	if(cv::norm(imageFramePoints[0]-imageFramePoints[3])<10)
	 {
		 isCentered = true;
		 circle((Mat)sideBySideImage, imageFramePoints[0], 2, CV_RGB(100,100,255),2);
	 }
	 else
		 isCentered = false;
	
	// Conversion of frame pixel values to real world distance and values in millimeters
	Mat worldValues, homographyInverse;
	vector<Point2f> vert;
	vector<Point3f> pixelValues;
	vert.push_back(Point(320, 240));
	vert.push_back(2*imageFramePoints[0]-imageFramePoints[3]);
	
	convertPointsToHomogeneous(vert, pixelValues);
	char text1[256];
	homographyInverse = (Mat)H;
	homographyInverse = homographyInverse.inv(DECOMP_SVD);
	Mat pixelValue = (Mat_<double>(3,2) << pixelValues[0].x, pixelValues[1].x,
		pixelValues[0].y, pixelValues[1].y, pixelValues[0].z, pixelValues[1].z);

	//Scaling required if position of Z on the machine is varied.
	//scalingFactor depends on the height(z value)
	double scalingFactor =zInitialCoordinates/(zInitialCoordinates+zPresentCoordinates);

	//Technically PixelValues (x,y,1)' = Homography*(X,Y,Z)/scalingFactor
	//Rearranging the above we can get the world coordinates.
	worldValues = homographyInverse*pixelValue*scalingFactor; 
	
	sprintf(text1, "Camera Centre: %f, %f, %f", worldValues.at<double>(0,0), worldValues.at<double>(1,0), worldValues.at<double>(2,0));
    cvPutText(sideBySideImage, text1, cvPoint(10, 250), &font, cvScalar(255, 0, 0));
	
	sprintf(text1, "Object Centre %f, %f, %f", worldValues.at<double>(0,1), worldValues.at<double>(1,1), worldValues.at<double>(2,1));
    cvPutText(sideBySideImage, text1, cvPoint(10, 270), &font, cvScalar(255, 0, 0));
	
	sprintf(text1, "PixelValue: %f, %f", (worldValues.at<double>(0,0)-worldValues.at<double>(0,1))/2, (worldValues.at<double>(1,1)-worldValues.at<double>(1,0))/2);
    cvPutText(sideBySideImage, text1, cvPoint(10, 290), &font, cvScalar(255, 0, 0));
	
	endTime = cvGetTickCount();
    AxesTime = endTime - startTime;
    }
  }

  // Mark the keypoints detected with plus signs:
  //showKeypoints(sideBySideImage, kpts);

  // Indicate the matches with lines:
  //showMatches(maximumNumberOfMatches);
}



// Main loop of the program
void run(void)
{
  while (true) {
    IplImage* result = outputImage;

    fpsCalculation();

    switch (appMode) {
    case TAKING_NEW_TEMPLATE:
      takeNewTemplateImage();
      break;
    case DETECTION:
      takeNewFrame();
      cvCopyImage(newFrameRGB, outputImage);
      putImagesSideBySide(sideBySideImage, newFrameRGB, templateImageRGBFull);
      doDetection();
      result = sideBySideImage;
      break;
    case DO_NOTHING:
      takeNewFrame();
      cvCopyImage(newFrameRGB, outputImage);
      break;
    case DO_NOTHING_CALIB:
      takeNewFrame();
	  cameraCalib();
      cvCopyImage(newFrameRGB, outputImage);
	  break;
    case END:
      return;
    default:
      break;
    }
    showOutput(result);
    waitKeyAndHandleKeyboardInput(10);
  }
}

//help to start camera calibration
static void help_camera_calib()
{
    cout <<  "This is a camera calibration section." << endl
         <<  "Usage: calibration configurationFile"  << endl
         <<  "Near this file you'll find the configuration file, which has detailed help of "
             "how to edit it.  It may be any OpenCV supported file format XML/YAML." << endl;
}

class Settings
{
public:
    Settings() : goodInput(false) {}
    enum Pattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
    enum InputType {INVALID, CAMERA, VIDEO_FILE, IMAGE_LIST};

    void write(FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "{" << "BoardSize_Width"  << boardSize.width
                  << "BoardSize_Height" << boardSize.height
                  << "Square_Size"         << squareSize
                  << "Calibrate_Pattern" << patternToUse
                  << "Calibrate_NrOfFrameToUse" << nrFrames
                  << "Calibrate_FixAspectRatio" << aspectRatio
                  << "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
                  << "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint

                  << "Write_DetectedFeaturePoints" << bwritePoints
                  << "Write_extrinsicParameters"   << bwriteExtrinsics
                  << "Write_outputFileName"  << outputFileName

                  << "Show_UndistortedImage" << showUndistorsed

                  << "Input_FlipAroundHorizontalAxis" << flipVertical
                  << "Input_Delay" << delay
                  << "Input" << input
           << "}";
    }
    void read(const FileNode& node)                          //Read serialization for this class
    {
        node["BoardSize_Width" ] >> boardSize.width;
        node["BoardSize_Height"] >> boardSize.height;
        node["Calibrate_Pattern"] >> patternToUse;
        node["Square_Size"]  >> squareSize;
        node["Calibrate_NrOfFrameToUse"] >> nrFrames;
        node["Calibrate_FixAspectRatio"] >> aspectRatio;
        node["Write_DetectedFeaturePoints"] >> bwritePoints;
        node["Write_extrinsicParameters"] >> bwriteExtrinsics;
        node["Write_outputFileName"] >> outputFileName;
        node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
        node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
        node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
        node["Show_UndistortedImage"] >> showUndistorsed;
        node["Input"] >> input;
        node["Input_Delay"] >> delay;
        interprate();
    }
    void interprate()
    {
        goodInput = true;
        if (boardSize.width <= 0 || boardSize.height <= 0)
        {
            cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << endl;
            goodInput = false;
        }
        if (squareSize <= 10e-6)
        {
            cerr << "Invalid square size " << squareSize << endl;
            goodInput = false;
        }
        if (nrFrames <= 0)
        {
            cerr << "Invalid number of frames " << nrFrames << endl;
            goodInput = false;
        }

        if (input.empty())      // Check for valid input
                inputType = INVALID;
        else
        {
            if (input[0] >= '0' && input[0] <= '9')
            {
                stringstream ss(input);
                ss >> cameraID;
                inputType = CAMERA;
            }
            else
            {
                if (readStringList(input, imageList))
                    {
                        inputType = IMAGE_LIST;
                        nrFrames = (nrFrames < (int)imageList.size()) ? nrFrames : (int)imageList.size();
                    }
                else
                    inputType = VIDEO_FILE;
            }
            if (inputType == CAMERA)
                inputCapture.open(cameraID);
            if (inputType == VIDEO_FILE)
                inputCapture.open(input);
            if (inputType != IMAGE_LIST && !inputCapture.isOpened())
                    inputType = INVALID;
        }
        if (inputType == INVALID)
        {
            cerr << " Inexistent input: " << input;
            goodInput = false;
        }

        flag = 0;
        if(calibFixPrincipalPoint) flag |= CV_CALIB_FIX_PRINCIPAL_POINT;
        if(calibZeroTangentDist)   flag |= CV_CALIB_ZERO_TANGENT_DIST;
        if(aspectRatio)            flag |= CV_CALIB_FIX_ASPECT_RATIO;


        calibrationPattern = NOT_EXISTING;
        if (!patternToUse.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
        if (!patternToUse.compare("CIRCLES_GRID")) calibrationPattern = CIRCLES_GRID;
        if (!patternToUse.compare("ASYMMETRIC_CIRCLES_GRID")) calibrationPattern = ASYMMETRIC_CIRCLES_GRID;
        if (calibrationPattern == NOT_EXISTING)
            {
                cerr << " Inexistent camera calibration mode: " << patternToUse << endl;
                goodInput = false;
            }
        atImageList = 0;

    }
    Mat nextImage()
    {
        Mat result;
        if( inputCapture.isOpened() )
        {
            Mat view0;
            inputCapture >> view0;
            view0.copyTo(result);
        }
        else if( atImageList < (int)imageList.size() )
            result = imread(imageList[atImageList++], CV_LOAD_IMAGE_COLOR);

        return result;
    }

    static bool readStringList( const string& filename, vector<string>& l )
    {
        l.clear();
        FileStorage fs(filename, FileStorage::READ);
        if( !fs.isOpened() )
            return false;
        FileNode n = fs.getFirstTopLevelNode();
        if( n.type() != FileNode::SEQ )
            return false;
        FileNodeIterator it = n.begin(), it_end = n.end();
        for( ; it != it_end; ++it )
            l.push_back((string)*it);
        return true;
    }
public:
    Size boardSize;            // The size of the board -> Number of items by width and height
    Pattern calibrationPattern;// One of the Chessboard, circles, or asymmetric circle pattern
    float squareSize;          // The size of a square in your defined unit (point, millimeter,etc).
    int nrFrames;              // The number of frames to use from the input for calibration
    float aspectRatio;         // The aspect ratio
    int delay;                 // In case of a video input
    bool bwritePoints;         //  Write detected feature points
    bool bwriteExtrinsics;     // Write extrinsic parameters
    bool calibZeroTangentDist; // Assume zero tangential distortion
    bool calibFixPrincipalPoint;// Fix the principal point at the center
    bool flipVertical;          // Flip the captured images around the horizontal axis
    string outputFileName;      // The name of the file where to write
    bool showUndistorsed;       // Show undistorted images after calibration
    string input;               // The input ->



    int cameraID;
    vector<string> imageList;
    int atImageList;
    VideoCapture inputCapture;
    InputType inputType;
    bool goodInput;
    int flag;

private:
    string patternToUse;


};

static void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

enum { DETECTION1 = 0, CAPTURING = 1, CALIBRATED = 2 };

bool runCalibrationAndSave(Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs,
                           vector<vector<Point2f> > imagePoints );


int cameraCalib()
{
    help_camera_calib();
    Settings s;
    const string inputSettingsFile = "cameracalib.xml";
    FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
		help_camera_calib();
        return -1;
    }
    fs["Settings"] >> s;
    fs.release();                                         // close Settings file

    if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        return -1;
    }

    vector<vector<Point2f> > imagePoints;
    Mat cameraMatrix, distCoeffs;
    Size imageSize;
    int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION1;
    clock_t prevTimestamp = 0;
    const Scalar RED(0,0,255), GREEN(0,255,0);
    const char ESC_KEY = 27;

    for(int i = 0;;++i)
    {
      Mat view;
      bool blinkOutput = false;

      view = s.nextImage();

      //-----  If no more image, or got enough, then stop calibration and show result -------------
      if( mode == CAPTURING && imagePoints.size() >= (unsigned)s.nrFrames )
      {
          if( runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints))
              mode = CALIBRATED;
          else
              mode = DETECTION1;
      }
      if(view.empty())          // If no more images then run calibration, save and stop loop.
      {
            if( imagePoints.size() > 0 )
                runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints);
            break;
      }


        imageSize = view.size();  // Format input image.
        if( s.flipVertical )    flip( view, view, 0 );

        vector<Point2f> pointBuf;

        bool found;
        switch( s.calibrationPattern ) // Find feature points on the input format
        {
        case Settings::CHESSBOARD:
            found = findChessboardCorners( view, s.boardSize, pointBuf,
                CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
            break;
        case Settings::CIRCLES_GRID:
            found = findCirclesGrid( view, s.boardSize, pointBuf );
            break;
        case Settings::ASYMMETRIC_CIRCLES_GRID:
            found = findCirclesGrid( view, s.boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID );
            break;
        default:
            found = false;
            break;
        }

        if ( found)                // If done with success,
        {
              // improve the found corners' coordinate accuracy for chessboard
                if( s.calibrationPattern == Settings::CHESSBOARD)
                {
                    Mat viewGray;
                    cvtColor(view, viewGray, COLOR_BGR2GRAY);
                    cornerSubPix( viewGray, pointBuf, Size(11,11),
                        Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
                }

                if( mode == CAPTURING &&  // For camera only take new samples after delay time
                    (!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC) )
                {
                    imagePoints.push_back(pointBuf);
                    prevTimestamp = clock();
                    blinkOutput = s.inputCapture.isOpened();
                }

                // Draw the corners.
                drawChessboardCorners( view, s.boardSize, Mat(pointBuf), found );
        }

        //----------------------------- Output Text ------------------------------------------------
        string msg = (mode == CAPTURING) ? "100/100" :
                      mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        if( mode == CAPTURING )
        {
            if(s.showUndistorsed)
                msg = format( "%d/%d Undist", (int)imagePoints.size(), s.nrFrames );
            else
                msg = format( "%d/%d", (int)imagePoints.size(), s.nrFrames );
        }

        putText( view, msg, textOrigin, 1, 1, mode == CALIBRATED ?  GREEN : RED);

        if( blinkOutput )
            bitwise_not(view, view);

        //------------------------- Video capture  output  undistorted ------------------------------
        if( mode == CALIBRATED && s.showUndistorsed )
        {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);
        }

        //------------------------------ Show image and check for input commands -------------------
        imshow("Calibration Window", view);
        char key = (char)waitKey(s.inputCapture.isOpened() ? 50 : s.delay);

        if( key  == ESC_KEY )
		{
			s.inputCapture.release();
			destroyWindow("Calibration Window");
			init();
            break;
		}
        if( key == 'u' && mode == CALIBRATED )
           s.showUndistorsed = !s.showUndistorsed;

        if( s.inputCapture.isOpened() && key == 'g' )
        {
            mode = CAPTURING;
            imagePoints.clear();
        }
    }

    // -----------------------Show the undistorted image for the image list ------------------------
    if( s.inputType == Settings::IMAGE_LIST && s.showUndistorsed )
    {
        Mat view, rview, map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
            imageSize, CV_16SC2, map1, map2);

        for(int i = 0; i < (int)s.imageList.size(); i++ )
        {
            view = imread(s.imageList[i], 1);
            if(view.empty())
                continue;
            remap(view, rview, map1, map2, INTER_LINEAR);
            imshow("Calibration Window", rview);
            char c = (char)waitKey();
            if( c  == ESC_KEY || c == 'q' || c == 'Q' )
			{	
				s.inputCapture.release();
				destroyWindow("Calibration Window");
				init();
                break;
			}
        }
    }


    return 0;
}

int main(void)
{
  init();

  cout << "Press:" << endl;
  cout << " 'c' to calibrate the camera;" << endl;
  cout << " 't' to capture a new template;" << endl;
  cout << " 'd' to enable/disable scale and rotation invariance;" << endl;
  cout << " 'q' to quit." << endl;
  cout << "Enter the current z coordinates: ";
  cin >> zPresentCoordinates;
  run();

  return 0;
}

static double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<Point2f> >& imagePoints,
                                         const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                         const Mat& cameraMatrix , const Mat& distCoeffs,
                                         vector<float>& perViewErrors)
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); ++i )
    {
        projectPoints( Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                       distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}


static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
                                     Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
{
    corners.clear();

    switch(patternType)
    {
    case Settings::CHESSBOARD:
    case Settings::CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; ++i )
            for( int j = 0; j < boardSize.width; ++j )
                corners.push_back(Point3f(float( j*squareSize ), float( i*squareSize ), 0));
        break;

    case Settings::ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
        break;
    default:
        break;
    }
}


static bool runCalibration( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                            vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
                            vector<float>& reprojErrs,  double& totalAvgErr)
{

    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( s.flag & CV_CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = 1.0;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                 distCoeffs, rvecs, tvecs, s.flag|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

    cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                             rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

// Print camera parameters to the output file
static void saveCameraParams( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                              const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                              const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
                              double totalAvgErr )
{
    FileStorage fs( s.outputFileName, FileStorage::WRITE );

    time_t tm;
    time( &tm );
    struct tm *t2 = localtime( &tm );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_Time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nrOfFrames" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_Width" << imageSize.width;
    fs << "image_Height" << imageSize.height;
    fs << "board_Width" << s.boardSize.width;
    fs << "board_Height" << s.boardSize.height;
    fs << "square_Size" << s.squareSize;

    if( s.flag & CV_CALIB_FIX_ASPECT_RATIO )
        fs << "FixAspectRatio" << s.aspectRatio;

    if( s.flag )
    {
        sprintf( buf, "flags: %s%s%s%s",
            s.flag & CV_CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "",
            s.flag & CV_CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "",
            s.flag & CV_CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "",
            s.flag & CV_CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "" );
        cvWriteComment( *fs, buf, 0 );

    }

    fs << "flagValue" << s.flag;

    fs << "Camera_Matrix" << cameraMatrix;
    fs << "Distortion_Coefficients" << distCoeffs;

    fs << "Avg_Reprojection_Error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "Per_View_Reprojection_Errors" << Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "Extrinsic_Parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "Image_points" << imagePtMat;
    }
}

bool runCalibrationAndSave(Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs,vector<vector<Point2f> > imagePoints )
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(s,imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
                             reprojErrs, totalAvgErr);
    cout << (ok ? "Calibration succeeded" : "Calibration failed")
        << ". avg re projection error = "  << totalAvgErr ;

    if( ok )
        saveCameraParams( s, imageSize, cameraMatrix, distCoeffs, rvecs ,tvecs, reprojErrs,
                            imagePoints, totalAvgErr);
    return ok;
}


